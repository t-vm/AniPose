"""
coco_converter.py
-----------------
COCO格式数据转换工具

支持两种调用方式：

方式一（新）：传入数据类对象（推荐）
    from coco_converter import convert_from_dataset, convert_from_samples

    # 从完整数据集对象导出，按数据集自身的 split 字段分组
    convert_from_dataset(dataset, output_dir="annotations/")

    # 传入任意 AnimeSkeletonData 列表，手动指定输出文件
    convert_from_samples(samples, output_path="annotations/train.json")

方式二（兼容旧版）：扫描文件系统（保留原有行为）
    from coco_converter import convert_and_split_data

    convert_and_split_data(model_dir="path/to/68 model", ...)

数据流说明：
    原始数据
        └─ AnimeSkeletonData (pose_format="mixamo")
               └─ .to_coco17()
                      └─ AnimeSkeletonData (pose_format="coco17")
                             └─ .save_json()  →  9.2.1 中间 JSON
                                    └─ convert_from_samples() / convert_and_split_data()
                                           └─ 9.2.2 可训练 COCO JSON
"""

from __future__ import annotations

import json
import glob
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    # 避免循环导入；运行时不强制依赖 anime_dataset 模块
    from anime_dataset import AnimePoseDataset, AnimeSkeletonData


# ─────────────────────────────────────────────
# COCO-17 标准定义
# ─────────────────────────────────────────────

COCO17_KEYPOINT_NAMES: list[str] = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# COCO 官方 skeleton（1-indexed）
COCO17_SKELETON: list[list[int]] = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]


# ─────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────

def _get_image_size(img_path: str | Path) -> Optional[tuple[int, int]]:
    """返回 (width, height)，失败返回 None。"""
    try:
        with Image.open(img_path) as img:
            return img.size
    except Exception:
        return None


def _find_render_image(
    render_dir: str | Path,
    frame_id: int,
    view_index: int,
) -> tuple[Optional[Path], Optional[str]]:
    """
    在渲染目录中定位帧图像。
    文件名格式：{frame_id:04d}_{view_index}.png / .jpg
    返回 (绝对路径, 文件名)，找不到时返回 (None, None)。
    """
    render_dir = Path(render_dir)
    cam_id = str(view_index)
    exts = [".png", ".jpg", ".jpeg"]

    # 优先匹配标准命名
    for ext in exts:
        name = f"{frame_id:04d}_{cam_id}{ext}"
        p = render_dir / name
        if p.exists():
            return p, name

    # 回退：扫描目录，按帧号匹配
    if render_dir.is_dir():
        for f in sorted(render_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                digits = "".join(ch for ch in f.stem if ch.isdigit())
                if digits and int(digits) == frame_id:
                    return f, f.name

    return None, None


def _build_coco_template(
    keypoint_names: list[str],
    skeleton: list[list[int]],
) -> dict:
    """构造空的 COCO 标注容器。"""
    return {
        "info": {
            "description": "AnimePose2D COCO Format",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": keypoint_names,
            "skeleton": skeleton,
        }],
    }


def _write_coco_json(
    records: list[dict],
    template: dict,
    output_path: str | Path,
) -> Path:
    """将记录列表写入 COCO JSON 文件，返回写入路径。"""
    coco = json.loads(json.dumps(template))   # 深拷贝
    for r in records:
        coco["images"].append({
            "id": r["id"],
            "file_name": r["file_name"],
            "height": r["height"],
            "width": r["width"],
        })
        coco["annotations"].append({
            "id": r["ann_id"],
            "image_id": r["id"],
            "category_id": 1,
            "bbox": r["bbox"],
            "area": r["area"],
            "keypoints": r["keypoints"],
            "num_keypoints": r["num_keypoints"],
            "iscrowd": 0,
            "segmentation": [],
        })
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    return output_path


def _sample_to_record(
    sample: "AnimeSkeletonData",
    img_id: int,
    ann_id: int,
    dataset_root: Optional[Path],
    keypoint_names: list[str],
    min_visible_keypoints: int,
) -> Optional[dict]:
    """
    将一个 AnimeSkeletonData（pose_format="coco17"）转换为 COCO 记录 dict。
    返回 None 表示该样本应跳过。

    dataset_root: 用于拼接图像绝对路径以读取图像尺寸；
                  若 sample.image_size 已知则可为 None。
    """
    # ── 图像尺寸 ──────────────────────────────────────────────
    img_size = sample.image_size
    if img_size is None and dataset_root is not None:
        abs_img = dataset_root / sample.image_path
        img_size = _get_image_size(abs_img)
    if img_size is None:
        return None
    img_w, img_h = img_size

    # ── 关键点 flat ────────────────────────────────────────────
    keypoints_flat: list[float] = []
    num_visible = 0
    for name in keypoint_names:
        kp = sample.keypoints.get(name, {"x": 0.0, "y": 0.0, "visibility": 0})
        x = float(kp.get("x", 0.0))
        y = float(kp.get("y", 0.0))
        v = int(kp.get("visibility", 0))
        keypoints_flat.extend([x, y, v])
        if v > 0:
            num_visible += 1

    if num_visible < min_visible_keypoints:
        return None

    # ── BBox ──────────────────────────────────────────────────
    bbox = sample.bbox
    if bbox and all(v is not None for v in bbox):
        bbox = [float(v) for v in bbox]
        area = bbox[2] * bbox[3]
    else:
        xs = [keypoints_flat[i]   for i in range(0, len(keypoints_flat), 3) if keypoints_flat[i + 2] > 0]
        ys = [keypoints_flat[i+1] for i in range(0, len(keypoints_flat), 3) if keypoints_flat[i + 2] > 0]
        if xs:
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
            w, h = max_x - min_x, max_y - min_y
            bbox = [max(0.0, min_x - w * 0.05), max(0.0, min_y - h * 0.05), w * 1.1, h * 1.1]
            area = bbox[2] * bbox[3]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]
            area = 0.0

    return {
        "id": img_id,
        "ann_id": ann_id,
        "file_name": sample.image_path,
        "width": int(img_w),
        "height": int(img_h),
        "bbox": bbox,
        "area": area,
        "keypoints": keypoints_flat,
        "num_keypoints": num_visible,
    }


# ─────────────────────────────────────────────
# 新接口（推荐）
# ─────────────────────────────────────────────

def convert_from_samples(
    samples: Iterable["AnimeSkeletonData"],
    output_path: str | Path,
    dataset_root: Optional[Path] = None,
    keypoint_names: Optional[list[str]] = None,
    skeleton: Optional[list[list[int]]] = None,
    min_visible_keypoints: int = 4,
    show_progress: bool = True,
) -> Path:
    """
    将任意 AnimeSkeletonData 可迭代对象转换为单个 COCO JSON 文件。

    Parameters
    ----------
    samples                 : 可迭代的 AnimeSkeletonData，pose_format 须为 "coco17"
    output_path             : 输出 JSON 路径
    dataset_root            : 数据集根目录，用于定位图像读取尺寸；
                              若样本的 image_size 字段已填充则可省略
    keypoint_names          : 关键点名称顺序，默认 COCO17
    skeleton                : COCO skeleton，默认 COCO17
    min_visible_keypoints   : 过滤阈值，可见点数低于此值的样本被丢弃
    show_progress           : 是否显示 tqdm 进度条

    Returns
    -------
    写入的 JSON 文件绝对路径
    """
    if keypoint_names is None:
        keypoint_names = COCO17_KEYPOINT_NAMES
    if skeleton is None:
        skeleton = COCO17_SKELETON

    template = _build_coco_template(keypoint_names, skeleton)
    records: list[dict] = []
    skipped_no_image = 0
    skipped_few_kpts = 0
    img_id = 1

    iterable = tqdm(samples, desc=f"Converting → {Path(output_path).name}") if show_progress else samples

    for sample in iterable:
        if sample.pose_format != "coco17":
            raise ValueError(
                f"convert_from_samples 仅接受 pose_format='coco17' 的样本，"
                f"收到 '{sample.pose_format}'（sample_key={sample.sample_key}）。"
                f"请先调用 sample.to_coco17() 进行转换。"
            )

        record = _sample_to_record(
            sample=sample,
            img_id=img_id,
            ann_id=img_id,
            dataset_root=dataset_root,
            keypoint_names=keypoint_names,
            min_visible_keypoints=min_visible_keypoints,
        )

        if record is None:
            # 区分跳过原因
            img_size = sample.image_size or (
                _get_image_size(dataset_root / sample.image_path)
                if dataset_root else None
            )
            if img_size is None:
                skipped_no_image += 1
            else:
                skipped_few_kpts += 1
            continue

        records.append(record)
        img_id += 1

    total = len(records)
    print(
        f"[convert_from_samples] 有效样本 {total} | "
        f"跳过(无图/尺寸) {skipped_no_image} | "
        f"跳过(可见点不足) {skipped_few_kpts}"
    )

    out = _write_coco_json(records, template, output_path)
    print(f"✓ 已写入: {out}  ({total} 条)")
    return out


def convert_from_dataset(
    dataset: "AnimePoseDataset",
    output_dir: str | Path = "annotations",
    filename_prefix: str = "animpose",
    keypoint_names: Optional[list[str]] = None,
    skeleton: Optional[list[list[int]]] = None,
    min_visible_keypoints: int = 4,
    face5_provider=None,
    strict: bool = True,
    splits: Optional[list[str]] = None,
) -> dict[str, Path]:
    """
    从 AnimePoseDataset 对象一次性导出所有 split 的可训练 COCO JSON。

    数据流：
        AnimeSkeletonData(mixamo) → .to_coco17() → convert_from_samples() → COCO JSON

    按数据集自身的 split 字段（train/val/test）分组，不做额外随机划分。

    Parameters
    ----------
    dataset                 : AnimePoseDataset 实例
    output_dir              : 输出目录
    filename_prefix         : 输出文件名前缀，生成 {prefix}_{split}.json
    keypoint_names          : 关键点名称，默认 COCO17
    skeleton                : skeleton，默认 COCO17
    min_visible_keypoints   : 可见点过滤阈值
    face5_provider          : 面部5点推理器（传给 .to_coco17()）
    strict                  : 是否使用 strict 可见性判断（传给 .to_coco17()）
    splits                  : 要导出的 split 列表，None 表示全部（train/val/test）

    Returns
    -------
    {split_name: json_path} 字典
    """
    if splits is None:
        splits = ["train", "val", "test"]

    output_dir = Path(output_dir)
    results: dict[str, Path] = {}

    for split_name in splits:
        print(f"\n── 处理 split: {split_name} ──")

        def _iter_coco17_for_split(split_name=split_name):
            for model in dataset.iter_models():
                if model.split != split_name:
                    continue
                for view in model.iter_views():
                    for frame_id in range(1, dataset.config.frames_per_view + 1):
                        mixamo_sample = view.get_frame(frame_id)
                        yield mixamo_sample.to_coco17(
                            strict=strict,
                            face5_provider=face5_provider,
                        )

        out_path = output_dir / f"{filename_prefix}_{split_name}.json"

        try:
            path = convert_from_samples(
                samples=_iter_coco17_for_split(),
                output_path=out_path,
                dataset_root=dataset.root,
                keypoint_names=keypoint_names,
                skeleton=skeleton,
                min_visible_keypoints=min_visible_keypoints,
            )
            results[split_name] = path
        except Exception as e:
            print(f"  [警告] split={split_name} 导出失败: {e}")

    return results


# ─────────────────────────────────────────────
# 旧接口（保持向后兼容）
# ─────────────────────────────────────────────

def convert_and_split_data(
    model_dir: str,
    exported_subdir: str = "exported_coco17_strict1",
    keypoint_names: Optional[list[str]] = None,
    skeleton: Optional[list[list[int]]] = None,
    coco_json_prefix: str = "annotations/animpose",
    train_val_split: float = 0.85,
    random_seed: int = 42,
    min_visible_keypoints: int = 4,
) -> tuple[str, str]:
    """
    【旧接口，向后兼容】

    扫描 batch_export_all_views_frames() 生成的标注 JSON，
    构建 COCO 格式，并按比例随机划分训练/验证集写入文件。

    若数据已通过新数据类生成，推荐改用 convert_from_dataset() 或
    convert_from_samples()，它们直接按数据集的 split 字段分组，
    无需随机划分。

    Parameters
    ----------
    model_dir               : 人物模型根目录（如 "path/to/68 model"）
    exported_subdir         : 标注子目录名，默认 "exported_coco17_strict1"
    keypoint_names          : COCO 关键点名称列表，默认 COCO-17
    skeleton                : COCO skeleton（1-indexed），默认 COCO-17
    coco_json_prefix        : 输出 JSON 前缀（自动添加 _train/_val.json）
    train_val_split         : 训练集比例，默认 0.85
    random_seed             : 随机种子
    min_visible_keypoints   : 可见点过滤阈值，默认 4

    Returns
    -------
    (train_anno_file, val_anno_file) 两个 JSON 文件的绝对路径
    """
    if keypoint_names is None:
        keypoint_names = COCO17_KEYPOINT_NAMES
    if skeleton is None:
        skeleton = COCO17_SKELETON

    random.seed(random_seed)
    np.random.seed(random_seed)

    model_dir = os.path.abspath(model_dir)
    anno_root = os.path.join(model_dir, exported_subdir)

    if not os.path.isdir(anno_root):
        raise FileNotFoundError(f"标注目录不存在: {anno_root}")

    view_dirs = sorted([
        d for d in os.listdir(anno_root)
        if os.path.isdir(os.path.join(anno_root, d)) and "visual_view" in d
    ])
    if not view_dirs:
        raise FileNotFoundError(f"在 {anno_root} 下未找到 *visual_view 目录")

    template = _build_coco_template(keypoint_names, skeleton)
    all_records: list[dict] = []
    img_id = 1
    skipped_no_image = 0
    skipped_few_kpts = 0

    print(f"模型目录: {model_dir}")
    print(f"标注目录: {anno_root}")
    print(f"发现视角: {view_dirs}\n")

    for view_name in view_dirs:
        # "01 visual_view" -> view_index=1
        try:
            view_index = int(view_name.split()[0])
        except (ValueError, IndexError):
            print(f"  [跳过] 无法解析视角编号: {view_name}")
            continue

        view_prefix = f"{view_index:02d}"
        anno_dir = os.path.join(anno_root, view_name)
        render_dir = os.path.join(
            model_dir, view_name, f"{view_prefix}_1 Rendering"
        )

        json_files = sorted(glob.glob(os.path.join(anno_dir, "*.json")))
        if not json_files:
            print(f"  [跳过] 无 JSON 文件: {anno_dir}")
            continue

        for json_path in tqdm(json_files, desc=view_name):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            frame_id = data.get("frame_id", 0)

            # ── 定位渲染图 ──────────────────────────────
            img_abs, img_filename = _find_render_image(render_dir, frame_id, view_index)
            if img_abs is None:
                skipped_no_image += 1
                continue

            img_size = _get_image_size(img_abs)
            if img_size is None:
                skipped_no_image += 1
                continue
            img_w, img_h = img_size

            # ── 关键点 ──────────────────────────────────
            kps_flat_src = data.get("keypoints_flat")
            kps_dict = data.get("keypoints", {})
            keypoints_out: list[float] = []
            num_visible = 0

            if kps_flat_src and len(kps_flat_src) == len(keypoint_names) * 3:
                keypoints_out = [float(v) for v in kps_flat_src]
                num_visible = sum(1 for i in range(2, len(keypoints_out), 3) if keypoints_out[i] > 0)
            else:
                for kp_name in keypoint_names:
                    if kp_name in kps_dict:
                        p = kps_dict[kp_name]
                        x, y, v = float(p["x"]), float(p["y"]), int(p.get("visibility", 2))
                        keypoints_out.extend([x, y, v])
                        if v > 0:
                            num_visible += 1
                    else:
                        keypoints_out.extend([0.0, 0.0, 0])

            if num_visible < min_visible_keypoints:
                skipped_few_kpts += 1
                continue

            # ── BBox ────────────────────────────────────
            bbox = data.get("bbox")
            if bbox and all(v is not None for v in bbox):
                bbox = [float(v) for v in bbox]
                area = bbox[2] * bbox[3]
            else:
                xs = [keypoints_out[i]   for i in range(0, len(keypoints_out), 3) if keypoints_out[i+2] > 0]
                ys = [keypoints_out[i+1] for i in range(0, len(keypoints_out), 3) if keypoints_out[i+2] > 0]
                if xs:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    w, h = max_x - min_x, max_y - min_y
                    bbox = [max(0.0, min_x - w*0.05), max(0.0, min_y - h*0.05), w*1.1, h*1.1]
                    area = bbox[2] * bbox[3]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    area = 0.0

            # ── 图像相对路径 ─────────────────────────────
            img_rel = "/".join([
                view_name,
                f"{view_prefix}_1 Rendering",
                img_filename,
            ])

            all_records.append({
                "id": img_id,
                "ann_id": img_id,
                "file_name": img_rel,
                "width": int(img_w),
                "height": int(img_h),
                "bbox": bbox,
                "area": area,
                "keypoints": keypoints_out,
                "num_keypoints": num_visible,
            })
            img_id += 1

    print(
        f"\n汇总: 有效样本 {len(all_records)} | "
        f"跳过(无图) {skipped_no_image} | "
        f"跳过(可见点不足) {skipped_few_kpts}"
    )

    if not all_records:
        raise ValueError("没有有效样本，请检查路径和数据格式")

    random.shuffle(all_records)
    split_idx = int(len(all_records) * train_val_split)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]

    train_path = _write_coco_json(train_records, template, f"{coco_json_prefix}_train.json")
    val_path = _write_coco_json(val_records, template, f"{coco_json_prefix}_val.json")

    print(f"\n✓ 生成完成")
    print(f"  训练集: {len(train_records)} 张  →  {train_path}")
    print(f"  验证集: {len(val_records)} 张  →  {val_path}")

    return str(train_path), str(val_path)