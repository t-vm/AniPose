"""
coco_converter.py
-----------------
COCO格式数据转换工具函数（从 notebook 解耦）
调用方式：
    from coco_converter import convert_and_split_data

输入数据结构（由 batch_export_all_views_frames() 生成）：
    <model_dir>/
    └── exported_coco17_strict1/
        ├── 01 visual_view/
        │   ├── frame_0001_coco17_strict1.json
        │   └── ...
        └── 02 visual_view/
            └── ...

单个 JSON 字段说明：
    - bbox: [x, y, w, h]  COCO 格式，直接来自 Bounding_Box.json
    - keypoints: { name: {x, y, z, visibility} }
    - keypoints_flat: [x1, y1, v1, x2, y2, v2, ...]  visibility: 0=画面外, 2=可见
"""

import os
import json
import glob
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image


# ─────────────────────────────────────────────
# COCO-17 标准关键点顺序
# ─────────────────────────────────────────────

COCO17_KEYPOINT_NAMES = [
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
COCO17_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def get_image_size(img_abs_path: str):
    """
    获取图像真实尺寸。
    Returns:
        (width, height) 或 None（文件不存在时）
    """
    try:
        with Image.open(img_abs_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def find_render_image(render_dir: str, frame_id: int, view_prefix: str):
    """
    在渲染目录下定位帧图像文件。
    文件名格式示例：0001_1.png、0016_4.png（帧号4位_相机编号.扩展名）
    Returns:
        (abs_path, rel_to_render_dir) 或 (None, None)
    """
    cam_id = view_prefix.lstrip("0") or "1"
    candidates = [
        f"{frame_id:04d}_{cam_id}.png",
        f"{frame_id:04d}_{cam_id}.jpg",
        f"{frame_id:04d}_{cam_id}.jpeg",
        f"{frame_id:04d}.png",
        f"{frame_id:04d}.jpg",
    ]
    for name in candidates:
        full = os.path.join(render_dir, name)
        if os.path.exists(full):
            return full, name
    # 如果标准名找不到，扫目录匹配帧号
    if os.path.isdir(render_dir):
        for f in sorted(os.listdir(render_dir)):
            stem = Path(f).stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            if digits and int(digits) == frame_id:
                return os.path.join(render_dir, f), f
    return None, None


def _build_coco_template(keypoint_names: list, skeleton: list) -> dict:
    """构造空的 COCO 标注模板。"""
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


def _save_coco_json(samples: list, coco_template: dict, output_path: str) -> str:
    """
    把 samples 列表序列化成 COCO JSON 文件。
    Returns:
        写入的绝对路径
    """
    coco = json.loads(json.dumps(coco_template))  # 深拷贝
    for s in samples:
        coco["images"].append({
            "id": s["id"],
            "file_name": s["file_name"],
            "height": s["height"],
            "width": s["width"],
        })
        coco["annotations"].append({
            "id": s["ann_id"],
            "image_id": s["id"],
            "category_id": 1,
            "bbox": s["bbox"],
            "area": s["area"],
            "keypoints": s["keypoints"],
            "num_keypoints": s["num_keypoints"],
            "iscrowd": 0,
            "segmentation": [],  # mmpose 必需字段
        })
    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path) if os.path.dirname(abs_path) else ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    return abs_path


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def convert_and_split_data(
    model_dir: str,
    exported_subdir: str = "exported_coco17_strict1",
    keypoint_names: list = None,
    skeleton: list = None,
    coco_json_prefix: str = "annotations/animpose",
    train_val_split: float = 0.85,
    random_seed: int = 42,
    min_visible_keypoints: int = 4,
) -> tuple:
    """
    扫描 batch_export_all_views_frames() 生成的标注 JSON
    → 构建 COCO 格式 → 按比例划分训练/验证集并写入文件。

    Parameters
    ----------
    model_dir               : 人物模型根目录（如 "path/to/68 model"）
    exported_subdir         : 标注子目录名，默认 "exported_coco17_strict1"
    keypoint_names          : COCO 关键点名称列表，默认 COCO-17
    skeleton                : COCO skeleton（1-indexed），默认 COCO-17
    coco_json_prefix        : 输出 JSON 前缀（自动添加 _train/_val.json）
    train_val_split         : 训练集比例，默认 0.85
    random_seed             : 随机种子
    min_visible_keypoints   : 过滤样本：可见点数低于此值的样本将被丢弃，默认 4

    Returns
    -------
    (train_anno_file, val_anno_file) : 两个 JSON 文件的绝对路径
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

    # 发现所有视角目录
    view_dirs = sorted([
        d for d in os.listdir(anno_root)
        if os.path.isdir(os.path.join(anno_root, d)) and "visual_view" in d
    ])
    if not view_dirs:
        raise FileNotFoundError(f"在 {anno_root} 下未找到 *visual_view 目录")

    coco_template = _build_coco_template(keypoint_names, skeleton)
    all_samples = []
    img_id = 1
    ann_id = 1
    skipped_no_image = 0
    skipped_few_kpts = 0

    print(f"模型目录: {model_dir}")
    print(f"标注目录: {anno_root}")
    print(f"发现视角: {view_dirs}\n")

    for view_name in view_dirs:
        view_prefix = view_name.split()[0]           # "01 visual_view" -> "01"
        anno_dir = os.path.join(anno_root, view_name)
        render_dir = os.path.join(
            model_dir, view_name, f"{view_prefix}_1 Rendering"
        )

        json_files = sorted(glob.glob(os.path.join(anno_dir, "*.json")))
        if not json_files:
            print(f"  [跳过] 无 JSON 文件: {anno_dir}")
            continue

        for json_path in tqdm(json_files, desc=view_name):
            # ── 读取标注 ──────────────────────────────
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            frame_id = data.get("frame_id", 0)

            # ── 定位渲染图 ────────────────────────────
            img_abs, img_filename = find_render_image(render_dir, frame_id, view_prefix)
            if img_abs is None:
                skipped_no_image += 1
                continue

            img_size = get_image_size(img_abs)
            if img_size is None:
                skipped_no_image += 1
                continue
            img_w, img_h = img_size

            # ── 解析关键点 ────────────────────────────
            # 优先使用 keypoints_flat（已按 COCO-17 顺序排列，含 visibility）
            kps_flat_src = data.get("keypoints_flat")
            kps_dict = data.get("keypoints", {})

            keypoints_out = []  # [x1, y1, v1, x2, y2, v2, ...]
            num_visible = 0

            if kps_flat_src and len(kps_flat_src) == len(keypoint_names) * 3:
                # 直接使用已有的 flat 格式
                keypoints_out = [float(v) for v in kps_flat_src]
                num_visible = sum(
                    1 for i in range(2, len(keypoints_out), 3)
                    if keypoints_out[i] > 0
                )
            else:
                # 从 keypoints dict 重建
                for kp_name in keypoint_names:
                    if kp_name in kps_dict:
                        p = kps_dict[kp_name]
                        x = float(p["x"])
                        y = float(p["y"])
                        v = int(p.get("visibility", 2))
                        keypoints_out.extend([x, y, v])
                        if v > 0:
                            num_visible += 1
                    else:
                        keypoints_out.extend([0.0, 0.0, 0])

            # 过滤可见点过少的样本
            if num_visible < min_visible_keypoints:
                skipped_few_kpts += 1
                continue

            # ── BBox ──────────────────────────────────
            # 优先使用标注文件中的 bbox（来自 Bounding_Box.json，已是 [x,y,w,h]）
            bbox = data.get("bbox")
            if bbox and all(v is not None for v in bbox):
                bbox = [float(v) for v in bbox]
                area = bbox[2] * bbox[3]
            else:
                # 回退：从可见关键点计算 bbox（外扩 5%）
                xs = [keypoints_out[i]   for i in range(0, len(keypoints_out), 3) if keypoints_out[i+2] > 0]
                ys = [keypoints_out[i+1] for i in range(0, len(keypoints_out), 3) if keypoints_out[i+2] > 0]
                if xs:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    w = max_x - min_x
                    h = max_y - min_y
                    bbox = [
                        max(0.0, min_x - w * 0.05),
                        max(0.0, min_y - h * 0.05),
                        w * 1.1,
                        h * 1.1,
                    ]
                    area = bbox[2] * bbox[3]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    area = 0.0

            # ── 图像相对路径（COCO file_name 字段）─────
            # 格式：<view_name>/<prefix>_1 Rendering/<filename>
            # 例如：04 visual_view/04_1 Rendering/0016_4.png
            img_rel = os.path.join(
                view_name,
                f"{view_prefix}_1 Rendering",
                img_filename,
            ).replace("\\", "/")

            all_samples.append({
                "id": img_id,
                "file_name": img_rel,
                "width": int(img_w),
                "height": int(img_h),
                "ann_id": ann_id,
                "bbox": bbox,
                "area": area,
                "keypoints": keypoints_out,
                "num_keypoints": num_visible,
            })
            img_id += 1
            ann_id += 1

    print(f"\n汇总: 有效样本 {len(all_samples)} | "
          f"跳过(无图) {skipped_no_image} | "
          f"跳过(可见点不足) {skipped_few_kpts}")

    if not all_samples:
        raise ValueError("没有有效样本，请检查路径和数据格式")

    # ── 随机划分 ──────────────────────────────────
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_val_split)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # ── 写文件 ────────────────────────────────────
    train_path = _save_coco_json(
        train_samples, coco_template,
        f"{coco_json_prefix}_train.json"
    )
    val_path = _save_coco_json(
        val_samples, coco_template,
        f"{coco_json_prefix}_val.json"
    )

    print(f"\n✓ 生成完成")
    print(f"  训练集: {len(train_samples)} 张  →  {train_path}")
    print(f"  验证集: {len(val_samples)} 张  →  {val_path}")

    return train_path, val_path