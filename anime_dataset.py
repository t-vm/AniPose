from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Any
import json
import uuid
from PIL import Image as PILImage
import numpy as np


# 全局配置
# =========================

@dataclass(frozen=True)
class DatasetConfig:
    frames_per_view: int = 72
    image_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg")
    source_dataset_name: str = "link-to-anime"
    default_source_format: str = "mixamo"


COCO17_ORDER = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


MIXAMO_TO_COCO17_BODY = {
    "left_shoulder": "mixamorig:LeftArm",
    "right_shoulder": "mixamorig:RightArm",
    "left_elbow": "mixamorig:LeftForeArm",
    "right_elbow": "mixamorig:RightForeArm",
    "left_wrist": "mixamorig:LeftHand",
    "right_wrist": "mixamorig:RightHand",
    "left_hip": "mixamorig:LeftUpLeg",
    "right_hip": "mixamorig:RightUpLeg",
    "left_knee": "mixamorig:LeftLeg",
    "right_knee": "mixamorig:RightLeg",
    "left_ankle": "mixamorig:LeftFoot",
    "right_ankle": "mixamorig:RightFoot",
    "neck": "mixamorig:Neck",  # 中间字段，可存在于 keypoints，但不进 COCO17_ORDER
}


# =========================
# 工具函数
# =========================

# 生成某个sample frame的相对于数据集root的路径,
# 也是生成uuid的原料
def make_sample_key(
    split: str,
    pack_name: str,
    model_name: str,
    view_name: str,
    frame_id: int,
) -> str:
    return f"{split}/{pack_name}/{model_name}/{view_name}/frame_{frame_id:04d}"


def make_uuid_from_sample_key(sample_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, sample_key))


def ensure_standard_keypoint(raw: dict[str, Any]) -> dict[str, Any]:
    """
    统一成:
    {"x": float, "y": float, "z": float, "visibility": int}
    """
    x = raw.get("x", 0.0)
    y = raw.get("y", 0.0)
    z = raw.get("z", 0.0)
    v = raw.get("visibility", raw.get("v", 0))
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "visibility": int(v),
    }


def point_in_image(kp: dict[str, Any], image_size: Optional[tuple[int, int]]) -> bool:
    if image_size is None:
        return kp.get("visibility", 0) > 0
    w, h = image_size
    x, y = kp["x"], kp["y"]
    return 0 <= x < w and 0 <= y < h


def default_outside_visibility(kp: dict[str, Any], image_size: Optional[tuple[int, int]]) -> int:
    return 2 if point_in_image(kp, image_size) else 0



# 单帧样本对象
# =========================
@dataclass
class AnimeSkeletonData:
    uuid: str
    sample_key: str

    split: str
    pack_name: str
    model_name: str
    view_name: str
    view_index: int
    frame_id: int
    frames_per_view: int

    image_path: str                 # 相对数据集根目录
    
    bbox: Optional[list[float]]
    keypoints: dict[str, dict[str, Any]]

    pose_format: str                 # "mixamo" / "coco17"
    source_format: str               # 通常 "mixamo"
    source_dataset: str              # "link-to-anime"

    strict: Optional[bool] = None
    face5_info: Optional[str] = None  # 面部五个点的情况。"inferred" / "Head out of frame" ...
    num_keypoints: Optional[int] = None
    image_size: Optional[tuple[int, int]] = None
    # json_out_path: Optional[Path] = field(default=None, init=False, repr=False)
    # coco_json_out_path: Optional[Path] = field(default=None, init=False, repr=False)
    json_out_path: Optional[Path] = field(default=None, repr=False)
    coco_json_out_path: Optional[Path] = field(default=None, repr=False)
    extra: dict[str, Any] = field(default_factory=dict)

    """
    四个基础导出函数：
    - to_keypoints_flat(order): 按照给定顺序展平关键点为列表 [x1, y1, v1, x2, y2, v2, ...]
    - count_visible_keypoints(order): 统计在给定顺序中 visibility > 0 的关键点数量
    - to_intermediate_dict(): 转换为一个包含所有字段的字典，适合直接 json.dump
    - save_json(out_path, keypoint_order, indent): 保存为 JSON 文件，keypoint_order 可选用于添加 "keypoints_flat" 字段
    """

    def to_keypoints_flat(self, order: list[str]) -> list[float]:
        flat = []
        for name in order:
            kp = self.keypoints.get(name, {"x": 0.0, "y": 0.0, "visibility": 0})
            flat.extend([kp["x"], kp["y"], kp["visibility"]])
        return flat

    def count_visible_keypoints(self, order: Optional[list[str]] = None) -> int:
        if order is None:
            values = self.keypoints.values()
        else:
            values = [self.keypoints.get(k, {"visibility": 0}) for k in order]
        return sum(1 for kp in values if int(kp.get("visibility", 0)) > 0)

    def to_intermediate_dict(self) -> dict[str, Any]:
        data = {
            "uuid": self.uuid,
            "sample_key": self.sample_key,
            "split": self.split,
            "pack_name": self.pack_name,
            "model_name": self.model_name,
            "view": self.view_name,
            "view_index": self.view_index,
            "frame_id": self.frame_id,
            "standard": self.pose_format,
            "source_format": self.source_format,
            "source_dataset": self.source_dataset,
            "strict": self.strict,
            "bbox": self.bbox,
            "face5_info": self.face5_info,
            "num_keypoints": self.num_keypoints,
            "keypoints": self.keypoints,
            "image_path": self.image_path,
        }
        if self.image_size is not None:
            data["image_size"] = [self.image_size[0], self.image_size[1]]
        data.update(self.extra)
        return data

    def save_json(
        self,
        out_path: Path,
        keypoint_order: Optional[list[str]] = None,
        indent: int = 2,
    ) -> None:
        data = self.to_intermediate_dict()
        if keypoint_order is not None:
            data["keypoints_flat"] = self.to_keypoints_flat(keypoint_order)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

    # ---------- 转换 ----------

    def to_coco17(self, strict: Optional[bool], face5_provider: "Face5Provider", dataset_root: Path) -> "AnimeSkeletonData":
        """
        将 Mixamo 骨骼数据转换为 COCO-17 格式。
    
        参数
        ----
        face5_provider : Face5Provider 实例（必填）
            负责推理面部5点，没有模型时直接抛 RuntimeError。
    
        返回
        ----
        pose_format="coco17" 的新 AnimeSkeletonData 实例。
    
        关于 visibility 的规则
        ----------------------
        - 身体12点：有像素坐标且在画面内 → visibility=2，画面外 → visibility=0
        - 面部5点 ：RTMLib 推理成功 → visibility=2，跳过/失败 → visibility=0
        - 判断"在画面内"依赖 self.image_size，若为 None 则不做裁剪检查
        """
        if self.pose_format != "mixamo":
            raise ValueError(
                f"to_coco17() 仅支持 pose_format='mixamo' 的样本，"
                f"当前为 '{self.pose_format}'"
            )
    
        # ── 工具：判断一个像素点是否在画面内 ──────────────────────────
        def _in_frame(x: float, y: float) -> bool:
            if self.image_size is None:
                return True          # 没有尺寸信息，不做裁剪判断
            w, h = self.image_size
            return (0.0 <= x < w) and (0.0 <= y < h)
    
        # ── 加载图像 ──────────────────────────────────────────────────
        # image_path 是相对 dataset_root 的 posix 路径

    
        abs_img_path = Path(dataset_root) / self.image_path          # 调用方保证传绝对路径或可解析路径
        img_rgb = np.array(PILImage.open(abs_img_path).convert("RGB"))
    
        # ── 推理面部5点 ───────────────────────────────────────────────
        # 没有模型时 Face5Provider._ensure_model() 直接抛 RuntimeError
        face5_result = face5_provider.infer(
            img_rgb=img_rgb,
            mixamo_pixels=self.keypoints,
        )
    
        # ── 构建 COCO-17 关键点字典 ──────────────────────────────────
        coco_kps: dict[str, dict] = {}
    
        # 面部5点
        for name in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]:
            if (not face5_result.skipped) and (name in face5_result.face5):
                p = face5_result.face5[name]
                x, y = p["x"], p["y"]
                vis = 2 if _in_frame(x, y) else 0
                coco_kps[name] = {"x": x, "y": y, "z": 0.0, "visibility": vis}
            else:
                coco_kps[name] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0}
    
        # 身体12点（从 Mixamo 骨骼映射）
        for coco_name, mixamo_bone in MIXAMO_TO_COCO17_BODY.items():
            src = self.keypoints.get(mixamo_bone)
            if src is None:
                coco_kps[coco_name] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0}
            else:
                x, y = float(src.get("x", 0.0)), float(src.get("y", 0.0))
                z    = float(src.get("z", 0.0))
                vis  = 2 if _in_frame(x, y) else 0
                coco_kps[coco_name] = {"x": x, "y": y, "z": z, "visibility": vis}
    
        # 严格按 COCO17_ORDER 排列，缺的点补零
        ordered_kps = {
            k: coco_kps.get(k, {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0})
            for k in COCO17_ORDER
        }
    
        num_visible = sum(1 for v in ordered_kps.values() if v["visibility"] > 0)
    
        return AnimeSkeletonData(
            uuid=self.uuid,
            sample_key=self.sample_key,
            split=self.split,
            pack_name=self.pack_name,
            model_name=self.model_name,
            view_name=self.view_name,
            view_index=self.view_index,
            frame_id=self.frame_id,
            frames_per_view=self.frames_per_view,
            image_path=self.image_path,
            bbox=self.bbox,
            keypoints=ordered_kps,
            pose_format="coco17",
            source_format=self.source_format,
            source_dataset=self.source_dataset,
            strict=None,                          # strict 参数已移除，不再存储
            face5_info=face5_result.reason,       # "inferred" / "Head out of frame" / ...
            num_keypoints=num_visible,
            image_size=self.image_size,
            extra=dict(self.extra),
        )



# 单视角对象
# =========================
@dataclass
class AnimeView:
    dataset_root: Path
    split: str
    pack_name: str
    model_name: str
    model_path: Path
    view_name: str
    json_out_dir: Optional[Path] = field(default=None, init=False, repr=False)
    coco_json_out_dir: Optional[Path] = field(default=None, init=False, repr=False)
    config: DatasetConfig = field(default_factory=DatasetConfig)

    _bone_pixels_cache: Optional[dict[str, Any]] = field(default=None, init=False, repr=False)
    _bbox_cache: Optional[dict[str, Any]] = field(default=None, init=False, repr=False)


    @property
    def view_index(self) -> int:
        return int(self.view_name.split()[0])


    @property
    def prefix(self) -> str:
        return f"{self.view_index:02d}"


    @property
    def view_path(self) -> Path:
        return self.model_path / self.view_name


    @property
    def rendering_dir(self) -> Path:
        return self.view_path / f"{self.prefix}_1 Rendering"


    @property
    def bone_pixels_json_path(self) -> Path:
        return self.view_path / f"{self.prefix}_9 bone_coordinates_pixels.json"


    @property
    def bbox_json_path(self) -> Path:
        return self.view_path / f"{self.prefix}_10 Bounding_Box.json"


    def load_bone_pixels(self) -> dict[str, Any]:
        if self._bone_pixels_cache is None:
            with open(self.bone_pixels_json_path, "r", encoding="utf-8") as f:
                self._bone_pixels_cache = json.load(f)
        return self._bone_pixels_cache


    def load_bbox(self) -> dict[str, Any]:
        if self._bbox_cache is None:
            with open(self.bbox_json_path, "r", encoding="utf-8") as f:
                self._bbox_cache = json.load(f)
        return self._bbox_cache


    # 允许外部调用以清除缓存
    def clear_cache(self) -> None:
        self._bone_pixels_cache = None
        self._bbox_cache = None


    def get_image_path(self, frame_id: int) -> Path:
        frame_prefix = f"{frame_id:04d}_"
        matches = [
            p for p in self.rendering_dir.iterdir()
            if p.is_file() and p.name.startswith(frame_prefix) and p.suffix.lower() in self.config.image_exts
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Cannot uniquely resolve image for frame {frame_id} in {self.rendering_dir}. "
                f"Found: {[m.name for m in matches]}"
            )
        return matches[0]


    def get_frame_bbox(self, frame_id: int) -> Optional[list[float]]:
        data = self.load_bbox()
        key = f"{frame_id:04d}_{self.view_index}"
        bbox = data.get(key)
        return bbox


    def get_frame_mixamo_keypoints(self, frame_id: int) -> dict[str, dict[str, Any]]:
        data = self.load_bone_pixels()
        frame_key = str(frame_id)
        raw = data.get(frame_key, {})

        normalized = {}
        for bone_name, kp in raw.items():
            normalized[bone_name] = ensure_standard_keypoint(kp)
        return normalized


    def get_frame_image_size(self, frame_id: int) -> Optional[tuple[int, int]]:
        # 如你后面需要可用 PIL 读；当前先不强依赖
        try:
            from PIL import Image
            img_path = self.get_image_path(frame_id)
            with Image.open(img_path) as img:
                w, h = img.size
            return (w, h)
        except Exception:
            return None


    def get_frame(self, frame_id: int) -> AnimeSkeletonData:
        if not (1 <= frame_id <= self.config.frames_per_view):
            raise ValueError(f"frame_id must be in [1, {self.config.frames_per_view}], got {frame_id}")

        image_path_abs = self.get_image_path(frame_id)
        image_path_rel = image_path_abs.relative_to(self.dataset_root).as_posix()

        keypoints = self.get_frame_mixamo_keypoints(frame_id)
        bbox = self.get_frame_bbox(frame_id)
        image_size = self.get_frame_image_size(frame_id)

        sample_key = make_sample_key(
            split=self.split,
            pack_name=self.pack_name,
            model_name=self.model_name,
            view_name=self.view_name,
            frame_id=frame_id,
        )
        sample_uuid = make_uuid_from_sample_key(sample_key)

        # gen coco json path
        coco_json_out_path = (
            Path(self.model_path) / "exported_coco17" / self.view_name
            / f"frame_{frame_id:04d}_coco17.json"
        )
        # gen json data(not edited ver.) path
        json_out_path = (
            Path(self.model_path) / "exported_data" / self.view_name
            / f"frame_{frame_id:04d}.json"
        )
        return AnimeSkeletonData(
            uuid=sample_uuid,
            sample_key=sample_key,
            split=self.split,
            pack_name=self.pack_name,
            model_name=self.model_name,
            view_name=self.view_name,
            view_index=self.view_index,
            frame_id=frame_id,
            frames_per_view=self.config.frames_per_view,
            image_path=image_path_rel,
            bbox=bbox,
            keypoints=keypoints,
            pose_format="mixamo",
            source_format=self.config.default_source_format,
            source_dataset=self.config.source_dataset_name,
            strict=None,
            face5_info=None,
            num_keypoints=None,
            image_size=image_size,
            extra={},
            coco_json_out_path=coco_json_out_path,
            json_out_path=json_out_path
        )


    # 该函数是一个generator，
    # 逐帧返回 AnimeSkeletonData 对象，适合用于迭代处理整个视角的数据。0
    def iter_frames(self) -> Iterator[AnimeSkeletonData]:
        for frame_id in range(1, self.config.frames_per_view + 1):
            yield self.get_frame(frame_id)

    # def export_coco17_jsons(
    #     self,
    #     output_dir_name: str = "exported_coco17_strict1",
    #     strict: bool = True,
    #     face5_provider: Optional[Any] = None,
    # ) -> None:
    #     for frame_id in range(1, self.config.frames_per_view + 1):
    #         mixamo_sample = self.get_frame(frame_id)
    #         coco_sample = mixamo_sample.to_coco17(strict=strict, face5_provider=face5_provider)

    #         out_path = (
    #             self.model_path
    #             / output_dir_name
    #             / self.view_name
    #             / f"frame_{frame_id:04d}_coco17_strict{int(strict)}.json"
    #         )
    #         coco_sample.save_json(out_path, keypoint_order=COCO17_ORDER)
    def export_coco17_jsons(
        self,
        face5_provider: "Face5Provider",
        output_dir_name: str = "exported_coco17",
    ) -> None:
        for frame_id in range(1, self.config.frames_per_view + 1):
            mixamo_sample = self.get_frame(frame_id)
            coco_sample   = mixamo_sample.to_coco17(face5_provider=face5_provider)
    
            out_path = (
                self.model_path
                / output_dir_name
                / self.view_name
                / f"frame_{frame_id:04d}_coco17.json"
            )
            coco_sample.save_json(out_path, keypoint_order=COCO17_ORDER)



# 单模型对象
# =========================
@dataclass
class AnimeModel:
    dataset_root: Path
    split: str
    pack_name: str
    model_name: str
    model_path: Path
    config: DatasetConfig = field(default_factory=DatasetConfig)

    _views_cache: Optional[dict[str, AnimeView]] = field(default=None, init=False, repr=False)

    def _scan_views(self) -> dict[str, AnimeView]:
        views = {}
        for p in sorted(self.model_path.iterdir()):
            if p.is_dir() and p.name.endswith("visual_view"):
                views[p.name] = AnimeView(
                    dataset_root=self.dataset_root,
                    split=self.split,
                    pack_name=self.pack_name,
                    model_name=self.model_name,
                    model_path=self.model_path,
                    view_name=p.name,
                    config=self.config,
                )
        return views

    def views(self) -> dict[str, AnimeView]:
        if self._views_cache is None:
            self._views_cache = self._scan_views()
        return self._views_cache

    def get_view(self, view_name: str) -> AnimeView:
        try:
            return self.views()[view_name]
        except KeyError:
            raise KeyError(f"View not found: {view_name} under model {self.model_name}")

    def iter_views(self) -> Iterator[AnimeView]:
        yield from self.views().values()

    def iter_frames(self) -> Iterator[AnimeSkeletonData]:
        for view in self.iter_views():
            yield from view.iter_frames()

    def export_all_views_coco17_jsons(
        self,
        output_dir_name: str = "exported_coco17_strict1",
        strict: bool = True,
        face5_provider: Optional[Any] = None,
    ) -> None:
        for view in self.iter_views():
            view.export_coco17_jsons(
                output_dir_name=output_dir_name,
                strict=strict,
                face5_provider=face5_provider,
            )



# 整个数据集对象
# =========================
@dataclass
class AnimePoseDataset:
    root: Path
    config: DatasetConfig = field(default_factory=DatasetConfig)
    debug: bool = True  # debug mode, 打印扫描信息

    _models_cache: Optional[list[AnimeModel]] = field(default=None, init=False, repr=False)
    _uuid_index: Optional[dict[str, tuple[str, str, str, str, int]]] = field(default=None, init=False, repr=False)

    def _scan_models(self) -> list[AnimeModel]:
        models: list[AnimeModel] = []

        # finding splits under root
        for split_name in ["train", "val", "test"]:
            split_dir = self.root / split_name
            if self.debug: print(f"[DEBUG] checking split_dir: {split_dir}")
            
            if not split_dir.exists():  # split : Path; if not exist:
                if self.debug: print(f"[DEBUG] split_dir does not exist: {split_dir}, skipping.")
                continue
            
            # finding packs under each split_dir 
            for pack_dir in sorted(split_dir.iterdir()):
                if not pack_dir.is_dir():
                    if self.debug: print(f"[DEBUG] skipping non-dir in split_dir: {pack_dir}")
                    continue
                
                # finding models under each pack_dir
                for model_dir in sorted(pack_dir.iterdir()):
                    if self.debug: print(f"[DEBUG] checking model_dir: {model_dir}")
                    if model_dir.is_dir() and model_dir.name.endswith("model"):
                        models.append(
                            AnimeModel(
                                dataset_root=self.root,
                                split=split_name,
                                pack_name=pack_dir.name,
                                model_name=model_dir.name,
                                model_path=model_dir,
                                config=self.config,
                            )
                        )
        return models

    # ---------- 第一优先级：核心层级接口 ----------

    def iter_models(self) -> Iterator[AnimeModel]:
        if self._models_cache is None:
            self._models_cache = self._scan_models()
        yield from self._models_cache

    def get_model(self, split: str, pack_name: str, model_name: str) -> AnimeModel:
        for model in self.iter_models():
            if model.split == split and model.pack_name == pack_name and model.model_name == model_name:
                return model
        raise KeyError(f"Model not found: split={split}, pack={pack_name}, model={model_name}")

    # ---------- 第三优先级：全局便捷遍历 ----------

    def iter_views(
        self,
        split: Optional[str] = None,
    ) -> Iterator[AnimeView]:
        for model in self.iter_models():
            if split is not None and model.split != split:
                continue
            yield from model.iter_views()

    def iter_frames(
        self,
        split: Optional[str] = None,
    ) -> Iterator[AnimeSkeletonData]:
        for model in self.iter_models():
            if split is not None and model.split != split:
                continue
            yield from model.iter_frames()

    # ---------- 第二优先级：uuid 索引 ----------

    def build_index(self) -> dict[str, tuple[str, str, str, str, int]]:
        """
        uuid -> (split, pack_name, model_name, view_name, frame_id)
        """
        if self._uuid_index is None:
            index = {}
            for model in self.iter_models():
                for view in model.iter_views():
                    for frame_id in range(1, self.config.frames_per_view + 1):
                        sample_key = make_sample_key(
                            split=model.split,
                            pack_name=model.pack_name,
                            model_name=model.model_name,
                            view_name=view.view_name,
                            frame_id=frame_id,
                        )
                        sample_uuid = make_uuid_from_sample_key(sample_key)
                        index[sample_uuid] = (
                            model.split,
                            model.pack_name,
                            model.model_name,
                            view.view_name,
                            frame_id,
                        )
            self._uuid_index = index
        return self._uuid_index

    def get_by_uuid(self, sample_uuid: str) -> AnimeSkeletonData:
        if self._uuid_index is None:
            self.build_index()

        if sample_uuid not in self._uuid_index:
            raise KeyError(f"UUID not found: {sample_uuid}")

        split, pack_name, model_name, view_name, frame_id = self._uuid_index[sample_uuid]
        model = self.get_model(split, pack_name, model_name)
        view = model.get_view(view_name)
        return view.get_frame(frame_id)

    # ---------- 额外辅助 ----------

    def clear_all_view_caches(self) -> None:
        for model in self.iter_models():
            for view in model.iter_views():
                view.clear_cache()