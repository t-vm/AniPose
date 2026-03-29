"""
face5_provider.py
-----------------
RTMLib Wholebody 面部5点推理模块（独立，与 notebook 解耦）

职责：
    给定一张 RGB 图像 + Mixamo 骨骼像素坐标 →
    用 RTMLib Wholebody 模型推理出 COCO-17 前5个面部关键点。

不做的事：
    - 不负责加载图像（调用方传入）
    - 不负责加载骨骼数据（调用方传入）
    - 不做粗糙几何估算兜底，没有模型直接报错

调用方式：
    from face5_provider import Face5Provider

    provider = Face5Provider(device="cpu", backend="onnxruntime")
    result = provider.infer(img_rgb, mixamo_pixels, img_shape=img_rgb.shape)
    # result.face5   -> dict | None
    # result.skipped -> bool
    # result.reason  -> str
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────

# RTMLib Wholebody 输出中面部5点的索引（COCO-17 顺序）
FACE5_IDX: dict[str, int] = {
    "nose":      0,
    "left_eye":  1,
    "right_eye": 2,
    "left_ear":  3,
    "right_ear": 4,
}

# Mixamo 骨骼中用于判断面部区域是否在画面内的骨骼名（优先顺序）
HEAD_BONE_CANDIDATES: list[str] = [
    "mixamorig:Head",
    "mixamorig:Neck",
    "mixamorig:HeadTop_End",
]


# ─────────────────────────────────────────────────────────────────
# 返回值数据类
# ─────────────────────────────────────────────────────────────────

@dataclass
class Face5Result:
    """
    infer() 的返回值。

    face5   : 面部5点坐标字典，格式 {"nose": {"x":..,"y":..,"z":0.0}, ...}
              若 skipped=True 则为空字典 {}
    skipped : 是否跳过了推理（面部区域不在画面内）
    reason  : 说明字符串，例如 "inferred" / "Head out of frame" / "no head bones found"
    """
    face5:   dict[str, dict]
    skipped: bool
    reason:  str


# ─────────────────────────────────────────────────────────────────
# 核心类
# ─────────────────────────────────────────────────────────────────

@dataclass
class Face5Provider:
    """
    RTMLib Wholebody 面部5点推理器。

    Parameters
    ----------
    device  : "cpu" / "cuda"
    backend : "onnxruntime" / "openvino"，默认 "onnxruntime"
    mode    : RTMLib Wholebody 模式，默认 "performance"
    skip_if_face_outside : 若面部骨骼区域不在画面内，跳过推理（不报错）

    用法：
        provider = Face5Provider()
        result   = provider.infer(img_rgb, mixamo_pixels, img_shape)
    """

    device:               str  = "cuda"
    backend:              str  = "onnxruntime"
    mode:                 str  = "performance"
    skip_if_face_outside: bool = True

    # 内部缓存，不暴露给构造参数
    _model: object = field(default=None, init=False, repr=False)

    # ── 模型懒加载 ────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """
        懒加载 RTMLib Wholebody 模型。
        没有 rtmlib 时直接抛出 RuntimeError（不做兜底估算）。
        """
        if self._model is not None:
            return

        try:
            from rtmlib import Wholebody
        except ImportError as e:
            raise RuntimeError(
                "[Face5Provider] rtmlib 未安装。\n"
                "请运行: pip install rtmlib onnxruntime\n"
                "没有 RTMLib 模型时，本项目不做粗糙几何估算兜底，"
                "请安装模型后再运行补点流程。"
            ) from e

        try:
            self._model = Wholebody(
                mode=self.mode,
                backend=self.backend,
                device=self.device,
            )
        except TypeError:
            # 部分旧版 rtmlib 不支持 mode 参数
            self._model = Wholebody(
                backend=self.backend,
                device=self.device,
            )

    # ── 面部区域可见性判断 ────────────────────────────────────────

    @staticmethod
    def _is_face_region_visible(
        mixamo_pixels: dict[str, dict],
        img_shape: tuple[int, int, int],
    ) -> tuple[bool, str]:
        """
        检查面部区域是否在画面内。

        Parameters
        ----------
        mixamo_pixels : {"骨骼名": {"x": float, "y": float, ...}, ...}
        img_shape     : (H, W, C) 或 (H, W)

        Returns
        -------
        (visible: bool, reason: str)
        """
        H, W = img_shape[:2]

        for bone_name in HEAD_BONE_CANDIDATES:
            p = mixamo_pixels.get(bone_name)
            if p is None:
                continue

            x, y = float(p.get("x", -1)), float(p.get("y", -1))
            in_frame = (0 <= x <= W) and (0 <= y <= H)

            if not in_frame:
                return False, f"{bone_name} out of frame (x={x:.1f}, y={y:.1f})"
            else:
                return True, f"visible via {bone_name}"

        return False, "no head bones found in mixamo_pixels"

    # ── RTMLib 输出解析 ───────────────────────────────────────────

    @staticmethod
    def _parse_rtmlib_output(rt_out) -> dict[str, dict]:
        """
        从 RTMLib Wholebody 输出中提取面部5点。

        rt_out 可能是 tuple(keypoints, scores) 或直接是 ndarray。
        返回 {"nose": {"x":..,"y":..,"z":0.0}, ...}
        """
        kps = np.asarray(rt_out[0] if isinstance(rt_out, tuple) else rt_out)
        if kps.ndim == 2:
            kps = kps[None]

        person = kps[0]   # 取第一个人

        face5: dict[str, dict] = {}
        for name, idx in FACE5_IDX.items():
            if idx < len(person):
                x, y = float(person[idx][0]), float(person[idx][1])
                face5[name] = {"x": x, "y": y, "z": 0.0}

        return face5

    # ── 主推理接口 ────────────────────────────────────────────────

    def infer(
        self,
        img_rgb: np.ndarray,
        mixamo_pixels: Optional[dict[str, dict]],
        img_shape: Optional[tuple] = None,
    ) -> Face5Result:
        """
        推理一帧的面部5点。

        Parameters
        ----------
        img_rgb        : HxWx3 uint8 RGB 图像数组
        mixamo_pixels  : Mixamo 骨骼像素坐标字典（用于可见性判断）
                         若传 None 则跳过可见性检查，直接推理
        img_shape      : 图像尺寸 (H, W, C)，None 时从 img_rgb.shape 获取

        Returns
        -------
        Face5Result
        """
        if img_shape is None:
            img_shape = img_rgb.shape

        # ── 可见性检查 ───────────────────────────────────────────
        if self.skip_if_face_outside and mixamo_pixels is not None:
            visible, reason = self._is_face_region_visible(mixamo_pixels, img_shape)
            if not visible:
                return Face5Result(face5={}, skipped=True, reason=reason)

        # ── 模型懒加载（此处可能抛 RuntimeError） ────────────────
        self._ensure_model()

        # ── 推理 ─────────────────────────────────────────────────
        rt_out = self._model(img_rgb)
        face5  = self._parse_rtmlib_output(rt_out)

        return Face5Result(face5=face5, skipped=False, reason="inferred")