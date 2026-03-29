"""
visualize.py
============
Interactive skeleton visualization for AnimePoseDataset.

Usage (Jupyter cell):
    from visualize import show_skeleton_widget
    show_skeleton_widget(dataset)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import ipywidgets as widgets
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image

if TYPE_CHECKING:
    pass  # AnimePoseDataset type hint only; no hard import needed

# ---------------------------------------------------------------------------
# COCO-17 skeleton connection groups (color, [(from, to), ...])
# Face-5 points are included and connected to body landmarks correctly.
# ---------------------------------------------------------------------------
COCO17_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    # ── Face internal connections ──────────────────────────────────────────
    ("#FF6B6B", [
        ("nose",      "left_eye"),
        ("nose",      "right_eye"),
        ("left_eye",  "left_ear"),
        ("right_eye", "right_ear"),
    ]),
    # ── Face-5 to shoulders (cross-layer face→body) ────────────────────────
    # COCO standard: nose sits above the shoulder midpoint;
    # we draw nose→left_shoulder and nose→right_shoulder so the head
    # is anchored to the torso even when neck is absent.
    ("#FF9F43", [
        ("left_ear",  "left_shoulder"),
        ("right_ear", "right_shoulder"),
    ]),
    # ── Torso / spine ─────────────────────────────────────────────────────
    ("#FFD93D", [
        ("left_shoulder",  "right_shoulder"),
        ("left_shoulder",  "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip",       "right_hip"),
    ]),
    # ── Left limbs ────────────────────────────────────────────────────────
    ("#6BCB77", [
        ("left_shoulder", "left_elbow"),
        ("left_elbow",    "left_wrist"),
        ("left_hip",      "left_knee"),
        ("left_knee",     "left_ankle"),
    ]),
    # ── Right limbs ───────────────────────────────────────────────────────
    ("#4D96FF", [
        ("right_shoulder", "right_elbow"),
        ("right_elbow",    "right_wrist"),
        ("right_hip",      "right_knee"),
        ("right_knee",     "right_ankle"),
    ]),
]

FACE5_NAMES: frozenset[str] = frozenset(
    {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}
)

LEGEND_ITEMS = [
    patches.Patch(color="#FF6B6B", label="face (internal)"),
    patches.Patch(color="#FF9F43", label="face → body"),
    patches.Patch(color="#FFD93D", label="spine / center"),
    patches.Patch(color="#6BCB77", label="left limb"),
    patches.Patch(color="#4D96FF", label="right limb"),
    patches.Patch(color="#FFFFFF", label="body keypoint"),
    patches.Patch(color="#FF6B6B", label="face keypoint"),
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _xy(kps: dict, name: str, img_w: int, img_h: int,
        clip_to_frame: bool) -> tuple[float, float] | None:
    """
    Return (x, y) for keypoint *name* or None if invisible / out-of-frame.

    Parameters
    ----------
    clip_to_frame : bool
        If True, points outside [0, img_w] × [0, img_h] are treated as
        invisible (not drawn).  If False they are drawn even off-canvas.
    """
    p = kps.get(name)
    if p is None:
        return None
    x = float(p.get("x", 0))
    y = float(p.get("y", 0))
    v = int(p.get("visibility", 0))
    if v <= 0:
        return None
    if clip_to_frame and (x < 0 or x > img_w or y < 0 or y > img_h):
        return None
    return (x, y)


def _load_img(data: dict, dataset_root: Path) -> np.ndarray | None:
    p = dataset_root / data["image_path"]
    if not p.exists():
        return None
    return np.array(Image.open(p).convert("RGB"))


def _load_coco_json(view_obj, frame_id: int) -> dict | None:
    """
    Load the pre-exported coco17 JSON for *frame_id* via the view object.

    Strategy:
      1. Use view.coco_json_out_dir (already confirmed to be set) + standard
         filename pattern.  This avoids relying on frame.coco_json_out_path
         which may not be populated at query time.
    """
    d = view_obj.coco_json_out_dir
    if d is None:
        return None
    p = Path(d) / f"frame_{frame_id:04d}_coco17.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _frame_ids_for_view(view_obj) -> list[int]:
    """Collect available frame IDs from view.coco_json_out_dir."""
    d = view_obj.coco_json_out_dir
    if d is None:
        return []
    d = Path(d)
    if not d.exists():
        return []
    ids = []
    for f in sorted(d.glob("frame_*_coco17.json")):
        try:
            ids.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(ids)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_bbox(ax: plt.Axes, data: dict, color: str = "lime") -> None:
    b = data.get("bbox")
    if not b or all(float(v) == 0 for v in b):
        return
    x, y, w, h = (float(v) for v in b)
    ax.add_patch(patches.Rectangle(
        (x, y), w, h, lw=1.5, edgecolor=color, facecolor="none", zorder=4,
    ))


def _draw_skeleton(
    ax: plt.Axes,
    kps: dict,
    img_w: int,
    img_h: int,
    clip_to_frame: bool,
    lw: float = 2.2,
) -> None:
    for color, pairs in COCO17_GROUPS:
        for a_name, b_name in pairs:
            a = _xy(kps, a_name, img_w, img_h, clip_to_frame)
            b = _xy(kps, b_name, img_w, img_h, clip_to_frame)
            if a and b:
                ax.plot(
                    [a[0], b[0]], [a[1], b[1]],
                    color=color, lw=lw, alpha=0.92, zorder=3,
                )


def _draw_keypoints(
    ax: plt.Axes,
    kps: dict,
    img_w: int,
    img_h: int,
    clip_to_frame: bool,
    show_labels: bool,
) -> int:
    """Draw keypoints and return count of visible ones."""
    n_vis = 0
    for name in kps:
        xy = _xy(kps, name, img_w, img_h, clip_to_frame)
        if xy is None:
            continue
        n_vis += 1
        c = "#FF6B6B" if name in FACE5_NAMES else "#FFFFFF"
        ax.scatter(*xy, s=28, c=c, zorder=6, linewidths=0.5, edgecolors="#000")
        if show_labels:
            ax.text(
                xy[0] + 3, xy[1] - 3, name,
                fontsize=5.5, color="white", zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", fc="#000", alpha=0.45, ec="none"),
            )
    return n_vis


def _compose_frame(
    ax: plt.Axes,
    img: np.ndarray,
    data: dict,
    title: str,
    show_labels: bool,
    show_bbox: bool,
    clip_to_frame: bool,
    alpha: float = 1.0,
) -> int:
    """Draw image + skeleton overlay on *ax*. Returns n_visible."""
    H, W = img.shape[:2]
    ax.set_facecolor("#111111")
    ax.imshow(img, alpha=alpha)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    kps = data.get("keypoints", {})
    if show_bbox:
        _draw_bbox(ax, data)

    _draw_skeleton(ax, kps, W, H, clip_to_frame)
    n_vis = _draw_keypoints(ax, kps, W, H, clip_to_frame, show_labels)

    ax.set_title(
        f"{title}\nvisible={n_vis}",
        fontsize=8, color="white",
        bbox=dict(fc="#000", alpha=0.45, ec="none", pad=2),
    )
    ax.axis("off")
    return n_vis


def _compose_pure_skeleton(
    ax: plt.Axes,
    img_shape: tuple,
    data: dict,
    clip_to_frame: bool,
) -> None:
    """Black-background pure skeleton subplot."""
    H, W = img_shape[:2]
    ax.set_facecolor("#111111")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    kps = data.get("keypoints", {})
    _draw_skeleton(ax, kps, W, H, clip_to_frame, lw=2.4)

    for name in kps:
        xy = _xy(kps, name, W, H, clip_to_frame)
        if xy is None:
            continue
        c = "#FF6B6B" if name in FACE5_NAMES else "#FFFFFF"
        ax.scatter(*xy, s=22, c=c, zorder=6, linewidths=0.4, edgecolors="#555")

    ax.set_title(
        "pure skeleton",
        fontsize=8, color="white",
        bbox=dict(fc="#000", alpha=0.45, ec="none", pad=2),
    )
    ax.axis("off")


# ---------------------------------------------------------------------------
# Core render function (called by interactive_output)
# ---------------------------------------------------------------------------

def _render(
    dataset,
    all_models: list,
    split: str,
    pack_name: str,
    model_name: str,
    view_name: str,
    frame_id: int,
    show_labels: bool,
    show_bbox: bool,
    clip_to_frame: bool,
) -> None:
    # ── Locate model + view ────────────────────────────────────────────────
    mdl = next(
        (m for m in all_models
         if m.split == split and m.pack_name == pack_name and m.model_name == model_name),
        None,
    )
    if mdl is None:
        print(f"[ERROR] Model not found: {split}/{pack_name}/{model_name}")
        return

    view_obj = next(
        (v for v in mdl.iter_views() if v.view_name == view_name),
        None,
    )
    if view_obj is None:
        print(f"[ERROR] View not found: {view_name}")
        return

    # ── Available frames ───────────────────────────────────────────────────
    ids = _frame_ids_for_view(view_obj)
    if not ids:
        print(
            "[ERROR] No coco17 JSON found.\n"
            f"  Expected dir: {view_obj.coco_json_out_dir}\n"
            "  Please run the batch export cell first."
        )
        return

    if frame_id not in ids:
        frame_id = ids[0]

    idx     = ids.index(frame_id)
    prev_id = ids[idx - 1]               # wraps to last  when idx == 0
    next_id = ids[(idx + 1) % len(ids)]  # wraps to first when idx == last

    # ── Load JSON ──────────────────────────────────────────────────────────
    d_curr = _load_coco_json(view_obj, frame_id)
    if d_curr is None:
        print(f"[ERROR] Could not read JSON for frame {frame_id}.")
        return
    d_prev = _load_coco_json(view_obj, prev_id) or d_curr
    d_next = _load_coco_json(view_obj, next_id) or d_curr

    # ── Load images ────────────────────────────────────────────────────────
    root = Path(dataset.root)
    img_curr = _load_img(d_curr, root)
    if img_curr is None:
        print(f"[ERROR] Image not found: {d_curr.get('image_path')}")
        return

    _tmp = _load_img(d_prev, root)
    img_prev = _tmp if _tmp is not None else img_curr

    _tmp = _load_img(d_next, root)
    img_next = _tmp if _tmp is not None else img_curr

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.patch.set_facecolor("#1a1a1a")

    _compose_frame(
        axes[0], img_prev, d_prev,
        title=f"frame {prev_id}  (prev)",
        show_labels=False, show_bbox=show_bbox,
        clip_to_frame=clip_to_frame, alpha=0.5,
    )

    title_curr = f"frame {frame_id}  ◀ current"
    if d_curr.get("face5_info"):
        title_curr += f"  [{d_curr['face5_info']}]"
    _compose_frame(
        axes[1], img_curr, d_curr,
        title=title_curr,
        show_labels=show_labels, show_bbox=show_bbox,
        clip_to_frame=clip_to_frame, alpha=1.0,
    )

    _compose_frame(
        axes[2], img_next, d_next,
        title=f"frame {next_id}  (next)",
        show_labels=False, show_bbox=show_bbox,
        clip_to_frame=clip_to_frame, alpha=0.5,
    )

    _compose_pure_skeleton(axes[3], img_curr.shape, d_curr, clip_to_frame)

    fig.legend(
        handles=LEGEND_ITEMS,
        loc="lower center", ncol=7,
        fontsize=7.5, framealpha=0.5,
        facecolor="#333", labelcolor="white",
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.suptitle(
        f"{pack_name}  ·  {model_name}  ·  {view_name}  ·  coco17",
        fontsize=10, color="white", y=1.01,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Widget builder (public API)
# ---------------------------------------------------------------------------

def show_skeleton_widget(dataset) -> None:
    """
    Display an interactive skeleton visualization widget.

    Parameters
    ----------
    dataset : AnimePoseDataset
        The dataset object.  Must have `iter_models()`, `root`, and each
        model must expose `iter_views()` with `view.coco_json_out_dir`.
    """
    all_models = list(dataset.iter_models())

    # ── Option helpers ─────────────────────────────────────────────────────
    def _s_opts() -> list[str]:
        return sorted({m.split for m in all_models})

    def _p_opts(s: str) -> list[str]:
        return sorted({m.pack_name for m in all_models if m.split == s})

    def _m_opts(s: str, p: str) -> list[str]:
        return sorted(
            {m.model_name for m in all_models if m.split == s and m.pack_name == p}
        )

    def _v_opts(s: str, p: str, m: str) -> list[str]:
        mdl = next(
            (x for x in all_models
             if x.split == s and x.pack_name == p and x.model_name == m),
            None,
        )
        if mdl is None:
            return []
        return sorted(
            v.view_name
            for v in mdl.iter_views()
            if v.coco_json_out_dir is not None and Path(v.coco_json_out_dir).exists()
        )

    def _fids(s: str, p: str, m: str, v: str) -> list[int]:
        mdl = next(
            (x for x in all_models
             if x.split == s and x.pack_name == p and x.model_name == m),
            None,
        )
        if mdl is None:
            return [1]
        view_obj = next((vv for vv in mdl.iter_views() if vv.view_name == v), None)
        if view_obj is None:
            return [1]
        ids = _frame_ids_for_view(view_obj)
        return ids if ids else [1]

    # ── Initial values ─────────────────────────────────────────────────────
    s0   = _s_opts()[0]
    p0   = _p_opts(s0)[0]
    m0   = _m_opts(s0, p0)[0]
    v0   = (_v_opts(s0, p0, m0) or [""])[0]
    fids0 = _fids(s0, p0, m0, v0)

    # ── Widgets ────────────────────────────────────────────────────────────
    lw = widgets.Layout(width="180px")
    w_split  = widgets.Dropdown(options=_s_opts(),          value=s0, description="Split",  layout=lw)
    w_pack   = widgets.Dropdown(options=_p_opts(s0),        value=p0, description="Pack",   layout=widgets.Layout(width="260px"))
    w_model  = widgets.Dropdown(options=_m_opts(s0, p0),    value=m0, description="Model",  layout=widgets.Layout(width="200px"))
    w_view   = widgets.Dropdown(options=_v_opts(s0, p0, m0), value=v0, description="View",  layout=widgets.Layout(width="240px"))
    w_frame  = widgets.SelectionSlider(
        options=fids0, value=fids0[0],
        description="Frame",
        layout=widgets.Layout(width="440px"),
        style={"description_width": "50px"},
    )
    w_labels       = widgets.Checkbox(value=True,  description="显示标签")
    w_bbox         = widgets.Checkbox(value=True,  description="显示 BBox")
    w_clip         = widgets.Checkbox(value=True,  description="仅显示画面内特征点")

    # ── Cascade observers ─────────────────────────────────────────────────
    def _upd_pack(c):
        opts = _p_opts(c["new"])
        w_pack.options = opts
        if opts:
            w_pack.value = opts[0]

    def _upd_model(c):
        opts = _m_opts(w_split.value, c["new"])
        w_model.options = opts
        if opts:
            w_model.value = opts[0]

    def _upd_view(c):
        opts = _v_opts(w_split.value, w_pack.value, c["new"])
        w_view.options = opts
        if opts:
            w_view.value = opts[0]

    def _upd_frame(c):
        ids = _fids(w_split.value, w_pack.value, w_model.value, c["new"])
        w_frame.options = ids
        w_frame.value   = ids[0]

    w_split.observe(_upd_pack,  names="value")
    w_pack.observe(_upd_model,  names="value")
    w_model.observe(_upd_view,  names="value")
    w_view.observe(_upd_frame,  names="value")

    # ── Bind render ────────────────────────────────────────────────────────
    # Use a lambda wrapper so dataset and all_models are captured in closure.
    def _render_bound(split, pack_name, model_name, view_name, frame_id,
                      show_labels, show_bbox, clip_to_frame):
        _render(
            dataset=dataset,
            all_models=all_models,
            split=split,
            pack_name=pack_name,
            model_name=model_name,
            view_name=view_name,
            frame_id=frame_id,
            show_labels=show_labels,
            show_bbox=show_bbox,
            clip_to_frame=clip_to_frame,
        )

    out = widgets.interactive_output(
        _render_bound,
        {
            "split":         w_split,
            "pack_name":     w_pack,
            "model_name":    w_model,
            "view_name":     w_view,
            "frame_id":      w_frame,
            "show_labels":   w_labels,
            "show_bbox":     w_bbox,
            "clip_to_frame": w_clip,
        },
    )

    ui = widgets.VBox([
        widgets.HBox([w_split, w_pack, w_model]),
        widgets.HBox([w_view,  w_frame]),
        widgets.HBox([w_labels, w_bbox, w_clip]),
    ])

    display(ui, out)



"""
single frame preview (legacy, non-interactive)
"""

def _draw_bbox_frame(ax, bbox, color="lime", linewidth=2):
    """在 ax 上绘制 [x, y, w, h] 格式的 bbox。"""
    if bbox is None:
        return
    if len(bbox) != 4:
        return
    if all(float(v) == 0 for v in bbox):
        return

    x, y, w, h = bbox
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)


def _split_keypoints_for_draw(keypoints: dict):
    """
    把关键点拆成 face5 / body / invisible 三类。
    """
    face_pts = []
    body_pts = []
    invisible_pts = []

    for name, p in keypoints.items():
        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        v = int(p.get("visibility", 2 if (x != 0 or y != 0) else 0))

        item = (name, x, y, v)

        if v <= 0:
            invisible_pts.append(item)
        elif name in FACE5_NAMES:
            face_pts.append(item)
        else:
            body_pts.append(item)

    return face_pts, body_pts, invisible_pts


def preview_keypoints(
    frame_data_instance: AnimeSkeletonData,
    dataset_root: Path,
    show_labels: bool = True,
    show_invisible: bool = False,
    figsize=(8, 8),
    debug: bool = True,
):
    """
    纯预览函数,只显示 frame_data_instance.keypoints 中已经存在的点

    Parameters
    ----------
    frame_data_instance : AnimeSkeletonData
        要显示的样本对象
    dataset_root : Path
        数据集根目录
    show_labels : bool
        是否显示关键点名称
    show_invisible : bool
        是否显示 visibility=0 的点
    figsize : tuple
        图像显示大小
    debug : bool
        是否输出调试信息
    """
    sample = frame_data_instance

    if debug:
        print(f"[DEBUG] sample.uuid       = {sample.uuid}")
        print(f"[DEBUG] sample.sample_key = {sample.sample_key}")
        print(f"[DEBUG] sample.format     = {sample.pose_format}")
        print(f"[DEBUG] sample.image_path = {sample.image_path}")

    # ── 加载图像 ───────────────────────────────────────────────
    abs_img_path = Path(dataset_root) / sample.image_path
    if debug:
        print(f"[DEBUG] abs_img_path      = {abs_img_path}")

    if not abs_img_path.exists():
        print(f"[preview_keypoints] 图像文件不存在: {abs_img_path}")
        return

    img_rgb = np.array(Image.open(abs_img_path).convert("RGB"))

    # ── 读取关键点 ─────────────────────────────────────────────
    keypoints = sample.keypoints
    if not keypoints:
        print("[preview_keypoints] 当前样本没有 keypoints 可显示。")
        return

    face_pts, body_pts, invisible_pts = _split_keypoints_for_draw(keypoints)

    # ── 绘图 ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_rgb)

    # bbox
    _draw_bbox_frame(ax, sample.bbox, color="lime")

    # body 点
    if body_pts:
        xs = [x for _, x, _, _ in body_pts]
        ys = [y for _, _, y, _ in body_pts]
        ax.scatter(xs, ys, s=45, c="deepskyblue", label="body", zorder=5)

        if show_labels:
            for name, x, y, _ in body_pts:
                ax.text(
                    x + 3, y - 3,
                    name,
                    fontsize=7,
                    color="deepskyblue",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.55, ec="none"),
                )

    # face5 点
    if face_pts:
        xs = [x for _, x, _, _ in face_pts]
        ys = [y for _, _, y, _ in face_pts]
        ax.scatter(xs, ys, s=70, c="red", label="face5", zorder=6)

        if show_labels:
            for name, x, y, _ in face_pts:
                ax.text(
                    x + 4, y - 4,
                    name,
                    fontsize=8,
                    color="red",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"),
                )

    # invisible 点
    if show_invisible and invisible_pts:
        xs = [x for _, x, _, _ in invisible_pts]
        ys = [y for _, _, y, _ in invisible_pts]
        ax.scatter(xs, ys, s=35, c="gray", marker="x", label="invisible", zorder=4)

        if show_labels:
            for name, x, y, _ in invisible_pts:
                ax.text(
                    x + 2, y + 2,
                    name,
                    fontsize=6,
                    color="gray",
                    alpha=0.8,
                )

    # 标题
    title_lines = [
        f"{sample.view_name}  frame={sample.frame_id}  |  format={sample.pose_format}",
        f"sample_key={sample.sample_key}",
    ]
    if getattr(sample, "face5_info", None):
        title_lines.append(f"face5_info={sample.face5_info}")

    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.axis("off")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # abstract info
    visible_count = sum(
        1 for p in keypoints.values()
        if int(p.get("visibility", 2 if (p.get("x", 0) != 0 or p.get("y", 0) != 0) else 0)) > 0
    )

    print(f"[preview_keypoints] pose_format     : {sample.pose_format}")
    print(f"[preview_keypoints] total keypoints : {len(keypoints)}")
    print(f"[preview_keypoints] visible         : {visible_count}")
    if getattr(sample, "face5_info", None):
        print(f"[preview_keypoints] face5_info      : {sample.face5_info}")