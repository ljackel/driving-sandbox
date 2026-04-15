"""
Open-loop behavioral-cloning roll-out on the bird's-eye map.

**State:** ``(x, y)`` is on the main centerline when ``SIM_PROJECT_REF_ONTO_MAIN_ROAD`` is true: each step
advances **arc length** along the spline (decreasing ``y``), so the logged path cannot chord off the road.
``psi`` still integrates from the network (``psi += steering * SIM_YAW_RATE_GAIN * SIM_DT``) for the
BC state / CSV. The camera uses the spline tangent at ``y`` and lateral offset vs row (right lane, or
left-lane detour when ``ROADKILL_ENABLE`` on the **main** road only (off-ramp poses skip detour, matching dataset off-ramp crops).

**Control:** The network predicts steering from the crop; ``train.py`` supervises **output channel 0``.
With projection on, steering does **not** translate the centerline reference with ``cos/sin(psi)``—that
mixture caused off-road jumps when combined with ``x = cs(y)``. Set ``SIM_PROJECT_REF_ONTO_MAIN_ROAD``
false for unconstrained ``(x,y)`` integration (can leave the lane).

**Visualization:** The live BEV trail (pink/lavender polyline) follows the **right-lane** footprint, matching the scaled top-down ego car icon; **yellow bars** mark
clean **main-road** training sample locations (same ``y`` grid as ``generate_dataset``). When off-ramp
dataset rows are enabled, **orange bars** mark ramp train samples (same as ``train/offramp_*.jpg``:
uniform arc length along each ramp at the **main-road train mean spacing** ``δ``, on the in-train
BEV segment; capped by ``DATASET_OFFRAMP_*`` when ``DATASET_OFFRAMP_MATCH_MAIN_SPACING``).
With ``SIM_FP_VIDEO_ENABLE``,
each driver crop is written to ``sim_first_person.mp4`` (same preprocessing as the model when
``PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY`` is true: bottom half of the warp, resized to ``CAMERA_IMAGE_SIZE``).
With ``SIM_REALTIME_BEV`` / ``SIM_REALTIME_DRIVER_VIEW``, live OpenCV windows show the map trail and/or
the ego camera (same crop as the model). **Pause** freezes the roll-out (Space, ``p``, or the **Pause**
button on each window); **Resume** continues. While **paused** on the BEV, **drag the ego vehicle icon**
to snap the pose to another spot on the road (trail resets), then resume. Press ``q`` to stop, or **Ctrl+C** in the terminal
(focus may need to be on the terminal; live windows use short ``waitKey`` polls so Ctrl+C can interrupt).
Windows are tiled
left-to-right (BEV then driver) using ``SIM_REALTIME_WINDOW_*`` so they do not overlap.
The BEV view draws a **speed bar** at the bottom (drag with the mouse; see ``SIM_REALTIME_SPEED_*`` in config)
that scales distance per simulation step. Use ``SIM_REALTIME_STEP_PAUSE_MS`` (and ``SIM_REALTIME_BEV_WAIT_MS``)
to slow the live display (set pause to ``0`` and turn off BEV/driver windows or ``SIM_FP_VIDEO_ENABLE`` for headless speed).
With CUDA, ``SIM_CUDNN_BENCHMARK`` and optional ``SIM_TORCH_COMPILE`` reduce per-step inference cost.

**Input crop:** ``PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY`` matches ``train.py`` / ``evaluate_test.py`` (bottom
half of the warp, then resize to ``CAMERA_IMAGE_SIZE``).

**Intent / off-ramp:** ``take_offramp`` to the network follows ``SIM_TAKE_OFFRAMP`` or geographic
``SIM_TAKE_OFFRAMP_UPPER_HALF_NAV``. With ``SIM_PROJECT_REF_ONTO_MAIN_ROAD`` and off-ramps enabled, kinematics
merge onto a ramp at a branch ``y`` only when that intent is active (so e.g. exit 117 in the upper half can be
taken while skipping a lower-half exit).

**Camera / overlay heading:** the warp and BEV trail use the **road tangent** at ``(x, y)``—main spline or
ramp Bézier when on the ramp—not the integrated BC ``psi``. Otherwise ``psi`` drift places the right-lane
offset in the grass and the red path “jumps” off the road.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import config as cfg
from data_loader import count_train_test_examples, train_perturb_stats_from_labels
from dataset_split import (
    dataset_train_test_y,
    offramp_clean_counts_matching_main_spacing,
    offramp_train_u_rid_pairs_main_spacing,
)
from perspective_camera import perspective_camera_view
from reproducibility import set_global_seed

set_global_seed(cfg.TRAIN_SEED)

from driving_model import DrivingNet
from generate_world import DrivingWorld, lateral_offset_px_avoid_roadkill

# Centerline ``(x,y)``, integrated ``psi``, whether pose is on the off-ramp, Bézier ``u`` (else0).
SimPathPoint = tuple[float, float, float, bool, float, int]


def _sim_path_point(
    x: float,
    y: float,
    psi: float,
    *,
    on_ramp: bool = False,
    u_ramp: float = 0.0,
    ramp_id: int = 0,
) -> SimPathPoint:
    return (
        float(x),
        float(y),
        float(psi),
        bool(on_ramp),
        float(u_ramp),
        int(ramp_id),
    )


def _project_root() -> str:
    """Absolute path to the project directory containing ``simulate.py``."""
    return os.path.dirname(os.path.abspath(__file__))


def training_epochs_from_checkpoint(ckpt: str) -> int | None:
    """
    Total number of training epochs from ``training_log.json`` in the same directory as ``ckpt``.

    Returns ``None`` if there is no log (e.g. weights copied only to ``data/driving_net.pt``).
    """
    log_path = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "training_log.json")
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, encoding="utf-8") as f:
            data = json.load(f)
        return int(data["epochs"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def simulation_output_dir(ckpt: str) -> str:
    """
    Resolve where to write ``sim_path.png`` and ``ego_path.csv``.

    If ``ckpt`` lives under ``runs/<stamp>/``, returns that directory; otherwise ``data/``.
    """
    root = _project_root()
    ckpt_abs = os.path.normpath(os.path.abspath(ckpt))
    runs_abs = os.path.normpath(os.path.abspath(os.path.join(root, cfg.RUNS_DIR)))
    ckpt_dir = os.path.dirname(ckpt_abs)
    try:
        if os.path.commonpath([runs_abs, ckpt_dir]) == runs_abs:
            return ckpt_dir
    except ValueError:
        pass
    return os.path.join(root, cfg.DATA_DIR)


def latest_checkpoint_path() -> str | None:
    """
    Find the best ``driving_net.pt`` for roll-out / eval.

    Picks the newest checkpoint under ``runs/*/``. Uses ``data/<CHECKPOINT_FILENAME>`` only if it is
    **meaningfully newer** (see ``CHECKPOINT_PREFER_DATA_IF_NEWER_BY_SEC``) so a fresh train + copy
    to ``data/`` still resolves to the ``runs/`` copy when mtimes are tied or skewed slightly.
    """
    root = _project_root()
    runs = os.path.join(root, cfg.RUNS_DIR)
    best_path: str | None = None
    best_mtime = -1.0
    if os.path.isdir(runs):
        for name in os.listdir(runs):
            sub = os.path.join(runs, name)
            if not os.path.isdir(sub):
                continue
            w = os.path.join(sub, cfg.CHECKPOINT_FILENAME)
            if os.path.isfile(w):
                m = os.path.getmtime(w)
                if m > best_mtime:
                    best_mtime = m
                    best_path = w
    data_w = os.path.join(root, cfg.DATA_DIR, cfg.CHECKPOINT_FILENAME)
    if best_path is None:
        return data_w if os.path.isfile(data_w) else None
    if not os.path.isfile(data_w):
        return best_path
    data_m = os.path.getmtime(data_w)
    thr = float(cfg.CHECKPOINT_PREFER_DATA_IF_NEWER_BY_SEC)
    if data_m > best_mtime + thr:
        return data_w
    return best_path


def _put_outlined_lines_bgr(
    img: np.ndarray,
    lines: list[str],
    org_xy: tuple[int, int],
    line_step_px: int = 14,
) -> None:
    """Stacked white labels with black outline (in-place); same style as ``generate_dataset`` overlays."""
    x0, y0 = org_xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = cfg.ANNOT_FONT_SCALE, cfg.ANNOT_FONT_THICKNESS
    outline_offsets = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
    )
    for i, label in enumerate(lines):
        x, y = x0, y0 + i * line_step_px
        for dx, dy in outline_offsets:
            cv2.putText(
                img,
                label,
                (x + dx, y + dy),
                font,
                scale,
                (0, 0, 0),
                thick + 1,
                cv2.LINE_AA,
            )
        cv2.putText(img, label, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def get_view_from_pose(
    world_bgr: np.ndarray,
    x: float,
    y: float,
    psi: float,
    lateral_offset_px: float,
) -> np.ndarray | None:
    """
    Warp a square forward-facing crop from the BEV map for pose ``(x, y)`` and heading ``psi``.

    Args:
        world_bgr: Full bird's-eye map (BGR).
        x, y: Vehicle reference point in pixel coordinates.
        psi: Heading (rad); velocity is ``(cos psi, sin psi)`` in image axes (+x right, +y down).
        lateral_offset_px: Shift camera along driver's right (same convention as training).

    Returns:
        ``CAMERA_IMAGE_SIZE`` square BGR view, or ``None`` if the source quad leaves the image.
    """
    fx, fy = float(np.cos(psi)), float(np.sin(psi))
    rx, ry = float(-np.sin(psi)), float(np.cos(psi))
    f = np.array([fx, fy], dtype=np.float32)
    r = np.array([rx, ry], dtype=np.float32)

    near_c = np.array([x, y], dtype=np.float32) + r * float(lateral_offset_px)
    return perspective_camera_view(world_bgr, near_c, f, r)


def _fp_video_frame_bgr(view: np.ndarray) -> np.ndarray:
    """BGR frame for MP4: full warp or bottom-half crop resized to ``CAMERA_IMAGE_SIZE`` (matches model)."""
    if not cfg.PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY:
        return view
    h0, w0 = view.shape[:2]
    crop = view[h0 // 2 : h0, 0:w0]
    return cv2.resize(
        crop,
        (cfg.CAMERA_IMAGE_SIZE, cfg.CAMERA_IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )


def preprocess_bgr_for_model(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert BGR uint8 warp output to a normalized NCHW float tensor on ``device``.

    Applies the same optional bottom-half crop + square resize as ``prepare_perspective_pil_for_model``
    + ``train`` transforms. Uses ``NORMALIZE_MEAN`` / ``NORMALIZE_STD``.
    """
    if cfg.PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY:
        h0, w0 = bgr.shape[:2]
        bgr = bgr[h0 // 2 : h0, 0:w0]
        bgr = cv2.resize(
            bgr,
            (cfg.CAMERA_IMAGE_SIZE, cfg.CAMERA_IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    for c in range(3):
        t[c] = (t[c] - cfg.NORMALIZE_MEAN[c]) / cfg.NORMALIZE_STD[c]
    return t.unsqueeze(0).to(device)


def initial_heading_road_aligned(cs, y: float) -> float:
    """
    Heading (rad) aligned with the road tangent, direction of travel toward decreasing ``y`` (up the map).

    Args:
        cs: ``scipy.interpolate.CubicSpline`` for centerline ``x(y)``.
        y: Sample row (pixels).
    """
    dxdy = float(cs(y, nu=1))
    norm = float(np.hypot(dxdy, 1.0))
    fx = -dxdy / norm
    fy = -1.0 / norm
    return float(np.arctan2(fy, fx))


def _blend_psi_toward(psi: float, psi_tgt: float, w: float) -> float:
    """Circular blend of headings; ``w`` in ``[0, 1]`` is weight on ``psi_tgt``."""
    w = float(np.clip(w, 0.0, 1.0))
    if w <= 0.0:
        return psi
    return float(
        np.arctan2(
            (1.0 - w) * np.sin(psi) + w * np.sin(psi_tgt),
            (1.0 - w) * np.cos(psi) + w * np.cos(psi_tgt),
        )
    )


def _ego_lateral_offset_px_at_y(
    dw: DrivingWorld,
    y: float,
    world_bgr: np.ndarray | None = None,
    x_center: float | None = None,
    psi: float | None = None,
    *,
    on_ramp: bool = False,
) -> float:
    """Camera/ego offset from centerline (px); roadkill detour applies only on the main road."""
    if on_ramp:
        base = float(cfg.offramp_camera_lateral_offset_px(dw.px_per_m))
    else:
        base = float(cfg.SIM_EGO_LATERAL_OFFSET_M) * float(dw.px_per_m)
    return lateral_offset_px_avoid_roadkill(
        float(y),
        base,
        for_training_labels=False,
        world_bgr=world_bgr,
        dw=dw,
        x_center=x_center,
        psi=psi,
        on_main_road=not on_ramp,
    )


def _draw_lateral_sampling_bar(
    vis: np.ndarray,
    qx: float,
    qy: float,
    psi: float,
    bar_half: float,
    bgr: tuple[int, int, int],
) -> None:
    """Short segment through the right-lane point, along the vehicle lateral axis (dataset viz)."""
    rx = float(-np.sin(psi))
    ry = float(np.cos(psi))
    x1 = int(round(qx - bar_half * rx))
    y1 = int(round(qy - bar_half * ry))
    x2 = int(round(qx + bar_half * rx))
    y2 = int(round(qy + bar_half * ry))
    cv2.line(vis, (x1, y1), (x2, y2), bgr, 1, cv2.LINE_AA)


def _draw_train_sampling_bars_on_bev(
    vis: np.ndarray,
    dw: DrivingWorld,
    train_y: np.ndarray,
) -> None:
    """
    Draw yellow bars at clean **main-road** training sample locations on the BEV (ego lateral vs row).
    """
    bar_half = max(8.0, 0.35 * float(cfg.LANE_WIDTH_METERS * dw.px_per_m))
    yellow_bgr = (0, 255, 255)
    for yf in train_y:
        yf = float(yf)
        xc = float(dw.get_road_center(yf))
        psi = initial_heading_road_aligned(dw.cs, yf)
        lat = _ego_lateral_offset_px_at_y(
            dw, yf, vis, float(xc), float(psi)
        )
        qx, qy = _right_lane_overlay_xy(xc, yf, psi, lat)
        _draw_lateral_sampling_bar(vis, qx, qy, psi, bar_half, yellow_bgr)


def _draw_offramp_train_sampling_bars_on_bev(
    vis: np.ndarray,
    dw: DrivingWorld,
    n_ramp_train_frames: int,
    lateral_offset_px: float,
    *,
    num_main_train: int,
) -> None:
    """
    Draw orange bars where ``generate_dataset`` places off-ramp train crops: uniform arc length along
    each ramp at the same mean spacing as main-road train rows (within the train BEV band).
    """
    if n_ramp_train_frames <= 0 or dw.offramp_num() <= 0:
        return
    bar_half = max(8.0, 0.35 * float(cfg.LANE_WIDTH_METERS * dw.px_per_m))
    orange_bgr = (0, 200, 255)
    n_r = int(n_ramp_train_frames)
    n_br = max(1, dw.offramp_num())
    pairs = offramp_train_u_rid_pairs_main_spacing(
        dw,
        int(dw.size),
        int(cfg.DATASET_MAP_MARGIN),
        n_br,
        int(num_main_train),
        n_r,
        uniform_along_ramp=cfg.DATASET_SAMPLE_UNIFORM_ALONG_ROAD,
    )
    for u, rid in pairs:
        ev = dw.offramp_bezier_evolution(u, rid)
        if ev is None:
            continue
        B, (fx, fy), _ = ev
        px, py = float(B[0]), float(B[1])
        psi = float(np.arctan2(fy, fx))
        qx, qy = _right_lane_overlay_xy(px, py, psi, lateral_offset_px)
        _draw_lateral_sampling_bar(vis, qx, qy, psi, bar_half, orange_bgr)


def _rt_speed_scale_from_trackbar_pos(pos: int) -> float:
    c = float(cfg.SIM_REALTIME_SPEED_TRACKBAR_CENTER)
    lo = float(cfg.SIM_REALTIME_SPEED_SCALE_MIN)
    hi = float(cfg.SIM_REALTIME_SPEED_SCALE_MAX)
    return float(np.clip(float(pos) / c, lo, hi))


def _rt_speed_slider_set_from_mx(ui: dict, mx: int, sx0: int, sx1: int) -> None:
    denom = float(max(1, sx1 - sx0))
    t = (float(mx) - float(sx0)) / denom
    t = float(np.clip(t, 0.0, 1.0))
    tmax = float(cfg.SIM_REALTIME_SPEED_TRACKBAR_MAX)
    pos = int(round(t * tmax))
    pos = int(np.clip(pos, 0, int(cfg.SIM_REALTIME_SPEED_TRACKBAR_MAX)))
    ui["speed_track_pos"] = pos
    ui["speed_scale"] = _rt_speed_scale_from_trackbar_pos(pos)


def _rt_draw_speed_slider(img: np.ndarray, ui: dict) -> None:
    """Draw a draggable speed control strip at the bottom of the BEV (HighGUI trackbars are often invisible)."""
    h, w = img.shape[:2]
    pad = 16
    sy1 = h - 8
    sy0 = h - 36
    sx0 = pad
    sx1 = w - pad
    if sx1 <= sx0 + 20 or sy1 <= sy0:
        ui["speed_slider_rect"] = (-1, -1, -2, -2)
        return
    ui["speed_slider_rect"] = (sx0, sy0, sx1, sy1)
    cv2.rectangle(img, (sx0, sy0), (sx1, sy1), (35, 35, 42), -1, cv2.LINE_AA)
    cv2.rectangle(img, (sx0, sy0), (sx1, sy1), (160, 160, 170), 1, cv2.LINE_AA)
    pos = int(
        np.clip(
            int(ui.get("speed_track_pos", cfg.SIM_REALTIME_SPEED_TRACKBAR_DEFAULT)),
            0,
            int(cfg.SIM_REALTIME_SPEED_TRACKBAR_MAX),
        )
    )
    span = float(max(1, sx1 - sx0))
    tx = int(round(sx0 + span * (pos / float(cfg.SIM_REALTIME_SPEED_TRACKBAR_MAX))))
    tx = int(np.clip(tx, sx0 + 6, sx1 - 6))
    cy = (sy0 + sy1) // 2
    cv2.circle(img, (tx, cy), 8, (55, 55, 62), -1, cv2.LINE_AA)
    cv2.circle(img, (tx, cy), 8, (120, 255, 140), 2, cv2.LINE_AA)
    sc = float(ui.get("speed_scale", 1.0))
    cap = f"Sim speed {sc:.2f}x  (drag bar)"
    cv2.putText(
        img,
        cap,
        (sx0, sy0 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 255, 215),
        1,
        cv2.LINE_AA,
    )


def _bev_ego_car_display_extents_px(px_per_m: float) -> tuple[float, float]:
    """Length and width (px) for the BEV ego icon; preserves 3 m × 1.5 m aspect, optional floor."""
    len_px = float(cfg.SIM_BEV_EGO_CAR_LENGTH_M) * float(px_per_m)
    wid_px = float(cfg.SIM_BEV_EGO_CAR_WIDTH_M) * float(px_per_m)
    mn = float(cfg.SIM_BEV_EGO_CAR_MIN_DISPLAY_LEN_PX)
    if mn > 0.0 and len_px < mn:
        s = mn / max(len_px, 1e-6)
        len_px *= s
        wid_px *= s
    return len_px, wid_px


def _bev_ego_car_polygon_xy(
    cx: float,
    cy: float,
    psi: float,
    len_px: float,
    wid_px: float,
) -> np.ndarray:
    """
    Closed quadrilateral: vehicle center ``(cx,cy)``, heading ``psi`` (forward = ``(cos ψ, sin ψ)``),
    half-length ``len_px/2`` along forward, half-width ``wid_px/2`` along driver's right ``(-sin ψ, cos ψ)``.
    """
    fx = float(np.cos(psi))
    fy = float(np.sin(psi))
    rx = float(-np.sin(psi))
    ry = float(np.cos(psi))
    hl = 0.5 * float(len_px)
    hw = 0.5 * float(wid_px)
    corners = np.array(
        [
            [cx + hl * fx + hw * rx, cy + hl * fy + hw * ry],
            [cx + hl * fx - hw * rx, cy + hl * fy - hw * ry],
            [cx - hl * fx - hw * rx, cy - hl * fy - hw * ry],
            [cx - hl * fx + hw * rx, cy - hl * fy + hw * ry],
        ],
        dtype=np.float64,
    )
    return np.round(corners).astype(np.int32).reshape(-1, 1, 2)


def _draw_bev_ego_car_icon(
    vis: np.ndarray,
    cx: float,
    cy: float,
    psi: float,
    px_per_m: float,
    *,
    body_bgr: tuple[int, int, int] = (55, 200, 75),
    outline_bgr: tuple[int, int, int] = (255, 255, 255),
    axis_bgr: tuple[int, int, int] = (120, 255, 200),
) -> None:
    """Top-down car: body fill, outline, thin hood line toward forward."""
    len_px, wid_px = _bev_ego_car_display_extents_px(px_per_m)
    poly = _bev_ego_car_polygon_xy(cx, cy, psi, len_px, wid_px)
    cv2.fillConvexPoly(vis, poly, body_bgr)
    cv2.polylines(vis, [poly], True, outline_bgr, 1, cv2.LINE_AA)
    fx = float(np.cos(psi))
    fy = float(np.sin(psi))
    hx0 = cx - 0.22 * len_px * fx
    hy0 = cy - 0.22 * len_px * fy
    hx1 = cx + 0.38 * len_px * fx
    hy1 = cy + 0.38 * len_px * fy
    cv2.line(
        vis,
        (int(round(hx0)), int(round(hy0))),
        (int(round(hx1)), int(round(hy1))),
        axis_bgr,
        1,
        cv2.LINE_AA,
    )


def _bev_draw_nav_instruction_box(
    vis: np.ndarray, text: str, *, x0: int = 10, y0: int = 8
) -> int:
    """Filled + bordered box in the upper-left; returns y just below the box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thick = 2
    pad_x, pad_y = 10, 8
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x1, y1 = x0, y0
    y_baseline = y1 + pad_y + th
    x2 = x1 + tw + 2 * pad_x
    y2 = y_baseline + baseline + pad_y
    h, w = vis.shape[:2]
    x2 = min(x2, w - 8)
    y2 = min(y2, h - 8)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (38, 42, 48), -1, cv2.LINE_AA)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        text,
        (x1 + pad_x, y_baseline),
        font,
        scale,
        (248, 252, 255),
        thick,
        cv2.LINE_AA,
    )
    return int(y2)


def _bev_nav_exit_active(y_img: float, h_map: int) -> bool:
    if not bool(cfg.SIM_TAKE_OFFRAMP_UPPER_HALF_NAV):
        return False
    return float(y_img) <= 0.5 * float(h_map)


def _bev_realtime_frame(
    world_bgr: np.ndarray,
    dw: DrivingWorld,
    path: list[SimPathPoint],
    x: float,
    y: float,
    psi_draw: float,
    lateral_px: float,
    *,
    extra_hint: str | None = None,
    nav_exit_active: bool = False,
    nav_exit_text: str = "take the next exit",
) -> np.ndarray:
    """Bird's-eye frame: trail and ego in the **right lane** (not centerline)."""
    vis = world_bgr.copy()
    if len(path) >= 2:
        n = len(path)
        qs = np.empty((n, 2), dtype=np.float32)
        for i, p in enumerate(path):
            xc, yc, _psi, on_r, ur, rid = p
            psi_d = _path_point_psi_draw(dw, xc, yc, on_r, ur, rid)
            lat_i = _ego_lateral_offset_px_at_y(
                dw,
                float(yc),
                world_bgr,
                float(xc),
                float(psi_d),
                on_ramp=bool(on_r),
            )
            qx, qy = _right_lane_overlay_xy(float(xc), float(yc), psi_d, lat_i)
            qs[i, 0] = qx
            qs[i, 1] = qy
        pts = qs.round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, (200, 200, 255), 2, cv2.LINE_AA)
    qx, qy = _right_lane_overlay_xy(float(x), float(y), float(psi_draw), lateral_px)
    _draw_bev_ego_car_icon(vis, float(qx), float(qy), float(psi_draw), float(dw.px_per_m))
    below_nav = 0
    if nav_exit_active and nav_exit_text:
        below_nav = _bev_draw_nav_instruction_box(vis, nav_exit_text)
    line1_y = 26 if below_nav == 0 else below_nav + 18
    line1 = f"step {len(path) - 1}   Space pause   q quit   Ctrl+C (terminal)"
    cv2.putText(
        vis,
        line1,
        (10, line1_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if extra_hint:
        hint_y = line1_y + 26
        cv2.putText(
            vis,
            extra_hint,
            (10, hint_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 220),
            2,
            cv2.LINE_AA,
        )
    return vis


def _rt_draw_pause_button(
    img: np.ndarray, *, paused: bool
) -> tuple[int, int, int, int]:
    """Draw a compact Pause / Resume control in the top-right; returns hit-test ``(x1,y1,x2,y2)``."""
    h, w = img.shape[:2]
    label = "Resume" if paused else "Pause"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thick = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    pad_x, pad_y = 3, 2
    margin = 5
    bw = tw + 2 * pad_x
    bh = th + 2 * pad_y
    x2 = w - margin
    x1 = max(margin, x2 - bw)
    y1 = margin
    y2 = min(h - margin, y1 + bh)
    if y2 <= y1:
        return (-1, -1, -2, -2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (45, 45, 45), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (220, 220, 220), 1, cv2.LINE_AA)
    tx = int(x1 + (x2 - x1 - tw) // 2)
    ty = int(y1 + (y2 - y1 + th) // 2)
    cv2.putText(
        img,
        label,
        (tx, ty),
        font,
        scale,
        (240, 240, 240),
        thick,
        cv2.LINE_AA,
    )
    return x1, y1, x2, y2


def _rt_draw_paused_banner(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    cv2.putText(
        img,
        "PAUSED",
        (max(8, w // 2 - 120), max(36, h // 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )


def _rt_annotate_driver_view(drv: np.ndarray, steering: float) -> None:
    """Overlay steering readout and hints on the live driver crop."""
    dh, drv_w = drv.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    steer_scale = max(0.28, min(0.5, float(dh) / 320.0))
    hint_scale = max(0.24, min(0.42, float(dh) / 360.0))
    thick = 1
    steer_txt = f"s{steering:+.3f}"
    (tw, _), _ = cv2.getTextSize(steer_txt, font, steer_scale, thick)
    if tw > drv_w - 4:
        steer_scale *= (drv_w - 4) / float(max(tw, 1))
    y_steer = max(13, int(round(0.14 * dh)))
    y_hint = min(dh - 3, max(y_steer + 8, int(round(0.26 * dh))))
    cv2.putText(
        drv,
        steer_txt,
        (2, y_steer),
        font,
        steer_scale,
        (0, 255, 0),
        thick,
        cv2.LINE_AA,
    )
    cv2.putText(
        drv,
        "Space/btn pause  q quit  Ctrl+C",
        (2, y_hint),
        font,
        hint_scale,
        (255, 255, 255),
        thick,
        cv2.LINE_AA,
    )


def _rt_mouse_bev(event: int, x: int, y: int, flags: int, ui: dict) -> None:
    if int(ui.get("sim_step", 0)) < 2:
        return
    sr = ui.get("speed_slider_rect")
    if sr is not None and sr[2] > sr[0] and sr[3] > sr[1]:
        sx0, sy0, sx1, sy1 = sr
        if event == cv2.EVENT_LBUTTONDOWN:
            if sx0 <= x <= sx1 and sy0 <= y <= sy1:
                ui["speed_drag"] = True
                _rt_speed_slider_set_from_mx(ui, x, sx0, sx1)
                return
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if ui.get("speed_drag"):
                _rt_speed_slider_set_from_mx(ui, x, sx0, sx1)
                return
        elif event == cv2.EVENT_LBUTTONUP:
            if ui.get("speed_drag"):
                ui["speed_drag"] = False
                _rt_speed_slider_set_from_mx(ui, x, sx0, sx1)
                return
    paused = bool(ui.get("rt_sim_paused", False))
    ctx = ui.get("relocate_ctx")
    hit = ui.get("ego_hit_xy")
    grab_r = float(ui.get("ego_grab_radius_px", _RT_EGO_GRAB_RADIUS_PX))
    if paused and ctx is not None and hit is not None and hit[0] >= 0:
        px1, py1, px2, py2 = ui.get("bev_rect", (-1, -1, -2, -2))
        on_pause_btn = px2 > px1 and py2 > py1 and px1 <= x <= px2 and py1 <= y <= py2
        if event == cv2.EVENT_LBUTTONDOWN and not ui.get("speed_drag"):
            if on_pause_btn:
                return
            hx, hy = hit
            if (x - hx) ** 2 + (y - hy) ** 2 <= grab_r**2:
                ui["ego_drag"] = True
                pose = _bev_reloc_snap_validated(
                    ctx["dw"],
                    ctx["world"],
                    float(x),
                    float(y),
                    int(ctx["h"]),
                    float(ctx["margin"]),
                    bool(ctx["ramp_kinematics"]),
                )
                if pose is not None:
                    ui["reloc_preview_pose"] = pose
            return
        if event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON) and ui.get(
            "ego_drag"
        ):
            pose = _bev_reloc_snap_validated(
                ctx["dw"],
                ctx["world"],
                float(x),
                float(y),
                int(ctx["h"]),
                float(ctx["margin"]),
                bool(ctx["ramp_kinematics"]),
            )
            if pose is not None:
                ui["reloc_preview_pose"] = pose
            return
        if event == cv2.EVENT_LBUTTONUP and ui.get("ego_drag"):
            ui["ego_drag"] = False
            pose = _bev_reloc_snap_validated(
                ctx["dw"],
                ctx["world"],
                float(x),
                float(y),
                int(ctx["h"]),
                float(ctx["margin"]),
                bool(ctx["ramp_kinematics"]),
            )
            ui.pop("reloc_preview_pose", None)
            if pose is not None:
                ui["reloc_pending"] = pose
            return
    # Pause: ``LBUTTONUP`` avoids spurious ``DOWN`` when the window grabs focus.
    if event != cv2.EVENT_LBUTTONUP:
        return
    x1, y1, x2, y2 = ui["bev_rect"]
    if x2 <= x1 or y2 <= y1:
        return
    if x1 <= x <= x2 and y1 <= y <= y2:
        ui["toggle"] = True


def _rt_mouse_drv(event: int, x: int, y: int, flags: int, ui: dict) -> None:
    if event != cv2.EVENT_LBUTTONUP:
        return
    if int(ui.get("sim_step", 0)) < 2:
        return
    x1, y1, x2, y2 = ui["drv_rect"]
    if x2 <= x1 or y2 <= y1:
        return
    if x1 <= x <= x2 and y1 <= y <= y2:
        ui["toggle"] = True


def _rt_wait_key_interruptible(delay_ms: int, interrupt_flag: list) -> int:
    """
    Like ``cv2.waitKey(delay_ms)`` but avoids a single unbounded native wait: when
    ``delay_ms <= 0`` (step-through mode), polls in chunks so Python can handle Ctrl+C.
    """
    if interrupt_flag[0]:
        return -1
    chunk_ms = 30
    if delay_ms <= 0:
        while not interrupt_flag[0]:
            key_raw = int(cv2.waitKey(chunk_ms))
            if key_raw >= 0:
                return key_raw
        return -1
    waited = 0
    while waited < delay_ms and not interrupt_flag[0]:
        step = int(min(chunk_ms, delay_ms - waited))
        key_raw = int(cv2.waitKey(max(1, step)))
        if key_raw >= 0:
            return key_raw
        waited += step
    return -1


def _tile_realtime_sim_windows(
    fr_bev: np.ndarray | None,
    fr_drv: np.ndarray | None,
) -> None:
    """Place BEV left, driver camera right, using image widths so the frames do not overlap."""
    ox = int(cfg.SIM_REALTIME_WINDOW_ORIGIN_X)
    oy = int(cfg.SIM_REALTIME_WINDOW_ORIGIN_Y)
    gap = int(cfg.SIM_REALTIME_WINDOW_GAP_PX)
    x = ox
    if fr_bev is not None:
        cv2.moveWindow(cfg.SIM_REALTIME_BEV_WINDOW, x, oy)
        x += int(fr_bev.shape[1]) + gap
    if fr_drv is not None:
        cv2.moveWindow(cfg.SIM_REALTIME_DRIVER_WINDOW, x, oy)


def _sim_arc_y_bounds(h: int, margin: float) -> tuple[float, float]:
    """
    Vertical range for centerline arc integration: top inset ``margin`` (same as dataset / early stop),
    bottom row ``h - 1`` so the ego can start at the map bottom (unlike ``h - margin``, which caps ``y`` in the train grid only).
    """
    return float(margin), float(h - 1)


def _right_lane_polyline_xy_chord(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    lateral_px: float,
    *,
    n: int = 24,
) -> np.ndarray:
    """Right-lane samples along the straight chord between two BEV centerline points (main or ramp)."""
    n = max(2, int(n))
    dx, dy = float(x1 - x0), float(y1 - y0)
    psi = float(np.arctan2(dy, dx))
    pts = np.empty((n, 2), dtype=np.float32)
    for k in range(n):
        t = k / (n - 1) if n > 1 else 0.0
        xi = x0 + t * dx
        yi = y0 + t * dy
        qx, qy = _right_lane_overlay_xy(xi, yi, psi, lateral_px)
        pts[k, 0] = qx
        pts[k, 1] = qy
    return pts


def _right_lane_polyline_along_centerline_y(
    dw: DrivingWorld,
    lateral_px: float,
    y_from: float,
    y_to: float,
    *,
    n: int = 24,
) -> np.ndarray:
    """
    Sample the right-lane footprint along ``x = cs(y)`` between two rows (for BEV drawing).

    Straight chords between sparse poses cut across the grass on sharp bends; this follows the road.
    """
    n = max(2, int(n))
    pts = np.empty((n, 2), dtype=np.float32)
    for k in range(n):
        t = k / (n - 1) if n > 1 else 0.0
        yi = float(y_from + t * (y_to - y_from))
        xi = float(dw.get_road_center(yi))
        pr = initial_heading_road_aligned(dw.cs, yi)
        qx, qy = _right_lane_overlay_xy(xi, yi, pr, lateral_px)
        pts[k, 0] = qx
        pts[k, 1] = qy
    return pts


def _right_lane_overlay_xy(
    x: float,
    y: float,
    psi: float,
    lateral_offset_px: float,
) -> tuple[float, float]:
    """
    Map centerline reference ``(x, y)`` and heading ``psi`` to the lateral camera/ego point on the BEV.

    Uses the same right vector as ``get_view_from_pose``: ``(-sin ψ, cos ψ)``.
    """
    rx = float(-np.sin(psi))
    ry = float(np.cos(psi))
    return x + rx * lateral_offset_px, y + ry * lateral_offset_px


def _path_point_psi_draw(
    dw: DrivingWorld,
    x: float,
    y: float,
    on_ramp: bool,
    u_ramp: float,
    ramp_id: int = 0,
) -> float:
    """Road tangent at a logged centerline pose (matches live ``psi_draw`` / camera heading)."""
    if on_ramp:
        ev = dw.offramp_bezier_evolution(float(u_ramp), int(ramp_id))
        if ev is None:
            return float(initial_heading_road_aligned(dw.cs, y))
        _, (fx, fy), _ = ev
        return float(np.arctan2(fy, fx))
    return float(initial_heading_road_aligned(dw.cs, y))


# BEV relocate: minimum hit radius (px) when the ego icon is tiny on the map.
_RT_EGO_GRAB_RADIUS_PX = 28.0


def _bev_ego_grab_radius_px(px_per_m: float) -> float:
    """Hit radius for drag-relocate: covers the car footprint plus a small margin."""
    len_px, wid_px = _bev_ego_car_display_extents_px(px_per_m)
    diag = float(np.hypot(0.5 * len_px, 0.5 * wid_px))
    return max(float(_RT_EGO_GRAB_RADIUS_PX), diag + 8.0)


def _snap_main_right_lane_nearest(
    dw: DrivingWorld,
    mx: float,
    my: float,
    h: int,
    margin: float,
    world_bgr: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """
    Closest main-road **right-lane** footprint to ``(mx, my)`` in BEV px.

    Returns ``(x_centerline, y, psi_road, d2)`` or ``None``.
    """
    y_lo, y_hi = _sim_arc_y_bounds(h, margin)
    n = 320
    ys = np.linspace(y_lo, y_hi, n, dtype=np.float64)
    qxs = np.empty(n, dtype=np.float64)
    qys = np.empty(n, dtype=np.float64)
    for i, yi in enumerate(ys):
        yi = float(yi)
        xc = float(dw.get_road_center(yi))
        pr = initial_heading_road_aligned(dw.cs, yi)
        lat = _ego_lateral_offset_px_at_y(dw, yi, world_bgr, xc, pr)
        qx, qy = _right_lane_overlay_xy(xc, yi, pr, lat)
        qxs[i] = qx
        qys[i] = qy
    d2 = (qxs - mx) ** 2 + (qys - my) ** 2
    idx = int(np.argmin(d2))
    best_y = float(ys[idx])
    best_d2 = float(d2[idx])
    step = float(ys[1] - ys[0]) if n > 1 else 1.0
    y_lo2 = max(y_lo, best_y - 2.0 * step)
    y_hi2 = min(y_hi, best_y + 2.0 * step)
    for yi in np.linspace(y_lo2, y_hi2, 48, dtype=np.float64):
        yi = float(yi)
        xc = float(dw.get_road_center(yi))
        pr = initial_heading_road_aligned(dw.cs, yi)
        lat = _ego_lateral_offset_px_at_y(dw, yi, world_bgr, xc, pr)
        qx, qy = _right_lane_overlay_xy(xc, yi, pr, lat)
        d2i = (qx - mx) ** 2 + (qy - my) ** 2
        if d2i < best_d2:
            best_d2 = float(d2i)
            best_y = yi
    xc_f = float(dw.get_road_center(best_y))
    psi_f = initial_heading_road_aligned(dw.cs, best_y)
    return xc_f, best_y, psi_f, best_d2


def _snap_ramp_right_lane_nearest(
    dw: DrivingWorld,
    mx: float,
    my: float,
    world_bgr: np.ndarray,
) -> tuple[float, float, float, float, float, int] | None:
    """
    Closest off-ramp **right-lane** footprint to ``(mx, my)`` over all ramp branches.

    Returns ``(u, x_centerline, y, psi_road, d2, ramp_id)`` or ``None``.
    """
    if dw.offramp_num() <= 0:
        return None
    n = 220
    us = np.linspace(0.0, 1.0, n, dtype=np.float64)
    best_d2 = 1e30
    best: tuple[float, float, float, float, int] | None = None
    for rid in range(dw.offramp_num()):
        for u in us:
            u = float(u)
            ev = dw.offramp_bezier_evolution(u, rid)
            if ev is None:
                continue
            B, (fx, fy), _ = ev
            px, py = float(B[0]), float(B[1])
            pr = float(np.arctan2(fy, fx))
            lat = _ego_lateral_offset_px_at_y(
                dw, py, world_bgr, px, pr, on_ramp=True
            )
            qx, qy = _right_lane_overlay_xy(px, py, pr, lat)
            d2 = (qx - mx) ** 2 + (qy - my) ** 2
            if d2 < best_d2:
                best_d2 = float(d2)
                best = (u, px, py, pr, rid)
    if best is None:
        return None
    u_b, px_b, py_b, pr_b, rid_b = best
    return u_b, px_b, py_b, pr_b, best_d2, rid_b


def _bev_reloc_snap_validated(
    dw: DrivingWorld,
    world_bgr: np.ndarray,
    mx: float,
    my: float,
    h: int,
    margin: float,
    ramp_kinematics: bool,
) -> dict[str, float | bool] | None:
    """
    Snap a BEV click to the nearest valid drivable pose (main and/or ramp), preferring smaller
    screen-space error. ``psi`` is road tangent; ``get_view_from_pose`` must succeed.
    """
    candidates: list[
        tuple[float, float, float, float, bool, float, int]
    ] = []
    main = _snap_main_right_lane_nearest(dw, mx, my, h, margin, world_bgr)
    if main is not None:
        xc, y, psi_m, d2m = main
        candidates.append((d2m, xc, y, psi_m, False, 0.0, 0))
    if ramp_kinematics and dw._offramp_bezier_controls() is not None:
        ramp = _snap_ramp_right_lane_nearest(dw, mx, my, world_bgr)
        if ramp is not None:
            u_r, rx, ry, psi_r, d2r, rid_r = ramp
            candidates.append((d2r, rx, ry, psi_r, True, u_r, rid_r))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    for _d2, x, y, psi_road, on_r, u_r, ramp_id in candidates:
        if on_r:
            ev = dw.offramp_bezier_evolution(float(u_r), int(ramp_id))
            if ev is None:
                continue
            _, (fx, fy), _ = ev
            psi_cam = float(np.arctan2(fy, fx))
        else:
            psi_cam = float(initial_heading_road_aligned(dw.cs, float(y)))
        if (
            get_view_from_pose(
                world_bgr,
                float(x),
                float(y),
                psi_cam,
                _ego_lateral_offset_px_at_y(
                    dw,
                    float(y),
                    world_bgr,
                    float(x),
                    float(psi_cam),
                    on_ramp=bool(on_r),
                ),
            )
            is None
        ):
            continue
        return {
            "x": float(x),
            "y": float(y),
            "psi": psi_cam,
            "on_ramp": bool(on_r),
            "u_ramp": float(u_r),
            "ramp_id": int(ramp_id),
        }
    return None


def _advance_main_centerline_arc_px(
    dw: DrivingWorld,
    y: float,
    step_dist_px: float,
    *,
    margin: float,
    h: int,
) -> tuple[float, float]:
    """
    Step ``step_dist_px`` along the main centerline toward decreasing ``y`` (forward / up the map).

    Uses arc length in the graph ``x = cs(y)``: ``ds = sqrt(1 + (dx/dy)^2) |dy|``.
    """
    y_lo, y_hi = _sim_arc_y_bounds(h, margin)
    y_c = float(np.clip(y, y_lo, y_hi))
    dxdy = float(dw.cs(y_c, nu=1))
    dyd_s = -1.0 / float(np.sqrt(1.0 + dxdy * dxdy))
    y_new = y_c + float(step_dist_px) * dyd_s
    y_new = float(np.clip(y_new, y_lo, y_hi))
    return float(dw.get_road_center(y_new)), y_new


def find_start_pose_bottom(
    world_bgr: np.ndarray,
    dw: DrivingWorld,
) -> tuple[float, float, float]:
    """
    Choose a start pose near the bottom row with a valid first ``get_view_from_pose``.

    Tries ``y = h - 1, h - 2, ...`` up to ``SIM_START_MAX_INSET_PX``, then falls back to
    ``h - DATASET_MAP_MARGIN``.

    Returns:
        ``(x0, y0, psi)`` in pixels and radians.
    """
    h, _w = world_bgr.shape[:2]
    for inset in range(0, cfg.SIM_START_MAX_INSET_PX + 1):
        y0 = float(h - 1 - inset)
        if y0 < 0:
            break
        x0 = float(dw.get_road_center(y0))
        psi = initial_heading_road_aligned(dw.cs, y0)
        if (
            get_view_from_pose(
                world_bgr,
                x0,
                y0,
                psi,
                _ego_lateral_offset_px_at_y(dw, y0, world_bgr, float(x0), float(psi)),
            )
            is not None
        ):
            return x0, y0, psi
    y0 = float(h - cfg.DATASET_MAP_MARGIN)
    x0 = float(dw.get_road_center(y0))
    psi = initial_heading_road_aligned(dw.cs, y0)
    return x0, y0, psi


def run_simulation() -> tuple[
    np.ndarray,
    list[SimPathPoint],
    str,
    float,
    str | None,
]:
    """
    Load the newest checkpoint and roll out: crop -> ``DrivingNet`` channel 0 -> ``psi`` via
    ``SIM_YAW_RATE_GAIN``. With ``SIM_PROJECT_REF_ONTO_MAIN_ROAD``, ``(x,y)`` advances by arc length along
    the main spline (``ds = speed * dt`` in px); otherwise ``(x,y)`` use ``cos/sin(psi)``.

    Stops early when ``SIM_STOP_WHEN_REACHES_MAP_TOP`` and ``y <= DATASET_MAP_MARGIN``, or at
    ``SIM_MAX_STEPS``.

    When ``SIM_FP_VIDEO_ENABLE`` is true, writes each driver crop (plus a final pose frame when
    available) to ``<simulation_output_dir>/<SIM_FP_VIDEO_FILENAME>`` at ``SIM_VIDEO_FPS``.

    Returns:
        ``world_bgr``, list of ``SimPathPoint`` (centerline ``x,y``, integrated ``psi``, ramp flag,
        ``u_ramp``), checkpoint path,
        ``px_per_m``, and absolute path to the first-person MP4 if recorded, else ``None``.
    """
    ckpt = latest_checkpoint_path()
    if ckpt is None:
        raise SystemExit(
            "No checkpoint found. Train first (runs/*/driving_net.pt or data/driving_net.pt)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dw = DrivingWorld()
    world = dw.image
    h, w = world.shape[:2]
    px_per_m = dw.px_per_m

    x0, y0, psi = find_start_pose_bottom(world, dw)
    margin = float(cfg.DATASET_MAP_MARGIN)

    model = DrivingNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    if device.type == "cuda" and bool(
        getattr(cfg, "SIM_CUDNN_BENCHMARK", True)
    ):
        torch.backends.cudnn.benchmark = True
    if bool(getattr(cfg, "SIM_TORCH_COMPILE", False)) and hasattr(
        torch, "compile"
    ):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("torch.compile enabled for DrivingNet (warmup may take a few steps).")
        except Exception as exc:
            print(
                f"Warning: torch.compile skipped ({type(exc).__name__}: {exc}).",
                file=sys.stderr,
            )
    n_params = sum(int(p.numel()) for p in model.parameters())
    cls = type(model)
    print(
        f"Model: {cls.__module__}.{cls.__qualname__} ({n_params:,} parameters), device={device}"
    )
    print(f"Weights: {os.path.abspath(ckpt)}")

    out_dir = simulation_output_dir(ckpt)
    fp_video_abs: str | None = None
    video_writer: cv2.VideoWriter | None = None
    if cfg.SIM_FP_VIDEO_ENABLE:
        os.makedirs(out_dir, exist_ok=True)
        fp_video_abs = os.path.join(out_dir, cfg.SIM_FP_VIDEO_FILENAME)

    path: list[SimPathPoint] = [_sim_path_point(x0, y0, psi)]
    x, y = x0, y0
    base_step_dist_px = cfg.SIM_SPEED_M_S * cfg.SIM_DT * px_per_m

    off_ok = dw.offramp_num() > 0
    ramp_kinematics = (
        cfg.OFFRAMP_ENABLE
        and off_ok
        and cfg.SIM_PROJECT_REF_ONTO_MAIN_ROAD
    )
    off_branch_ys = list(dw.offramp_branch_y_pxs())
    on_ramp = False
    u_ramp = 0.0
    ramp_id = 0
    if ramp_kinematics and off_branch_ys:
        start_rid = -1
        for rid, yb in enumerate(off_branch_ys):
            if float(y0) <= yb + 1e-9:
                start_rid = rid
        if cfg.SIM_TAKE_OFFRAMP_UPPER_HALF_NAV:
            want_ramp_start = float(y0) <= 0.5 * float(h)
        else:
            want_ramp_start = bool(cfg.SIM_TAKE_OFFRAMP)
        if start_rid >= 0 and want_ramp_start:
            on_ramp = True
            u_ramp = 0.0
            ramp_id = start_rid
            ev_start = dw.offramp_bezier_evolution(0.0, ramp_id)
            if ev_start is not None:
                B0, _, _ = ev_start
                x0 = float(B0[0])
                y0 = float(B0[1])
                x, y = x0, y0
                path[0] = _sim_path_point(
                    x0, y0, psi, on_ramp=True, u_ramp=0.0, ramp_id=ramp_id
                )

    rt_window = False
    rt_driver_window = False
    rt_gui_disabled = False
    rt_windows_placed = False
    rt_sim_paused = False
    rt_ui: dict = {
        "toggle": False,
        "bev_rect": (-1, -1, -2, -2),
        "drv_rect": (-1, -1, -2, -2),
        "sim_step": 0,
        "speed_scale": _rt_speed_scale_from_trackbar_pos(
            int(cfg.SIM_REALTIME_SPEED_TRACKBAR_DEFAULT)
        ),
        "speed_track_pos": int(cfg.SIM_REALTIME_SPEED_TRACKBAR_DEFAULT),
        "speed_drag": False,
        "speed_slider_rect": (-1, -1, -2, -2),
        "rt_sim_paused": False,
        "relocate_ctx": None,
        "ego_hit_xy": (-1, -1),
        "ego_grab_radius_px": _RT_EGO_GRAB_RADIUS_PX,
        "ego_drag": False,
        "reloc_preview_pose": None,
        "reloc_pending": None,
    }
    user_quit_rt = False
    interrupt_rt = [False]
    take_t = torch.zeros((1, 1), device=device, dtype=torch.float32)

    def _sim_sigint(_signum: int, _frame: object) -> None:
        interrupt_rt[0] = True
        print(
            "\nCtrl+C: stopping simulation…",
            file=sys.stderr,
        )

    try:
        _prev_sigint = signal.signal(signal.SIGINT, _sim_sigint)
    except ValueError:
        _prev_sigint = None

    try:
        for sim_i in range(cfg.SIM_MAX_STEPS):
            if interrupt_rt[0]:
                user_quit_rt = True
                break
            rt_ui["sim_step"] = sim_i
            step_dist_px = base_step_dist_px * float(rt_ui.get("speed_scale", 1.0))
            y_for_model = float(y)
            if cfg.SIM_TAKE_OFFRAMP_UPPER_HALF_NAV:
                nav_take = 1.0 if y_for_model <= 0.5 * float(h) else 0.0
            else:
                nav_take = 1.0 if cfg.SIM_TAKE_OFFRAMP else 0.0
            take_t[0, 0] = nav_take
            merge_main_to_ramp = (
                (cfg.SIM_TAKE_OFFRAMP_UPPER_HALF_NAV and nav_take >= 0.5)
                or (
                    not cfg.SIM_TAKE_OFFRAMP_UPPER_HALF_NAV
                    and cfg.SIM_TAKE_OFFRAMP
                )
            )
            if on_ramp:
                ev_pose = dw.offramp_bezier_evolution(u_ramp, ramp_id)
                if ev_pose is None:
                    break
                _, (fx, fy), _ = ev_pose
                psi_cam = float(np.arctan2(fy, fx))
            else:
                psi_cam = initial_heading_road_aligned(dw.cs, y)
            view = get_view_from_pose(
                world,
                x,
                y,
                psi_cam,
                _ego_lateral_offset_px_at_y(
                    dw,
                    float(y),
                    world,
                    float(x),
                    float(psi_cam),
                    on_ramp=on_ramp,
                ),
            )
            if view is None:
                break
    
            if fp_video_abs is not None:
                vframe = _fp_video_frame_bgr(view)
                vh, vw = vframe.shape[:2]
                if video_writer is None:
                    video_writer = cv2.VideoWriter(
                        fp_video_abs,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(cfg.SIM_VIDEO_FPS),
                        (vw, vh),
                    )
                    if not video_writer.isOpened():
                        print(
                            f"Warning: could not open video writer for {fp_video_abs!r}; "
                            "skipping first-person MP4."
                        )
                        fp_video_abs = None
                        video_writer = None
                if video_writer is not None:
                    video_writer.write(vframe)
    
            with torch.inference_mode():
                inp = preprocess_bgr_for_model(view, device)
                steering = float(model(inp, take_t)[0, 0].item())
    
            # Gain matches kappa_max * speed * px_per_m (see config._compute_sim_yaw_rate_gain).
            psi += steering * cfg.SIM_YAW_RATE_GAIN * cfg.SIM_DT
            if cfg.SIM_PROJECT_REF_ONTO_MAIN_ROAD:
                if ramp_kinematics and on_ramp:
                    step_r = dw.offramp_step_arc_px(
                        u_ramp, step_dist_px, ramp_id
                    )
                    if step_r is None:
                        break
                    u_ramp, x, y = step_r
                elif ramp_kinematics and not on_ramp:
                    y_before = float(y)
                    x, y = _advance_main_centerline_arc_px(
                        dw, y, step_dist_px, margin=margin, h=h
                    )
                    for rid, yb in enumerate(off_branch_ys):
                        if y_before > yb + 1e-9 and float(y) <= yb + 1e-9:
                            if not merge_main_to_ramp:
                                continue
                            on_ramp = True
                            u_ramp = 0.0
                            ramp_id = rid
                            ev_br = dw.offramp_bezier_evolution(0.0, ramp_id)
                            if ev_br is None:
                                break
                            Bb, _, _ = ev_br
                            x, y = float(Bb[0]), float(Bb[1])
                            break
                else:
                    x, y = _advance_main_centerline_arc_px(
                        dw, y, step_dist_px, margin=margin, h=h
                    )
            else:
                x += step_dist_px * np.cos(psi)
                y += step_dist_px * np.sin(psi)
            wb = float(cfg.SIM_BLEND_PSI_TO_MAIN_ROAD)
            if wb > 0.0:
                if on_ramp:
                    evb = dw.offramp_bezier_evolution(u_ramp, ramp_id)
                    if evb is not None:
                        _, (bx, by), _ = evb
                        psi = _blend_psi_toward(
                            psi, float(np.arctan2(by, bx)), wb
                        )
                else:
                    psi = _blend_psi_toward(
                        psi, initial_heading_road_aligned(dw.cs, y), wb
                    )
            path.append(
                _sim_path_point(
                    float(x),
                    float(y),
                    float(psi),
                    on_ramp=on_ramp,
                    u_ramp=u_ramp,
                    ramp_id=ramp_id,
                )
            )
    
            if on_ramp:
                ev_draw = dw.offramp_bezier_evolution(u_ramp, ramp_id)
                if ev_draw is not None:
                    _, (fxd, fyd), _ = ev_draw
                    psi_draw = float(np.arctan2(fyd, fxd))
                else:
                    psi_draw = psi_cam
            else:
                psi_draw = initial_heading_road_aligned(dw.cs, y)

            rt_ui["relocate_ctx"] = {
                "dw": dw,
                "world": world,
                "h": h,
                "margin": margin,
                "ramp_kinematics": ramp_kinematics,
            }
            lat_hit = _ego_lateral_offset_px_at_y(
                dw,
                float(y),
                world,
                float(x),
                float(psi_draw),
                on_ramp=on_ramp,
            )
            qx_hit, qy_hit = _right_lane_overlay_xy(
                float(x), float(y), float(psi_draw), lat_hit
            )
            rt_ui["ego_hit_xy"] = (int(round(qx_hit)), int(round(qy_hit)))
            rt_ui["ego_grab_radius_px"] = _bev_ego_grab_radius_px(float(dw.px_per_m))

            live_bev = cfg.SIM_REALTIME_BEV and not rt_gui_disabled
            live_drv = cfg.SIM_REALTIME_DRIVER_VIEW and not rt_gui_disabled
            if live_bev or live_drv:
                try:
                    view_live = get_view_from_pose(
                        world,
                        x,
                        y,
                        psi_draw,
                        lat_hit,
                    )
                    if view_live is None:
                        break
                    vframe = _fp_video_frame_bgr(view_live)
                    fr: np.ndarray | None = None
                    drv: np.ndarray | None = None
                    if live_drv:
                        if not rt_driver_window:
                            cv2.namedWindow(
                                cfg.SIM_REALTIME_DRIVER_WINDOW,
                                cv2.WINDOW_NORMAL,
                            )
                            rt_driver_window = True
                            cv2.setMouseCallback(
                                cfg.SIM_REALTIME_DRIVER_WINDOW,
                                _rt_mouse_drv,
                                rt_ui,
                            )
                        drv = vframe.copy()
                        _rt_annotate_driver_view(drv, steering)
                    if live_bev:
                        if not rt_window:
                            cv2.namedWindow(
                                cfg.SIM_REALTIME_BEV_WINDOW, cv2.WINDOW_NORMAL
                            )
                            rt_window = True
                            cv2.setMouseCallback(
                                cfg.SIM_REALTIME_BEV_WINDOW,
                                _rt_mouse_bev,
                                rt_ui,
                            )
                        bev_hint = (
                            "PAUSED: drag ego car to move, release on road"
                            if rt_sim_paused
                            else None
                        )
                        fr = _bev_realtime_frame(
                            world,
                            dw,
                            path,
                            x,
                            y,
                            psi_draw,
                            lat_hit,
                            extra_hint=bev_hint,
                            nav_exit_active=_bev_nav_exit_active(float(y), h),
                            nav_exit_text=str(cfg.SIM_NAV_EXIT_INSTRUCTION_TEXT),
                        )
                        _rt_draw_speed_slider(fr, rt_ui)
                    if not rt_windows_placed:
                        _tile_realtime_sim_windows(fr, drv)
                        rt_windows_placed = True
    
                    pause_ms = int(cfg.SIM_REALTIME_STEP_PAUSE_MS)
                    while True:
                        rp = rt_ui.pop("reloc_pending", None)
                        if rp is not None:
                            x = float(rp["x"])
                            y = float(rp["y"])
                            psi = float(rp["psi"])
                            on_ramp = bool(rp["on_ramp"])
                            u_ramp = float(rp["u_ramp"])
                            ramp_id = int(rp.get("ramp_id", 0))
                            psi_draw = float(rp["psi"])
                            path.clear()
                            path.append(
                                _sim_path_point(
                                    x,
                                    y,
                                    psi,
                                    on_ramp=on_ramp,
                                    u_ramp=u_ramp,
                                    ramp_id=ramp_id,
                                )
                            )
                            lat_hit = _ego_lateral_offset_px_at_y(
                                dw,
                                float(y),
                                world,
                                float(x),
                                float(psi_draw),
                                on_ramp=on_ramp,
                            )
                            qx_hit, qy_hit = _right_lane_overlay_xy(
                                float(x), float(y), float(psi_draw), lat_hit
                            )
                            rt_ui["ego_hit_xy"] = (
                                int(round(qx_hit)),
                                int(round(qy_hit)),
                            )
                            rt_ui["ego_grab_radius_px"] = _bev_ego_grab_radius_px(
                                float(dw.px_per_m)
                            )
                            if fr is not None:
                                bh = (
                                    "PAUSED: drag ego car to move, release on road"
                                    if rt_sim_paused
                                    else None
                                )
                                fr = _bev_realtime_frame(
                                    world,
                                    dw,
                                    path,
                                    x,
                                    y,
                                    psi_draw,
                                    lat_hit,
                                    extra_hint=bh,
                                    nav_exit_active=_bev_nav_exit_active(float(y), h),
                                    nav_exit_text=str(
                                        cfg.SIM_NAV_EXIT_INSTRUCTION_TEXT
                                    ),
                                )
                                _rt_draw_speed_slider(fr, rt_ui)
                            view_live = get_view_from_pose(
                                world,
                                x,
                                y,
                                psi_draw,
                                lat_hit,
                            )
                            if view_live is not None and drv is not None:
                                vframe = _fp_video_frame_bgr(view_live)
                                drv = vframe.copy()
                                _rt_annotate_driver_view(drv, steering)
                        rt_ui["toggle"] = False
                        paused = rt_sim_paused
                        rt_ui["rt_sim_paused"] = paused
                        disp_x, disp_y = float(x), float(y)
                        disp_psi_draw = float(psi_draw)
                        disp_lat = lat_hit
                        prv = rt_ui.get("reloc_preview_pose")
                        if paused and isinstance(prv, dict) and rt_ui.get("ego_drag"):
                            disp_x = float(prv["x"])
                            disp_y = float(prv["y"])
                            disp_psi_draw = float(prv["psi"])
                            disp_lat = _ego_lateral_offset_px_at_y(
                                dw,
                                disp_y,
                                world,
                                disp_x,
                                disp_psi_draw,
                                on_ramp=bool(prv.get("on_ramp", False)),
                            )
                        if fr is not None:
                            bh = (
                                "PAUSED: drag ego car to move, release on road"
                                if paused
                                else None
                            )
                            fr = _bev_realtime_frame(
                                world,
                                dw,
                                path,
                                disp_x,
                                disp_y,
                                disp_psi_draw,
                                disp_lat,
                                extra_hint=bh,
                                nav_exit_active=_bev_nav_exit_active(
                                    float(disp_y), h
                                ),
                                nav_exit_text=str(
                                    cfg.SIM_NAV_EXIT_INSTRUCTION_TEXT
                                ),
                            )
                            _rt_draw_speed_slider(fr, rt_ui)
                        fr_disp = fr.copy() if fr is not None else None
                        drv_disp = drv.copy() if drv is not None else None
                        if fr_disp is not None:
                            if paused:
                                _rt_draw_paused_banner(fr_disp)
                            rt_ui["bev_rect"] = _rt_draw_pause_button(
                                fr_disp, paused=paused
                            )
                        if drv_disp is not None:
                            if paused:
                                _rt_draw_paused_banner(drv_disp)
                            rt_ui["drv_rect"] = _rt_draw_pause_button(
                                drv_disp, paused=paused
                            )
                        if fr_disp is not None:
                            cv2.imshow(cfg.SIM_REALTIME_BEV_WINDOW, fr_disp)
                        if drv_disp is not None:
                            cv2.imshow(cfg.SIM_REALTIME_DRIVER_WINDOW, drv_disp)

                        wk = (
                            30
                            if paused
                            else max(0, int(cfg.SIM_REALTIME_BEV_WAIT_MS))
                        )
                        key_raw = _rt_wait_key_interruptible(wk, interrupt_rt)
                        key_ch = None if key_raw < 0 else (key_raw & 0xFF)
                        clicked = bool(rt_ui["toggle"])
                        rt_ui["toggle"] = False
                        if interrupt_rt[0]:
                            user_quit_rt = True
                            break
                        if key_ch == ord("q"):
                            user_quit_rt = True
                            break
                        if (
                            key_ch is not None
                            and key_ch in (ord(" "), ord("p"))
                        ) or clicked:
                            rt_sim_paused = not rt_sim_paused
                            if not rt_sim_paused:
                                rt_ui["ego_drag"] = False
                                rt_ui.pop("reloc_preview_pose", None)
                            if rt_sim_paused:
                                continue
                            break
                        if not rt_sim_paused:
                            break
                        continue
                    if user_quit_rt:
                        break
                    if pause_ms > 0 and not interrupt_rt[0]:
                        rem = pause_ms / 1000.0
                        while rem > 0 and not interrupt_rt[0]:
                            chunk = min(0.02, rem)
                            time.sleep(chunk)
                            rem -= chunk
                        if interrupt_rt[0]:
                            user_quit_rt = True
                    if user_quit_rt:
                        break
                except cv2.error:
                    if not rt_gui_disabled:
                        print(
                            "Warning: live display failed (no GUI / OpenCV backend?). "
                            "Set SIM_REALTIME_BEV / SIM_REALTIME_DRIVER_VIEW False in config.py. "
                            "Disabling for this run."
                        )
                    rt_gui_disabled = True
                    if rt_window:
                        try:
                            cv2.destroyWindow(cfg.SIM_REALTIME_BEV_WINDOW)
                        except cv2.error:
                            pass
                        rt_window = False
                    if rt_driver_window:
                        try:
                            cv2.destroyWindow(cfg.SIM_REALTIME_DRIVER_WINDOW)
                        except cv2.error:
                            pass
                        rt_driver_window = False
    
            if on_ramp and u_ramp >= 1.0 - 1e-8:
                break
            if (not on_ramp) and cfg.SIM_STOP_WHEN_REACHES_MAP_TOP and float(y) <= margin:
                break
            if x < 0 or x >= w or y < 0 or float(y) >= float(h):
                break
    finally:
        if _prev_sigint is not None:
            signal.signal(signal.SIGINT, _prev_sigint)

    if rt_window:
        try:
            cv2.destroyWindow(cfg.SIM_REALTIME_BEV_WINDOW)
        except cv2.error:
            pass
    if rt_driver_window:
        try:
            cv2.destroyWindow(cfg.SIM_REALTIME_DRIVER_WINDOW)
        except cv2.error:
            pass

    if video_writer is not None:
        if on_ramp:
            ev_fin = dw.offramp_bezier_evolution(u_ramp, ramp_id)
            if ev_fin is not None:
                _, (ex, ey), _ = ev_fin
                psi_end = float(np.arctan2(ey, ex))
            else:
                psi_end = initial_heading_road_aligned(dw.cs, y)
        else:
            psi_end = initial_heading_road_aligned(dw.cs, y)
        view_end = get_view_from_pose(
            world,
            x,
            y,
            psi_end,
            _ego_lateral_offset_px_at_y(
                dw,
                float(y),
                world,
                float(x),
                float(psi_end),
                on_ramp=on_ramp,
            ),
        )
        if view_end is not None:
            video_writer.write(_fp_video_frame_bgr(view_end))
        video_writer.release()
        fp_video_abs = os.path.abspath(fp_video_abs) if fp_video_abs else None
    elif cfg.SIM_FP_VIDEO_ENABLE and fp_video_abs is not None:
        # Enabled but no frames (e.g. immediate warp failure).
        fp_video_abs = None

    n_steps = len(path) - 1
    print(f"Simulation steps taken: {n_steps}")

    return world, path, ckpt, px_per_m, fp_video_abs


def main() -> None:
    """Run simulation, save BEV overlay, CSV, first-person MP4 (if enabled), print stats, show figure."""
    n_train, n_test = count_train_test_examples()
    print(f"Dataset: {n_train} train examples, {n_test} test examples")
    world_bgr, path, ckpt, px_per_m, fp_video = run_simulation()
    path_arr = np.array([[p[0], p[1], p[2]] for p in path], dtype=np.float64)

    vis = world_bgr.copy()
    dw_overlay = DrivingWorld()
    train_y_rows, _ = dataset_train_test_y(
        int(cfg.NUM_TRAIN_FRAMES),
        int(cfg.NUM_TEST_FRAMES),
        dw_overlay.size,
        cfg.DATASET_MAP_MARGIN,
        mix_train_test_geography=cfg.DATASET_MIX_TRAIN_TEST_GEOGRAPHY,
        seed=cfg.DATASET_SEED,
        uniform_along_road=cfg.DATASET_SAMPLE_UNIFORM_ALONG_ROAD,
        road_cs=dw_overlay.cs,
    )
    _draw_train_sampling_bars_on_bev(vis, dw_overlay, train_y_rows)
    L_ramp_px = float(dw_overlay.offramp_total_arc_length_px())
    n_offramp_train_vis, _ = offramp_clean_counts_matching_main_spacing(
        L_ramp_px,
        int(cfg.NUM_TRAIN_FRAMES),
        int(cfg.NUM_TEST_FRAMES),
        int(dw_overlay.size),
        float(cfg.DATASET_MAP_MARGIN),
        mix_train_test_geography=cfg.DATASET_MIX_TRAIN_TEST_GEOGRAPHY,
        cs=dw_overlay.cs,
        cap_train=int(cfg.DATASET_OFFRAMP_TRAIN_FRAMES),
        cap_test=int(cfg.DATASET_OFFRAMP_TEST_FRAMES),
        match_spacing=cfg.DATASET_OFFRAMP_MATCH_MAIN_SPACING,
    )
    if (
        cfg.OFFRAMP_ENABLE
        and cfg.DATASET_OFFRAMP_LABELS_ENABLE
        and n_offramp_train_vis > 0
    ):
        _draw_offramp_train_sampling_bars_on_bev(
            vis,
            dw_overlay,
            int(n_offramp_train_vis),
            float(cfg.offramp_camera_lateral_offset_px(px_per_m)),
            num_main_train=int(cfg.NUM_TRAIN_FRAMES),
        )
    poly_xy: list[list[float]] = []
    for i in range(len(path) - 1):
        x_a, y_a, _, on_a, u_a, rid_a = path[i]
        x_b, y_b, _, on_b, u_b, rid_b = path[i + 1]
        ps_a = _path_point_psi_draw(dw_overlay, x_a, y_a, on_a, u_a, rid_a)
        ps_b = _path_point_psi_draw(dw_overlay, x_b, y_b, on_b, u_b, rid_b)
        lat_m = 0.5 * (
            _ego_lateral_offset_px_at_y(
                dw_overlay,
                float(y_a),
                vis,
                float(x_a),
                float(ps_a),
                on_ramp=bool(on_a),
            )
            + _ego_lateral_offset_px_at_y(
                dw_overlay,
                float(y_b),
                vis,
                float(x_b),
                float(ps_b),
                on_ramp=bool(on_b),
            )
        )
        seg = _right_lane_polyline_xy_chord(
            x_a, y_a, x_b, y_b, lat_m, n=24
        )
        if i == 0:
            poly_xy.extend(seg.tolist())
        else:
            poly_xy.extend(seg[1:].tolist())
    if len(poly_xy) >= 2:
        arr = np.asarray(poly_xy, dtype=np.float32)
        arr_i = np.round(arr).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [arr_i], False, (0, 0, 255), 2, cv2.LINE_AA)
    if len(path) > 0:
        x0, y0, _, on0, u0, rid0 = path[0]
        xe, ye, _, one, ue, ride = path[-1]
        ps0 = _path_point_psi_draw(dw_overlay, x0, y0, on0, u0, rid0)
        pse = _path_point_psi_draw(dw_overlay, xe, ye, one, ue, ride)
        sq = _right_lane_overlay_xy(
            x0,
            y0,
            ps0,
            _ego_lateral_offset_px_at_y(
                dw_overlay,
                float(y0),
                vis,
                float(x0),
                float(ps0),
                on_ramp=bool(on0),
            ),
        )
        eq = _right_lane_overlay_xy(
            xe,
            ye,
            pse,
            _ego_lateral_offset_px_at_y(
                dw_overlay,
                float(ye),
                vis,
                float(xe),
                float(pse),
                on_ramp=bool(one),
            ),
        )
        _draw_bev_ego_car_icon(
            vis, float(sq[0]), float(sq[1]), ps0, float(dw_overlay.px_per_m)
        )
        _draw_bev_ego_car_icon(
            vis,
            float(eq[0]),
            float(eq[1]),
            pse,
            float(dw_overlay.px_per_m),
            body_bgr=(55, 55, 220),
            axis_bgr=(180, 220, 255),
        )

    out_dir = simulation_output_dir(ckpt)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "sim_path.png")
    out_csv = os.path.join(out_dir, "ego_path.csv")
    n_tr_stat, n_pert, frac_pert = train_perturb_stats_from_labels()
    n_epochs = training_epochs_from_checkpoint(ckpt)
    if n_epochs is not None:
        epochs_line = f"Training epochs: {n_epochs}"
    else:
        epochs_line = f"Training epochs: {cfg.EPOCHS} (config; no log by weights)"
    _put_outlined_lines_bgr(
        vis,
        [
            f"Training examples: {n_tr_stat}",
            f"Perturbed: {frac_pert:.1%} ({n_pert}/{n_tr_stat})",
            epochs_line,
        ],
        cfg.ANNOT_STEERING_POS,
    )
    print(epochs_line)
    cv2.imwrite(out_png, vis)
    np.savetxt(
        out_csv,
        path_arr,
        delimiter=",",
        header="x_center_px,y_center_px,psi_integrated_rad",
        comments="",
    )
    if len(path) > 1:
        xy = path_arr[:, :2]
        dpx = float(np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1)).sum())
        print(f"Poses: {len(path)}, path length ~ {dpx / px_per_m:.1f} m")
    else:
        print(f"Poses: {len(path)}")
    print(f"Saved overlay: {out_png!r}")
    print(f"Saved trajectory: {out_csv!r}")
    if fp_video:
        print(f"Saved first-person video: {fp_video!r} ({cfg.SIM_VIDEO_FPS} FPS)")

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(
        f"Ego path (open-loop, {cfg.SIM_SPEED_M_S} m/s, Δt={cfg.SIM_DT}s, "
        f"{len(path)} steps)"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=False)
    try:
        fig = plt.gcf()
        while plt.fignum_exists(fig.number):
            plt.pause(0.2)
    except KeyboardInterrupt:
        print("\nInterrupted (matplotlib summary).", file=sys.stderr)
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
