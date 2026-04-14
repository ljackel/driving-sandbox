"""
Open-loop behavioral-cloning roll-out on the bird's-eye map.

**State:** ``(x, y)`` is on the main centerline when ``SIM_PROJECT_REF_ONTO_MAIN_ROAD`` is true: each step
advances **arc length** along the spline (decreasing ``y``), so the logged path cannot chord off the road.
``psi`` still integrates from the network (``psi += steering * SIM_YAW_RATE_GAIN * SIM_DT``) for the
BC state / CSV. The camera uses the spline tangent at ``y`` and ``SIM_EGO_LATERAL_OFFSET_M`` (same as
``generate_dataset``).

**Control:** The network predicts steering from the crop; ``train.py`` supervises **output channel 0``.
With projection on, steering does **not** translate the centerline reference with ``cos/sin(psi)``—that
mixture caused off-road jumps when combined with ``x = cs(y)``. Set ``SIM_PROJECT_REF_ONTO_MAIN_ROAD``
false for unconstrained ``(x,y)`` integration (can leave the lane).

**Visualization:** The red BEV overlay traces the right-lane camera path; **yellow bars** mark
clean **main-road** training sample locations (same ``y`` grid as ``generate_dataset``). When off-ramp
dataset rows are enabled, **orange bars** mark ramp samples (same spacing as ``train/offramp_*.jpg``:
arc length along the Bézier when ``DATASET_SAMPLE_UNIFORM_ALONG_ROAD``; counts follow
``DATASET_OFFRAMP_MATCH_MAIN_SPACING`` like ``generate_dataset``).
With ``SIM_FP_VIDEO_ENABLE``,
each driver crop is written to ``sim_first_person.mp4`` (same preprocessing as the model when
``PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY`` is true: bottom half of the warp, resized to ``CAMERA_IMAGE_SIZE``).

**Input crop:** ``PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY`` matches ``train.py`` / ``evaluate_test.py`` (bottom
half of the warp, then resize to ``CAMERA_IMAGE_SIZE``).

**Intent / off-ramp:** ``SIM_TAKE_OFFRAMP`` selects main vs ramp steering targets. With
``SIM_PROJECT_REF_ONTO_MAIN_ROAD`` and an enabled off-ramp, the reference switches from the **main** spline
to the **ramp** Bézier after the branch ``y``; motion on the ramp uses arc-length steps in ``u``.

**Camera / overlay heading:** the warp and BEV trail use the **road tangent** at ``(x, y)``—main spline or
ramp Bézier when on the ramp—not the integrated BC ``psi``. Otherwise ``psi`` drift places the right-lane
offset in the grass and the red path “jumps” off the road.
"""
from __future__ import annotations

import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import config as cfg
from data_loader import count_train_test_examples, train_perturb_stats_from_labels
from dataset_split import (
    dataset_train_test_y,
    offramp_clean_counts_matching_main_spacing,
)
from perspective_camera import perspective_camera_view
from reproducibility import set_global_seed

set_global_seed(cfg.TRAIN_SEED)

from driving_model import DrivingNet
from generate_world import DrivingWorld


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


def _ego_lateral_offset_px(dw: DrivingWorld) -> float:
    """Camera/ego offset from centerline (px), same convention as training."""
    return float(cfg.SIM_EGO_LATERAL_OFFSET_M) * float(dw.px_per_m)


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
    lateral_offset_px: float,
) -> None:
    """
    Draw yellow bars at clean **main-road** training sample locations on the BEV (right-lane ego).
    """
    bar_half = max(8.0, 0.35 * float(cfg.LANE_WIDTH_METERS * dw.px_per_m))
    yellow_bgr = (0, 255, 255)
    for yf in train_y:
        yf = float(yf)
        xc = float(dw.get_road_center(yf))
        psi = initial_heading_road_aligned(dw.cs, yf)
        qx, qy = _right_lane_overlay_xy(xc, yf, psi, lateral_offset_px)
        _draw_lateral_sampling_bar(vis, qx, qy, psi, bar_half, yellow_bgr)


def _draw_offramp_train_sampling_bars_on_bev(
    vis: np.ndarray,
    dw: DrivingWorld,
    n_ramp_train_frames: int,
    lateral_offset_px: float,
) -> None:
    """
    Draw orange bars where ``generate_dataset`` places clean off-ramp train crops (``u`` in ``[0.05, 0.95]``).
    """
    if n_ramp_train_frames <= 0 or dw._offramp_bezier_controls() is None:
        return
    bar_half = max(8.0, 0.35 * float(cfg.LANE_WIDTH_METERS * dw.px_per_m))
    orange_bgr = (0, 200, 255)
    n_r = int(n_ramp_train_frames)
    u_grid = (
        dw.offramp_u_samples_uniform_arc_length(n_r)
        if cfg.DATASET_SAMPLE_UNIFORM_ALONG_ROAD
        else np.linspace(0.05, 0.95, n_r, dtype=np.float64)
    )
    for u in u_grid:
        ev = dw.offramp_bezier_evolution(float(u))
        if ev is None:
            continue
        B, (fx, fy), _ = ev
        px, py = float(B[0]), float(B[1])
        psi = float(np.arctan2(fy, fx))
        qx, qy = _right_lane_overlay_xy(px, py, psi, lateral_offset_px)
        _draw_lateral_sampling_bar(vis, qx, qy, psi, bar_half, orange_bgr)


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
                _ego_lateral_offset_px(dw),
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
    list[tuple[float, float, float]],
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
        ``world_bgr``, list of **centerline** ``(x, y, psi)`` (psi in rad), checkpoint path,
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

    path: list[tuple[float, float, float]] = [(x0, y0, psi)]
    x, y = x0, y0
    step_dist_px = cfg.SIM_SPEED_M_S * cfg.SIM_DT * px_per_m

    lateral_px = _ego_lateral_offset_px(dw)
    off_ok = dw._offramp_bezier_controls() is not None
    follow_ramp_geom = (
        cfg.SIM_TAKE_OFFRAMP
        and cfg.OFFRAMP_ENABLE
        and off_ok
        and cfg.SIM_PROJECT_REF_ONTO_MAIN_ROAD
    )
    y_branch = float(np.clip(cfg.OFFRAMP_BRANCH_Y_FRAC, 0.0, 1.0)) * float(h)
    on_ramp = False
    u_ramp = 0.0
    if follow_ramp_geom and float(y0) <= y_branch + 1e-9:
        on_ramp = True
        u_ramp = 0.0
        ev_start = dw.offramp_bezier_evolution(0.0)
        if ev_start is not None:
            B0, _, _ = ev_start
            x0 = float(B0[0])
            y0 = float(B0[1])
            x, y = x0, y0
            path[0] = (x0, y0, psi)

    for _ in range(cfg.SIM_MAX_STEPS):
        if on_ramp:
            ev_pose = dw.offramp_bezier_evolution(u_ramp)
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
            lateral_px,
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

        with torch.no_grad():
            inp = preprocess_bgr_for_model(view, device)
            intent = 1.0 if cfg.SIM_TAKE_OFFRAMP else 0.0
            take_t = torch.tensor([[intent]], device=device, dtype=inp.dtype)
            steering = float(model(inp, take_t)[0, 0].item())

        # Gain matches kappa_max * speed * px_per_m (see config._compute_sim_yaw_rate_gain).
        psi += steering * cfg.SIM_YAW_RATE_GAIN * cfg.SIM_DT
        if cfg.SIM_PROJECT_REF_ONTO_MAIN_ROAD:
            if follow_ramp_geom and on_ramp:
                step_r = dw.offramp_step_arc_px(u_ramp, step_dist_px)
                if step_r is None:
                    break
                u_ramp, x, y = step_r
            elif follow_ramp_geom and not on_ramp:
                y_before = float(y)
                x, y = _advance_main_centerline_arc_px(
                    dw, y, step_dist_px, margin=margin, h=h
                )
                if y_before > y_branch + 1e-9 and float(y) <= y_branch + 1e-9:
                    on_ramp = True
                    u_ramp = 0.0
                    ev_br = dw.offramp_bezier_evolution(0.0)
                    if ev_br is None:
                        break
                    Bb, _, _ = ev_br
                    x, y = float(Bb[0]), float(Bb[1])
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
                evb = dw.offramp_bezier_evolution(u_ramp)
                if evb is not None:
                    _, (bx, by), _ = evb
                    psi = _blend_psi_toward(
                        psi, float(np.arctan2(by, bx)), wb
                    )
            else:
                psi = _blend_psi_toward(
                    psi, initial_heading_road_aligned(dw.cs, y), wb
                )
        path.append((float(x), float(y), float(psi)))

        if on_ramp and u_ramp >= 1.0 - 1e-8:
            break
        if (not on_ramp) and cfg.SIM_STOP_WHEN_REACHES_MAP_TOP and float(y) <= margin:
            break
        if x < 0 or x >= w or y < 0 or float(y) >= float(h):
            break

    if video_writer is not None:
        if on_ramp:
            ev_fin = dw.offramp_bezier_evolution(u_ramp)
            if ev_fin is not None:
                _, (ex, ey), _ = ev_fin
                psi_end = float(np.arctan2(ey, ex))
            else:
                psi_end = initial_heading_road_aligned(dw.cs, y)
        else:
            psi_end = initial_heading_road_aligned(dw.cs, y)
        view_end = get_view_from_pose(world, x, y, psi_end, lateral_px)
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
    path_arr = np.array(path, dtype=np.float64)

    lateral_px = float(cfg.SIM_EGO_LATERAL_OFFSET_M) * px_per_m
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
    _draw_train_sampling_bars_on_bev(vis, dw_overlay, train_y_rows, lateral_px)
    L_ramp_px = float(dw_overlay.offramp_arc_length_px())
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
            lateral_px,
        )
    poly_xy: list[list[float]] = []
    for i in range(len(path) - 1):
        x_a, y_a, _ = path[i]
        x_b, y_b, _ = path[i + 1]
        seg = _right_lane_polyline_xy_chord(
            x_a, y_a, x_b, y_b, lateral_px, n=24
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
        x0, y0, _ = path[0]
        xe, ye, _ = path[-1]
        if len(path) >= 2:
            xn, yn, _ = path[1]
            ps0 = float(np.arctan2(yn - y0, xn - x0))
        else:
            ps0 = initial_heading_road_aligned(dw_overlay.cs, y0)
        if len(path) >= 2:
            xp, yp, _ = path[-2]
            pse = float(np.arctan2(ye - yp, xe - xp))
        else:
            pse = initial_heading_road_aligned(dw_overlay.cs, ye)
        sq = _right_lane_overlay_xy(x0, y0, ps0, lateral_px)
        eq = _right_lane_overlay_xy(xe, ye, pse, lateral_px)
        cv2.circle(vis, (int(round(sq[0])), int(round(sq[1]))), 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(round(eq[0])), int(round(eq[1]))), 6, (255, 0, 0), -1, cv2.LINE_AA)

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
    plt.show()


if __name__ == "__main__":
    main()
