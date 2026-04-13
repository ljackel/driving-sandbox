"""
Open-loop behavioral-cloning roll-out on the bird's-eye map.

**State:** ``(x, y, ψ)`` is a centerline reference point and heading; each step warps the camera to
the **right lane** using ``SIM_EGO_LATERAL_OFFSET_M`` (same meters as ``generate_dataset``), matching
training geometry.

**Control:** The network predicts steering from the crop; ``train.py`` supervises **output channel 0**.
Heading updates as ``psi += steering * SIM_YAW_RATE_GAIN * SIM_DT`` where ``SIM_YAW_RATE_GAIN`` is
set in ``config`` (from ``kappa_max * SIM_SPEED_M_S * px_per_m`` on the train/test ``y`` grid) so
units line up with scaled-curvature labels. There is no separate tracking controller—small prediction
errors compound (classic BC / "open loop" in the ML sense).

**Visualization:** The red BEV overlay traces the right-lane camera path. With ``SIM_FP_VIDEO_ENABLE``,
each driver crop is written to ``sim_first_person.mp4`` (same preprocessing as the model when
``PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY`` is true: bottom half of the warp, resized to ``CAMERA_IMAGE_SIZE``).

**Input crop:** ``PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY`` matches ``train.py`` / ``evaluate_test.py`` (bottom
half of the warp, then resize to ``CAMERA_IMAGE_SIZE``).
"""
from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import config as cfg
from perspective_camera import perspective_camera_view
from reproducibility import set_global_seed

set_global_seed(cfg.TRAIN_SEED)

from driving_model import DrivingNet
from generate_world import DrivingWorld


def _project_root() -> str:
    """Absolute path to the project directory containing ``simulate.py``."""
    return os.path.dirname(os.path.abspath(__file__))


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
    Find the newest ``driving_net.pt`` by modification time.

    Scans ``runs/*/<CHECKPOINT_FILENAME>`` then compares with ``data/<CHECKPOINT_FILENAME>``.
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
    if os.path.isfile(data_w):
        m = os.path.getmtime(data_w)
        if m > best_mtime:
            best_path = data_w
    return best_path


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


def _ego_lateral_offset_px(dw: DrivingWorld) -> float:
    """Camera/ego offset from centerline (px), same convention as training."""
    return float(cfg.SIM_EGO_LATERAL_OFFSET_M) * float(dw.px_per_m)


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
    Load the newest checkpoint and roll out: crop -> ``DrivingNet`` channel 0 -> yaw rate via
    ``SIM_YAW_RATE_GAIN`` -> integrate ``(x, y, psi)`` with step length ``SIM_SPEED_M_S * SIM_DT * px_per_m``.

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
    for _ in range(cfg.SIM_MAX_STEPS):
        view = get_view_from_pose(
            world,
            x,
            y,
            psi,
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
            steering = float(model(inp)[0, 0].item())

        # Gain matches kappa_max * speed * px_per_m (see config._compute_sim_yaw_rate_gain).
        psi += steering * cfg.SIM_YAW_RATE_GAIN * cfg.SIM_DT
        x += step_dist_px * np.cos(psi)
        y += step_dist_px * np.sin(psi)
        path.append((float(x), float(y), float(psi)))

        if x < 0 or x >= w or y < 0 or y >= h:
            break

    if video_writer is not None:
        view_end = get_view_from_pose(world, x, y, psi, lateral_px)
        if view_end is not None:
            video_writer.write(_fp_video_frame_bgr(view_end))
        video_writer.release()
        fp_video_abs = os.path.abspath(fp_video_abs) if fp_video_abs else None
    elif cfg.SIM_FP_VIDEO_ENABLE and fp_video_abs is not None:
        # Enabled but no frames (e.g. immediate warp failure).
        fp_video_abs = None

    return world, path, ckpt, px_per_m, fp_video_abs


def main() -> None:
    """Run simulation, save BEV overlay, CSV, first-person MP4 (if enabled), print stats, show figure."""
    world_bgr, path, ckpt, px_per_m, fp_video = run_simulation()
    path_arr = np.array(path, dtype=np.float64)

    lateral_px = float(cfg.SIM_EGO_LATERAL_OFFSET_M) * px_per_m
    vis = world_bgr.copy()
    for i in range(len(path) - 1):
        x0, y0, p0 = path[i]
        x1, y1, p1 = path[i + 1]
        q0 = _right_lane_overlay_xy(x0, y0, p0, lateral_px)
        q1 = _right_lane_overlay_xy(x1, y1, p1, lateral_px)
        cv2.line(
            vis,
            (int(round(q0[0])), int(round(q0[1]))),
            (int(round(q1[0])), int(round(q1[1]))),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if len(path) > 0:
        x0, y0, p0 = path[0]
        xe, ye, pe = path[-1]
        sq = _right_lane_overlay_xy(x0, y0, p0, lateral_px)
        eq = _right_lane_overlay_xy(xe, ye, pe, lateral_px)
        cv2.circle(vis, (int(round(sq[0])), int(round(sq[1]))), 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(round(eq[0])), int(round(eq[1]))), 6, (255, 0, 0), -1, cv2.LINE_AA)

    out_dir = simulation_output_dir(ckpt)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "sim_path.png")
    out_csv = os.path.join(out_dir, "ego_path.csv")
    cv2.imwrite(out_png, vis)
    np.savetxt(
        out_csv,
        path_arr,
        delimiter=",",
        header="x_center_px,y_center_px,psi_rad",
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
