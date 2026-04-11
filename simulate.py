"""
Open-loop driving simulator on the bird's-eye map.
Ego starts at the bottom, centered on the road, moves at SIM_SPEED_M_S, steering from DrivingNet.
"""
from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import config as cfg
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
    h, w = world_bgr.shape[:2]
    fx, fy = float(np.cos(psi)), float(np.sin(psi))
    rx, ry = float(-np.sin(psi)), float(np.cos(psi))
    f = np.array([fx, fy], dtype=np.float32)
    r = np.array([rx, ry], dtype=np.float32)

    near_c = np.array([x, y], dtype=np.float32) + r * float(lateral_offset_px)
    far_c = near_c + f * float(cfg.PERSPECTIVE_FAR_OFFSET_PX)

    tl = far_c - r * float(cfg.PERSPECTIVE_FAR_HALF_WIDTH)
    tr = far_c + r * float(cfg.PERSPECTIVE_FAR_HALF_WIDTH)
    br = near_c + r * float(cfg.PERSPECTIVE_NEAR_HALF_WIDTH)
    bl = near_c - r * float(cfg.PERSPECTIVE_NEAR_HALF_WIDTH)
    src = np.float32([tl, tr, br, bl])

    mrg = float(cfg.PERSPECTIVE_SRC_MARGIN_PX)
    if (
        (src[:, 0] < -mrg).any()
        or (src[:, 0] >= w + mrg).any()
        or (src[:, 1] < -mrg).any()
        or (src[:, 1] >= h + mrg).any()
    ):
        return None

    s = float(cfg.CAMERA_IMAGE_SIZE)
    dst = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
    m = cv2.getPerspectiveTransform(src, dst)
    wh = cfg.CAMERA_IMAGE_SIZE
    view = cv2.warpPerspective(
        world_bgr,
        m,
        (wh, wh),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return cv2.flip(view, 0)


def preprocess_bgr_for_model(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert BGR uint8 warp output to a normalized NCHW float tensor on ``device``.

    Matches training normalization (RGB channel order, ``NORMALIZE_MEAN`` / ``NORMALIZE_STD``).
    """
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


def run_simulation() -> tuple[np.ndarray, list[tuple[float, float]], str, float]:
    """
    Open-loop roll-out: load latest weights, integrate bicycle kinematics on the BEV map.

    Returns:
        ``world_bgr``, list of ``(x, y)`` poses, checkpoint path string, and ``px_per_m``.
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

    path: list[tuple[float, float]] = [(x0, y0)]
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

        with torch.no_grad():
            inp = preprocess_bgr_for_model(view, device)
            steering = float(model(inp).squeeze()[0].item())

        psi += steering * cfg.SIM_YAW_RATE_GAIN * cfg.SIM_DT
        x += step_dist_px * np.cos(psi)
        y += step_dist_px * np.sin(psi)
        path.append((float(x), float(y)))

        if x < 0 or x >= w or y < 0 or y >= h:
            break

    return world, path, ckpt, px_per_m


def main() -> None:
    """Run simulation, save path overlay and CSV, print stats, show matplotlib figure."""
    world_bgr, path, ckpt, px_per_m = run_simulation()
    path_arr = np.array(path, dtype=np.float64)

    vis = world_bgr.copy()
    for i in range(len(path) - 1):
        p0 = (int(round(path[i][0])), int(round(path[i][1])))
        p1 = (int(round(path[i + 1][0])), int(round(path[i + 1][1])))
        cv2.line(vis, p0, p1, (0, 0, 255), 2, cv2.LINE_AA)
    if len(path) > 0:
        sx, sy = int(round(path[0][0])), int(round(path[0][1]))
        ex, ey = int(round(path[-1][0])), int(round(path[-1][1]))
        cv2.circle(vis, (sx, sy), 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (ex, ey), 6, (255, 0, 0), -1, cv2.LINE_AA)

    out_dir = simulation_output_dir(ckpt)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "sim_path.png")
    out_csv = os.path.join(out_dir, "ego_path.csv")
    cv2.imwrite(out_png, vis)
    np.savetxt(out_csv, path_arr, delimiter=",", header="x,y", comments="")
    if len(path) > 1:
        dpx = float(np.sqrt(np.sum(np.diff(path_arr, axis=0) ** 2, axis=1)).sum())
        print(f"Poses: {len(path)}, path length ~ {dpx / px_per_m:.1f} m")
    else:
        print(f"Poses: {len(path)}")
    print(f"Saved overlay: {out_png!r}")
    print(f"Saved trajectory: {out_csv!r}")

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
