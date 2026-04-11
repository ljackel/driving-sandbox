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
    return os.path.dirname(os.path.abspath(__file__))


def latest_checkpoint_path() -> str | None:
    """Prefer newest weights under runs/*/; fall back to data/driving_net.pt."""
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
    Camera view with heading psi (rad), velocity direction (cos psi, sin psi) in image axes (+x right, +y down).
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

    if (src < 0).any() or (src[:, 0] >= w).any() or (src[:, 1] >= h).any():
        return None

    s = float(cfg.CAMERA_IMAGE_SIZE)
    dst = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
    m = cv2.getPerspectiveTransform(src, dst)
    wh = cfg.CAMERA_IMAGE_SIZE
    view = cv2.warpPerspective(world_bgr, m, (wh, wh))
    return cv2.flip(view, 0)


def preprocess_bgr_for_model(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    for c in range(3):
        t[c] = (t[c] - cfg.NORMALIZE_MEAN[c]) / cfg.NORMALIZE_STD[c]
    return t.unsqueeze(0).to(device)


def initial_heading_road_aligned(cs, y: float) -> float:
    """Heading (rad) tangent to road, driving toward decreasing y (up the map)."""
    dxdy = float(cs(y, nu=1))
    norm = float(np.hypot(dxdy, 1.0))
    fx = -dxdy / norm
    fy = -1.0 / norm
    return float(np.arctan2(fy, fx))


def run_simulation() -> tuple[np.ndarray, list[tuple[float, float]], str, float]:
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

    y0 = float(h - cfg.DATASET_MAP_MARGIN)
    x0 = float(dw.get_road_center(y0))
    psi = initial_heading_road_aligned(dw.cs, y0)

    model = DrivingNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    path: list[tuple[float, float]] = [(x0, y0)]
    x, y = x0, y0
    step_dist_px = cfg.SIM_SPEED_M_S * cfg.SIM_DT * px_per_m

    for _ in range(cfg.SIM_MAX_STEPS):
        view = get_view_from_pose(
            world,
            x,
            y,
            psi,
            cfg.SIM_EGO_LATERAL_OFFSET_PX,
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

    out_path = os.path.join(_project_root(), cfg.DATA_DIR, "sim_path.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"Checkpoint: {ckpt!r}")
    if len(path) > 1:
        dpx = float(np.sqrt(np.sum(np.diff(path_arr, axis=0) ** 2, axis=1)).sum())
        print(f"Poses: {len(path)}, path length ~ {dpx / px_per_m:.1f} m")
    else:
        print(f"Poses: {len(path)}")
    print(f"Saved overlay: {out_path!r}")

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
