import cv2
import numpy as np
import os
import pandas as pd
from generate_world import DrivingWorld


def signed_path_curvature(cs, y: float) -> float:
    """
    Curvature of the road centerline (x as a function of y) in bird's-eye pixels.
    Positive / negative encodes turn direction; scaled later to [-1, 1].
    """
    d1 = float(cs(y, nu=1))
    d2 = float(cs(y, nu=2))
    denom = (1.0 + d1 * d1) ** 1.5
    if denom < 1e-12:
        return 0.0
    return d2 / denom


def get_perspective_view(world_img, pos_y, pos_x, dxdY, lateral_offset_px=0.0):
    """
    Bird's-eye patch →128×128 forward view. Forward is decreasing y (up the image).
    dxdY: spline derivative dx/dy at the vehicle row.
    lateral_offset_px: distance from centerline toward the right-hand lane (see DrivingWorld lane width).
    """
    h, w = world_img.shape[:2]
    norm = float(np.hypot(dxdY, 1.0))
    fx = -dxdY / norm
    fy = -1.0 / norm
    rx, ry = fy, -fx
    r = np.array([rx, ry], dtype=np.float32)

    # Camera sits in the right lane: offset along +r (driver's right when facing forward on the map).
    near_c = np.array([pos_x, pos_y], dtype=np.float32) + r * float(lateral_offset_px)
    far_c = near_c + np.array([fx, fy], dtype=np.float32) * 20.0

    tl = far_c - r * 10.0
    tr = far_c + r * 10.0
    br = near_c + r * 60.0
    bl = near_c - r * 60.0
    src = np.float32([tl, tr, br, bl])

    if (src < 0).any() or (src[:, 0] >= w).any() or (src[:, 1] >= h).any():
        return None

    dst = np.float32([[0, 0], [128, 0], [128, 128], [0, 128]])
    m = cv2.getPerspectiveTransform(src, dst)
    view = cv2.warpPerspective(world_img, m, (128, 128))
    # Horizon (far) must appear at the top of the camera image; warp was vertically inverted.
    return cv2.flip(view, 0)


def generate_data():
    dw = DrivingWorld()
    world = dw.image
    size = dw.size
    half = size // 2

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    # 4 m lanes in DrivingWorld; right-lane center is 2 m from the dashed centerline.
    right_lane_offset_px = 2.0 * dw.px_per_m

    # Margin keeps perspective source quad inside the map
    margin = 80
    records = []
    for y in range(size - margin, margin, -10):
        folder = "train" if y > half else "test"
        yf = float(y)
        road_x = dw.get_road_center(yf)
        dxdY = float(dw.cs(yf, nu=1))

        view = get_perspective_view(
            world, y, road_x, dxdY, lateral_offset_px=right_lane_offset_px
        )
        if view is None:
            continue

        rel_path = f"{folder}/frame_{y:04d}.jpg"
        out = os.path.join("data", rel_path.replace("/", os.sep))
        cv2.imwrite(out, view)
        kappa = signed_path_curvature(dw.cs, yf)
        records.append((rel_path, kappa))

    if records:
        kappas = np.array([k for _, k in records], dtype=np.float64)
        scale = 1.0 / max(float(np.max(np.abs(kappas))), 1e-9)
        rows = [
            {
                "image_path": path,
                "steering": float(np.clip(k * scale, -1.0, 1.0)),
            }
            for path, k in records
        ]
        pd.DataFrame(rows).to_csv(
            os.path.join("data", "labels.csv"), index=False
        )

    print(f"Data split complete. ({len(records)} frames, labels in data/labels.csv)")


if __name__ == "__main__":
    generate_data()
