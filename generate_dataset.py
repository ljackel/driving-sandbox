import cv2
import numpy as np
import os
from generate_world import DrivingWorld


def get_perspective_view(world_img, pos_y, pos_x, dxdY):
    """
    Bird's-eye patch →128×128 forward view. Forward is decreasing y (up the image).
    dxdY: spline derivative dx/dy at the vehicle row.
    """
    h, w = world_img.shape[:2]
    norm = float(np.hypot(dxdY, 1.0))
    fx = -dxdY / norm
    fy = -1.0 / norm
    rx, ry = fy, -fx

    near_c = np.array([pos_x, pos_y], dtype=np.float32)
    far_c = near_c + np.array([fx, fy], dtype=np.float32) * 20.0
    r = np.array([rx, ry], dtype=np.float32)

    tl = far_c - r * 10.0
    tr = far_c + r * 10.0
    br = near_c + r * 60.0
    bl = near_c - r * 60.0
    src = np.float32([tl, tr, br, bl])

    if (src < 0).any() or (src[:, 0] >= w).any() or (src[:, 1] >= h).any():
        return None

    dst = np.float32([[0, 0], [128, 0], [128, 128], [0, 128]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(world_img, m, (128, 128))


def generate_data():
    dw = DrivingWorld()
    world = dw.image
    size = dw.size
    half = size // 2

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    d_cs = dw.cs.derivative()

    # Margin keeps perspective source quad inside the map
    margin = 80
    for y in range(size - margin, margin, -10):
        folder = "train" if y > half else "test"
        yf = float(y)
        road_x = dw.get_road_center(yf)
        dxdY = float(d_cs(yf))

        view = get_perspective_view(world, y, road_x, dxdY)
        if view is None:
            continue

        out = os.path.join("data", folder, f"frame_{y:04d}.jpg")
        cv2.imwrite(out, view)

    print("Data split complete.")


if __name__ == "__main__":
    generate_data()
