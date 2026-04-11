import cv2
import numpy as np
import os
import time
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


def save_labels_csv(df: pd.DataFrame) -> str:
    """
    Write labels to data/labels.csv. If the file is locked (Excel, editor preview),
    retry briefly, then fall back to data/labels_new.csv so the rest of the pipeline
    can still run.
    """
    os.makedirs("data", exist_ok=True)
    final_path = os.path.join("data", "labels.csv")
    tmp_path = os.path.join("data", "labels.partial.tmp")
    df.to_csv(tmp_path, index=False)
    for _ in range(20):
        try:
            os.replace(tmp_path, final_path)
            return final_path
        except PermissionError:
            time.sleep(0.15)
    alt = os.path.join("data", "labels_new.csv")
    os.replace(tmp_path, alt)
    print(
        "\nWARNING: could not replace data/labels.csv (is it open in another app?). "
        f"Labels written to {alt}. Close the lock, then rename it to labels.csv or "
        "delete the old CSV and rename.\n"
    )
    return alt


def annotate_steering_bgr(img: np.ndarray, steering: float) -> None:
    """Draw normalized steering label on a BGR image (in-place)."""
    label = f"steering: {steering:+.4f}"
    x, y = 4, 14
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.4, 1
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv2.putText(
            img, label, (x + dx, y + dy), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA
        )
    cv2.putText(img, label, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def generate_data(num_train=1000):
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

    # Training: evenly sample along the road (bottom → top) in the y > half half of the map
    y_hi = float(size - margin)
    y_lo = float(half + 1)
    for i, yf in enumerate(np.linspace(y_hi, y_lo, num_train, dtype=np.float64)):
        road_x = dw.get_road_center(yf)
        dxdY = float(dw.cs(yf, nu=1))
        view = get_perspective_view(
            world, yf, road_x, dxdY, lateral_offset_px=right_lane_offset_px
        )
        if view is None:
            continue
        rel_path = f"train/frame_{i:04d}.jpg"
        out = os.path.join("data", rel_path.replace("/", os.sep))
        cv2.imwrite(out, view)
        kappa = signed_path_curvature(dw.cs, yf)
        records.append((rel_path, kappa))

    # Test: stepped integer rows in y <= half (same style as before)
    for y in range(size - margin, margin, -10):
        if y > half:
            continue
        yf = float(y)
        road_x = dw.get_road_center(yf)
        dxdY = float(dw.cs(yf, nu=1))
        view = get_perspective_view(
            world, yf, road_x, dxdY, lateral_offset_px=right_lane_offset_px
        )
        if view is None:
            continue
        rel_path = f"test/frame_{y:04d}.jpg"
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

        os.makedirs("data/test_labeled", exist_ok=True)
        for row in rows:
            if not row["image_path"].startswith("test/"):
                continue
            src = os.path.join("data", row["image_path"].replace("/", os.sep))
            annotated = cv2.imread(src)
            if annotated is None:
                continue
            annotate_steering_bgr(annotated, row["steering"])
            base = os.path.basename(row["image_path"])
            cv2.imwrite(os.path.join("data", "test_labeled", base), annotated)

        labels_path = save_labels_csv(pd.DataFrame(rows))
        n_train = sum(1 for p, _ in records if p.startswith("train/"))
        n_test = sum(1 for p, _ in records if p.startswith("test/"))
        print(
            f"Data split complete. ({n_train} train, {n_test} test, labels in {labels_path!r}; "
            "test previews with steering in data/test_labeled/)"
        )
    else:
        print("Data split complete. (0 frames)")


if __name__ == "__main__":
    generate_data()
