import cv2
import numpy as np
import os
import time
import pandas as pd

import config as cfg
from generate_world import DrivingWorld


def signed_path_curvature(cs, y: float) -> float:
    """
    Curvature of the road centerline (x as a function of y) in bird's-eye pixels.
    Positive / negative encodes turn direction; scaled later to [-1, 1].
    """
    d1 = float(cs(y, nu=1))
    d2 = float(cs(y, nu=2))
    denom = (1.0 + d1 * d1) ** 1.5
    if denom < cfg.CURVATURE_DENOM_EPS:
        return 0.0
    return d2 / denom


def get_perspective_view(
    world_img,
    pos_y,
    pos_x,
    dxdY,
    lateral_offset_px=0.0,
    yaw_offset_rad=0.0,
):
    """
    Bird's-eye patch →128×128 forward view. Forward is decreasing y (up the image).
    dxdY: spline derivative dx/dy at the vehicle row.
    lateral_offset_px: distance from centerline toward the right-hand lane (see DrivingWorld lane width).
    yaw_offset_rad: rotate camera heading CCW in image plane (rad) from path tangent.
    """
    h, w = world_img.shape[:2]
    norm = float(np.hypot(dxdY, 1.0))
    fx = -dxdY / norm
    fy = -1.0 / norm
    rx, ry = fy, -fx
    c = float(np.cos(yaw_offset_rad))
    s = float(np.sin(yaw_offset_rad))
    fx, fy = c * fx - s * fy, s * fx + c * fy
    rx, ry = c * rx - s * ry, s * rx + c * ry
    r = np.array([rx, ry], dtype=np.float32)
    f = np.array([fx, fy], dtype=np.float32)

    # Camera sits in the right lane: offset along +r (driver's right when facing forward on the map).
    near_c = np.array([pos_x, pos_y], dtype=np.float32) + r * float(lateral_offset_px)
    far_c = near_c + f * cfg.PERSPECTIVE_FAR_OFFSET_PX

    tl = far_c - r * float(cfg.PERSPECTIVE_FAR_HALF_WIDTH)
    tr = far_c + r * float(cfg.PERSPECTIVE_FAR_HALF_WIDTH)
    br = near_c + r * cfg.PERSPECTIVE_NEAR_HALF_WIDTH
    bl = near_c - r * cfg.PERSPECTIVE_NEAR_HALF_WIDTH
    src = np.float32([tl, tr, br, bl])

    if (src < 0).any() or (src[:, 0] >= w).any() or (src[:, 1] >= h).any():
        return None

    cam_s = float(cfg.CAMERA_IMAGE_SIZE)
    dst = np.float32([[0, 0], [cam_s, 0], [cam_s, cam_s], [0, cam_s]])
    m = cv2.getPerspectiveTransform(src, dst)
    wh = cfg.CAMERA_IMAGE_SIZE
    view = cv2.warpPerspective(world_img, m, (wh, wh))
    # Horizon (far) must appear at the top of the camera image; warp was vertically inverted.
    return cv2.flip(view, 0)


def save_labels_csv(df: pd.DataFrame) -> str:
    """
    Write labels to data/labels.csv. If the file is locked (Excel, editor preview),
    retry briefly, then fall back to data/labels_new.csv so the rest of the pipeline
    can still run.
    """
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    final_path = os.path.join(cfg.DATA_DIR, cfg.LABELS_CSV)
    tmp_path = os.path.join(cfg.DATA_DIR, cfg.LABELS_TMP)
    df.to_csv(tmp_path, index=False)
    for _ in range(cfg.LABELS_SAVE_RETRIES):
        try:
            os.replace(tmp_path, final_path)
            return final_path
        except PermissionError:
            time.sleep(cfg.LABELS_SAVE_RETRY_SLEEP_SEC)
    alt = os.path.join(cfg.DATA_DIR, cfg.LABELS_CSV_ALT)
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
    x, y = cfg.ANNOT_STEERING_POS
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = cfg.ANNOT_FONT_SCALE, cfg.ANNOT_FONT_THICKNESS
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv2.putText(
            img, label, (x + dx, y + dy), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA
        )
    cv2.putText(img, label, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def generate_data(num_train=cfg.NUM_TRAIN_FRAMES):
    dw = DrivingWorld()
    world = dw.image
    size = dw.size
    half = size // 2

    os.makedirs(os.path.join(cfg.DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(cfg.DATA_DIR, "test"), exist_ok=True)

    right_lane_offset_px = cfg.RIGHT_LANE_OFFSET_METERS * dw.px_per_m

    margin = cfg.DATASET_MAP_MARGIN
    records: list[tuple[str, float, float, float]] = []
    rng = np.random.default_rng(cfg.DATASET_SEED)
    yaw_std = float(np.deg2rad(cfg.TRAIN_PERTURB_YAW_STD_DEG))

    # Training: evenly sample along the road (bottom → top) in the y > half half of the map
    y_hi = float(size - margin)
    y_lo = float(half + 1)
    y_samples = np.linspace(y_hi, y_lo, num_train, dtype=np.float64)

    # Half clean (nominal right-lane pose), half Gaussian lateral + yaw vs path tangent.
    for i, yf in enumerate(y_samples):
        road_x = dw.get_road_center(yf)
        dxdY = float(dw.cs(yf, nu=1))
        view = get_perspective_view(
            world, yf, road_x, dxdY, lateral_offset_px=right_lane_offset_px
        )
        if view is None:
            continue
        rel_path = f"train/frame_{i:04d}.jpg"
        out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
        cv2.imwrite(out, view)
        kappa = signed_path_curvature(dw.cs, yf)
        records.append((rel_path, kappa, 0.0, 0.0))

    for j, yf in enumerate(y_samples):
        road_x = dw.get_road_center(yf)
        dxdY = float(dw.cs(yf, nu=1))
        idx = num_train + j
        view = None
        lat_m = 0.0
        yaw_rad = 0.0
        for _ in range(cfg.TRAIN_PERTURB_VIEW_RETRIES):
            lat_m = float(rng.normal(0.0, cfg.TRAIN_PERTURB_LATERAL_STD_M))
            yaw_rad = float(rng.normal(0.0, yaw_std))
            lateral_px = right_lane_offset_px + lat_m * dw.px_per_m
            view = get_perspective_view(
                world,
                yf,
                road_x,
                dxdY,
                lateral_offset_px=lateral_px,
                yaw_offset_rad=yaw_rad,
            )
            if view is not None:
                break
        if view is None:
            continue
        rel_path = f"train/frame_{idx:04d}.jpg"
        out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
        cv2.imwrite(out, view)
        kappa = signed_path_curvature(dw.cs, yf)
        records.append((rel_path, kappa, lat_m, yaw_rad))

    # Test: stepped integer rows in y <= half (same style as before)
    for y in range(size - margin, margin, -cfg.TEST_Y_STEP):
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
        out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
        cv2.imwrite(out, view)
        kappa = signed_path_curvature(dw.cs, yf)
        records.append((rel_path, kappa, 0.0, 0.0))

    if records:
        kappas = np.array([k for _, k, _, _ in records], dtype=np.float64)
        scale = 1.0 / max(float(np.max(np.abs(kappas))), cfg.KAPPA_SCALE_EPS)
        rows = []
        for path, k, lat_m, yaw_rad in records:
            kappa_scaled = float(k * scale)
            steering = (
                kappa_scaled
                - cfg.TRAIN_PERTURB_RECENTER_GAIN_LAT * lat_m
                - cfg.TRAIN_PERTURB_RECENTER_GAIN_YAW * yaw_rad
            )
            rows.append(
                {
                    "image_path": path,
                    "steering": float(
                        np.clip(
                            steering,
                            cfg.STEERING_CLIP_MIN,
                            cfg.STEERING_CLIP_MAX,
                        )
                    ),
                }
            )

        os.makedirs(os.path.join(cfg.DATA_DIR, "test_labeled"), exist_ok=True)
        for row in rows:
            if not row["image_path"].startswith("test/"):
                continue
            src = os.path.join(cfg.DATA_DIR, row["image_path"].replace("/", os.sep))
            annotated = cv2.imread(src)
            if annotated is None:
                continue
            annotate_steering_bgr(annotated, row["steering"])
            base = os.path.basename(row["image_path"])
            cv2.imwrite(os.path.join(cfg.DATA_DIR, "test_labeled", base), annotated)

        labels_path = save_labels_csv(pd.DataFrame(rows))
        n_train = sum(1 for p, _, _, _ in records if p.startswith("train/"))
        n_test = sum(1 for p, _, _, _ in records if p.startswith("test/"))
        print(
            f"Data split complete. ({n_train} train, {n_test} test, labels in {labels_path!r}; "
            "test previews with steering in data/test_labeled/)"
        )
    else:
        print("Data split complete. (0 frames)")


if __name__ == "__main__":
    generate_data()
