"""
Render perspective crops and ``data/labels.csv``: train from bottom BEV half, test from top half,
optional perturbations (shared ``PERTURB_*``), global kappa scaling, and steering recentering on perturbed rows.
"""
import math
import os
import time

import cv2
import numpy as np
import pandas as pd

import config as cfg
from perspective_camera import perspective_camera_view
from reproducibility import set_global_seed

set_global_seed(cfg.DATASET_SEED)

from generate_world import DrivingWorld


def signed_path_curvature(cs, y: float) -> float:
    """
    Signed curvature κ of the centerline treated as ``x(y)`` in bird's-eye pixel coordinates.

    Uses the standard planar formula for a graph. ``generate_data`` scales κ by ``1/max|κ|`` over
    **all** label rows (train + test, clean + perturbed) before clipping to steering in ``[-1, 1]``,
    then subtracts lateral/yaw recentering terms on perturbed rows.

    Args:
        cs: Spline ``x(y)``.
        y: Sample row (pixels).
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
    Warp a forward-facing ``CAMERA_IMAGE_SIZE`` crop from the BEV ``world_img``.

    Forward direction follows the path tangent (decreasing ``y`` / up the image) plus optional
    ``yaw_offset_rad``. Lateral shift moves the viewpoint along driver's right in pixels
    (isotropic meters-to-pixels scale).

    Args:
        world_img: Full map (BGR).
        pos_y, pos_x: Sample point on the centerline (pixels).
        dxdY: ``dx/dy`` of the centerline spline at ``pos_y``.
        lateral_offset_px: Signed offset along local right, in pixels.
        yaw_offset_rad: Extra CCW rotation (rad) applied to forward/right in the image plane.

    Returns:
        Square BGR image, or ``None`` if the perspective source quad is out of bounds.
    """
    norm = float(np.hypot(dxdY, 1.0))
    fx = -dxdY / norm
    fy = -1.0 / norm
    # Match simulate.get_view_from_pose: for f = (cos ψ, sin ψ), right is (-sin ψ, cos ψ) = (-fy, fx).
    rx, ry = -fy, fx
    c = float(np.cos(yaw_offset_rad))
    s = float(np.sin(yaw_offset_rad))
    fx, fy = c * fx - s * fy, s * fx + c * fy
    rx, ry = c * rx - s * ry, s * rx + c * ry
    r = np.array([rx, ry], dtype=np.float32)
    f = np.array([fx, fy], dtype=np.float32)

    # Camera sits in the right lane: offset along +r (driver's right when facing forward on the map).
    near_c = np.array([pos_x, pos_y], dtype=np.float32) + r * float(lateral_offset_px)
    return perspective_camera_view(world_img, near_c, f, r)


# Large lateral/yaw often pushes the BEV frustum off-map; scale the same draw down until it fits.
_PERTURB_BACKOFF_SCALES = (1.0, 0.65, 0.4, 0.2, 0.08, 0.03, 0.01, 0.0)


def _sample_perturbed_perspective_view(
    world_img,
    yf: float,
    road_x: float,
    dxdY: float,
    right_lane_offset_px: float,
    px_per_m: float,
    rng: np.random.Generator,
    lateral_std_m: float,
    yaw_std_rad: float,
):
    """
    Sample Gaussian lateral (m) and yaw (rad), shrinking until ``get_perspective_view`` succeeds.

    Returns:
        ``(view, lat_m, yaw_rad)`` with ``view`` possibly ``None`` if every attempt fails.
    """
    for _ in range(cfg.TRAIN_PERTURB_VIEW_RETRIES):
        lat_draw = float(rng.normal(0.0, lateral_std_m))
        yaw_draw = float(rng.normal(0.0, yaw_std_rad))

        def _warp_attempt(lat_m: float, yaw_rad: float):
            lateral_px = right_lane_offset_px + lat_m * px_per_m
            return get_perspective_view(
                world_img,
                yf,
                road_x,
                dxdY,
                lateral_offset_px=lateral_px,
                yaw_offset_rad=yaw_rad,
            )

        # Coupled backoff (same factor on lateral and yaw).
        for b in _PERTURB_BACKOFF_SCALES:
            lat_m = float(lat_draw * b)
            yaw_rad = float(yaw_draw * b)
            view = _warp_attempt(lat_m, yaw_rad)
            if view is not None:
                return view, lat_m, yaw_rad
        # Full lateral from draw; reduce yaw only (yaw often leaves the map first).
        for b_yaw in _PERTURB_BACKOFF_SCALES:
            lat_m = float(lat_draw)
            yaw_rad = float(yaw_draw * b_yaw)
            view = _warp_attempt(lat_m, yaw_rad)
            if view is not None:
                return view, lat_m, yaw_rad
        # Full yaw from draw, reduce lateral only.
        for b_lat in _PERTURB_BACKOFF_SCALES:
            lat_m = float(lat_draw * b_lat)
            yaw_rad = float(yaw_draw)
            view = _warp_attempt(lat_m, yaw_rad)
            if view is not None:
                return view, lat_m, yaw_rad
    return None, 0.0, 0.0


def save_labels_csv(df: pd.DataFrame) -> str:
    """
    Atomically write ``df`` to ``DATA_DIR/LABELS_CSV`` via a temp file and replace.

    Retries on ``PermissionError`` (e.g. file open in Excel). If replacement still fails,
    writes ``LABELS_CSV_ALT`` instead.

    Args:
        df: Rows with ``image_path`` and ``steering`` columns.

    Returns:
        Path to the CSV that was successfully written.
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


def _draw_train_near_vehicle_row_debug_bgr(img: np.ndarray) -> None:
    """
    In-place debug: red horizontal line on the crop row nearest the vehicle.

    The perspective warp maps the BEV near edge to the bottom of the square image (+y down).
    """
    h, w = img.shape[:2]
    y = h - 1
    cv2.line(img, (0, y), (w - 1, y), (0, 0, 255), thickness=2)


def annotate_steering_bgr(img: np.ndarray, steering: float) -> None:
    """
    Draw the normalized steering value as white text with black outline (in-place).

    Args:
        img: BGR image to modify.
        steering: Clipped steering label in ``[-1, 1]``.
    """
    label = f"steering: {steering:+.4f}"
    x, y = cfg.ANNOT_STEERING_POS
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = cfg.ANNOT_FONT_SCALE, cfg.ANNOT_FONT_THICKNESS
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv2.putText(
            img, label, (x + dx, y + dy), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA
        )
    cv2.putText(img, label, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _put_outlined_lines_bgr(
    img: np.ndarray,
    lines: list[str],
    org_xy: tuple[int, int],
    line_step_px: int = 14,
) -> None:
    """Draw stacked lines of white text with black outline (in-place)."""
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


def annotate_perturb_debug_bgr(
    img: np.ndarray,
    lat_m: float,
    yaw_rad: float,
    kappa_raw: float,
) -> None:
    """
    Draw sampled perturbation and raw curvature on a BGR image (in-place).

    Used only for ``TRAIN_PERTURB_DEBUG_SUBDIR`` companions; not for training labels.
    """
    yaw_deg = math.degrees(yaw_rad)
    lines = [
        f"perturb lat: {lat_m:+.4f} m",
        f"perturb yaw: {yaw_deg:+.3f} deg ({yaw_rad:+.4f} rad)",
        f"raw kappa: {kappa_raw:+.6e}",
    ]
    _put_outlined_lines_bgr(img, lines, cfg.ANNOT_STEERING_POS)


def generate_data(num_train=cfg.NUM_TRAIN_FRAMES, num_test=cfg.NUM_TEST_FRAMES):
    """
    Render train/test perspective frames, steering labels, and ``data/labels.csv``.

    **Spatial split:** train ``y`` spans the bottom BEV half (large ``y``); test ``y`` the top half
    (small ``y``), separated at ``size // 2`` so there is no leakage.

    **Train filenames:** ``train/frame_{0..num_train-1}.jpg`` (clean), then
    ``train/frame_{num_train..2*num_train-1}.jpg`` (aligned perturbed, same ``y`` grid as clean) when
    ``PERTURB_LATERAL_STD_M`` or ``PERTURB_YAW_STD_DEG`` > 0, then
    ``train/frame_{2*num_train..}.jpg`` for ``TRAIN_PERTURB_EXTRA_FRAMES`` extra perturbed frames with
    ``y`` drawn uniformly from the train grid (same Gaussian lateral/yaw and backoff as aligned).

    **Test filenames:** ``test/frame_{0..num_test-1}.jpg`` (clean), then
    ``test/frame_{num_test..2*num_test-1}.jpg`` (perturbed) when perturbations are on.

    Args:
        num_train: Number of clean road samples (and of aligned perturbed mates if perturbing).
        num_test: Number of clean test samples (and of aligned perturbed mates if perturbing).
    """
    dw = DrivingWorld()
    world = dw.image
    size = dw.size

    os.makedirs(os.path.join(cfg.DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(cfg.DATA_DIR, "test"), exist_ok=True)

    right_lane_offset_px = (
        cfg.LANE_WIDTH_METERS * cfg.DATASET_RIGHT_LANE_LATERAL_FRAC * dw.px_per_m
    )

    margin = cfg.DATASET_MAP_MARGIN
    records: list[tuple[str, float, float, float]] = []

    half = size // 2
    train_y = np.linspace(
        float(size - margin),
        float(half) + 1.0,
        int(num_train),
        dtype=np.float64,
    )
    test_y = np.linspace(
        float(half),
        float(margin),
        int(num_test),
        dtype=np.float64,
    )

    # Right-lane center, heading = road tangent (no lateral/yaw noise unless perturbation σ > 0).
    for i, yf in enumerate(train_y):
        road_x = dw.get_road_center(yf)
        dxdY = float(dw.cs(yf, nu=1))
        view = get_perspective_view(
            world,
            yf,
            road_x,
            dxdY,
            lateral_offset_px=right_lane_offset_px,
            yaw_offset_rad=0.0,
        )
        if view is None:
            continue
        rel_path = f"train/frame_{i:04d}.jpg"
        out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
        _draw_train_near_vehicle_row_debug_bgr(view)
        cv2.imwrite(out, view)
        kappa = signed_path_curvature(dw.cs, yf)
        records.append((rel_path, kappa, 0.0, 0.0))

    perturb_train = (
        cfg.PERTURB_LATERAL_STD_M > 0.0 or cfg.PERTURB_YAW_STD_DEG > 0.0
    )
    if perturb_train:
        debug_dir = os.path.join(cfg.DATA_DIR, cfg.TRAIN_PERTURB_DEBUG_SUBDIR)
        os.makedirs(debug_dir, exist_ok=True)
        rng = np.random.default_rng(cfg.DATASET_SEED)
        yaw_std = float(np.deg2rad(cfg.PERTURB_YAW_STD_DEG))
        for j, yf in enumerate(train_y):
            road_x = dw.get_road_center(yf)
            dxdY = float(dw.cs(yf, nu=1))
            idx = num_train + j
            view, lat_m, yaw_rad = _sample_perturbed_perspective_view(
                world,
                yf,
                road_x,
                dxdY,
                right_lane_offset_px,
                dw.px_per_m,
                rng,
                cfg.PERTURB_LATERAL_STD_M,
                yaw_std,
            )
            if view is None:
                continue
            rel_path = f"train/frame_{idx:04d}.jpg"
            out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
            _draw_train_near_vehicle_row_debug_bgr(view)
            cv2.imwrite(out, view)
            kappa = signed_path_curvature(dw.cs, yf)
            records.append((rel_path, kappa, lat_m, yaw_rad))
            dbg = view.copy()
            annotate_perturb_debug_bgr(dbg, lat_m, yaw_rad, kappa)
            dbg_name = f"frame_{idx:04d}.jpg"
            cv2.imwrite(os.path.join(debug_dir, dbg_name), dbg)

        extra_n = int(cfg.TRAIN_PERTURB_EXTRA_FRAMES)
        if extra_n > 0:
            base_idx = 2 * num_train
            for k in range(extra_n):
                yf = float(rng.choice(train_y))
                road_x = dw.get_road_center(yf)
                dxdY = float(dw.cs(yf, nu=1))
                idx = base_idx + k
                view, lat_m, yaw_rad = _sample_perturbed_perspective_view(
                    world,
                    yf,
                    road_x,
                    dxdY,
                    right_lane_offset_px,
                    dw.px_per_m,
                    rng,
                    cfg.PERTURB_LATERAL_STD_M,
                    yaw_std,
                )
                if view is None:
                    continue
                rel_path = f"train/frame_{idx:04d}.jpg"
                out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
                _draw_train_near_vehicle_row_debug_bgr(view)
                cv2.imwrite(out, view)
                kappa = signed_path_curvature(dw.cs, yf)
                records.append((rel_path, kappa, lat_m, yaw_rad))
                dbg = view.copy()
                annotate_perturb_debug_bgr(dbg, lat_m, yaw_rad, kappa)
                dbg_name = f"frame_{idx:04d}.jpg"
                cv2.imwrite(os.path.join(debug_dir, dbg_name), dbg)

    # Test: same y grid shape as train (clean then aligned perturbed with matching indices).
    for i, yf in enumerate(test_y):
        road_x = dw.get_road_center(yf)
        dxdY = float(dw.cs(yf, nu=1))
        view = get_perspective_view(
            world,
            yf,
            road_x,
            dxdY,
            lateral_offset_px=right_lane_offset_px,
            yaw_offset_rad=0.0,
        )
        if view is None:
            continue
        rel_path = f"test/frame_{i:04d}.jpg"
        out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
        cv2.imwrite(out, view)
        kappa = signed_path_curvature(dw.cs, yf)
        records.append((rel_path, kappa, 0.0, 0.0))

    perturb_test = (
        cfg.PERTURB_LATERAL_STD_M > 0.0 or cfg.PERTURB_YAW_STD_DEG > 0.0
    )
    if perturb_test:
        rng_test = np.random.default_rng(
            cfg.DATASET_SEED + cfg.TEST_PERTURB_SEED_OFFSET
        )
        yaw_std = float(np.deg2rad(cfg.PERTURB_YAW_STD_DEG))
        for j, yf in enumerate(test_y):
            road_x = dw.get_road_center(yf)
            dxdY = float(dw.cs(yf, nu=1))
            idx = int(num_test) + j
            view, lat_m, yaw_rad = _sample_perturbed_perspective_view(
                world,
                yf,
                road_x,
                dxdY,
                right_lane_offset_px,
                dw.px_per_m,
                rng_test,
                cfg.PERTURB_LATERAL_STD_M,
                yaw_std,
            )
            if view is None:
                continue
            rel_path = f"test/frame_{idx:04d}.jpg"
            out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
            cv2.imwrite(out, view)
            kappa = signed_path_curvature(dw.cs, yf)
            records.append((rel_path, kappa, lat_m, yaw_rad))

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
        extra = ""
        if perturb_train:
            extra = (
                f"; perturbed train debug overlays in "
                f"{os.path.join(cfg.DATA_DIR, cfg.TRAIN_PERTURB_DEBUG_SUBDIR)!r}"
            )
        if perturb_test:
            extra += (
                f"; test includes perturbed frames (test/frame_{num_test:04d}.jpg through "
                f"test/frame_{2 * num_test - 1:04d}.jpg, same sigma as train)"
            )
        print(
            f"Data split complete. ({n_train} train, {n_test} test, labels in {labels_path!r}; "
            f"test previews with steering in data/test_labeled/{extra})"
        )
    else:
        print("Data split complete. (0 frames)")


if __name__ == "__main__":
    generate_data()
