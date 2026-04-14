"""
Render perspective crops and ``data/labels.csv``.

By default train samples the bottom BEV half and test the top half. With
``DATASET_MIX_TRAIN_TEST_GEOGRAPHY``, both splits mix the full road (shuffle-split). Lateral/yaw
perturbations when ``DATASET_PERTURBATIONS_ENABLE`` and ``PERTURB_*`` σ > 0; global κ scaling and clip.
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

from dataset_split import (
    dataset_train_test_y,
    offramp_clean_counts_matching_main_spacing,
    offramp_train_u_rid_pairs_main_spacing,
)
from generate_world import DrivingWorld, lateral_offset_px_avoid_roadkill


def _psi_from_dxdY(dxdY: float) -> float:
    """Road tangent (rad) matching ``simulate.initial_heading_road_aligned`` for graph ``x(y)``."""
    norm = float(np.hypot(float(dxdY), 1.0))
    return float(np.arctan2(-1.0 / norm, -float(dxdY) / norm))


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


def get_perspective_view_from_forward(
    world_img,
    pos_x,
    pos_y,
    fx,
    fy,
    lateral_offset_px=0.0,
    yaw_offset_rad=0.0,
):
    """
    Same warp as ``get_perspective_view`` but with an explicit unitless forward direction ``(fx, fy)``
    in BEV (+x right, +y down), e.g. from a parametric off-ramp tangent.
    """
    norm = float(np.hypot(fx, fy))
    if norm < 1e-9:
        return None
    fx, fy = fx / norm, fy / norm
    rx, ry = -fy, fx
    c = float(np.cos(yaw_offset_rad))
    s = float(np.sin(yaw_offset_rad))
    fx, fy = c * fx - s * fy, s * fx + c * fy
    rx, ry = c * rx - s * ry, s * rx + c * ry
    r = np.array([rx, ry], dtype=np.float32)
    f = np.array([fx, fy], dtype=np.float32)
    near_c = np.array([pos_x, pos_y], dtype=np.float32) + r * float(lateral_offset_px)
    return perspective_camera_view(world_img, near_c, f, r)


# Large lateral/yaw often pushes the BEV frustum off-map; scale the same draw down until it fits.
_PERTURB_BACKOFF_SCALES = (1.0, 0.65, 0.4, 0.2, 0.08, 0.03, 0.01, 0.0)


def _sample_perturbed_perspective_view(
    world_img,
    yf: float,
    road_x: float,
    dxdY: float,
    lateral_base_px: float,
    px_per_m: float,
    rng: np.random.Generator,
    lateral_std_m: float,
    yaw_std_rad: float,
):
    """
    Sample Gaussian lateral (m) and yaw (rad), shrinking until ``get_perspective_view`` succeeds.

    ``lateral_base_px`` is the nominal offset at this row (includes roadkill detour when enabled).

    Returns:
        ``(view, lat_m, yaw_rad)`` with ``view`` possibly ``None`` if every attempt fails.
    """
    for _ in range(cfg.TRAIN_PERTURB_VIEW_RETRIES):
        lat_draw = float(rng.normal(0.0, lateral_std_m))
        yaw_draw = float(rng.normal(0.0, yaw_std_rad))

        def _warp_attempt(lat_m: float, yaw_rad: float):
            lateral_px = lateral_base_px + lat_m * px_per_m
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
        df: Rows with ``image_path``, ``steering``, and ``take_offramp`` columns.

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


def _clear_jpgs_in_dir(dir_path: str) -> None:
    """Remove ``*.jpg`` in ``dir_path`` if the directory exists (stale frames from smaller runs)."""
    if not os.path.isdir(dir_path):
        return
    for name in os.listdir(dir_path):
        if name.lower().endswith(".jpg"):
            os.remove(os.path.join(dir_path, name))


def generate_data(num_train=cfg.NUM_TRAIN_FRAMES, num_test=cfg.NUM_TEST_FRAMES):
    """
    Render train/test perspective frames, steering labels, and ``data/labels.csv``.

    Existing ``*.jpg`` under ``data/train``, ``data/test``, ``data/test_labeled``, and
    ``data/{TRAIN_PERTURB_DEBUG_SUBDIR}`` are deleted first so folder counts match the new run.

    **Spatial split:** if ``DATASET_MIX_TRAIN_TEST_GEOGRAPHY`` is false, train ``y`` is only the bottom
    BEV half and test ``y`` only the top half (no overlap). If true, both splits sample the full road
    via shuffle-split (see ``dataset_train_test_y``).

    **Train filenames:** ``train/frame_{0..num_train-1}.jpg`` (clean), then
    ``train/frame_{num_train..2*num_train-1}.jpg`` (aligned perturbed) when
    ``DATASET_PERTURBATIONS_ENABLE`` and lateral or yaw σ > 0, then
    ``train/frame_{2*num_train..}.jpg`` for ``TRAIN_PERTURB_EXTRA_FRAMES`` extra perturbed frames.

    **Test filenames:** ``test/frame_{0..num_test-1}.jpg`` (clean), then
    ``test/frame_{num_test..2*num_test-1}.jpg`` (perturbed) under the same perturbation conditions.

    **Roadkill:** when ``ROADKILL_ENABLE``, the map may show extra splats in
    ``ROADKILL_EVAL_ONLY_OBSTACLE_Y_PX``, but **main-road** crops and lateral reference for labels use only
    ``ROADKILL_OBSTACLE_Y_PX`` (``lateral_offset_px_avoid_roadkill(..., for_training_labels=True)``).
    ``simulate`` uses the full set (training + eval-only) for open-loop generalization tests.

    **Off-ramp:** when ``OFFRAMP_ENABLE`` and ``DATASET_OFFRAMP_LABELS_ENABLE``, also writes
    ``train/offramp_*.jpg`` / ``test/offramp_*.jpg`` with ``take_offramp=1`` and κ from the Bézier.
    **Train** ramp crops: on each ramp, samples are uniform in arc length (or linear in ``u``) along the
    portion inside the train BEV band, at the **same mean spacing** as main-road train rows; the total
    is capped by ``DATASET_OFFRAMP_*_FRAMES`` when ``DATASET_OFFRAMP_MATCH_MAIN_SPACING`` is true.

    Args:
        num_train: Number of clean road samples (and of aligned perturbed mates if perturbing).
        num_test: Number of clean test samples (and of aligned perturbed mates if perturbing).
    """
    dw = DrivingWorld()
    world = dw.image
    size = dw.size

    os.makedirs(os.path.join(cfg.DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(cfg.DATA_DIR, "test"), exist_ok=True)
    _clear_jpgs_in_dir(os.path.join(cfg.DATA_DIR, "train"))
    _clear_jpgs_in_dir(os.path.join(cfg.DATA_DIR, "test"))
    _clear_jpgs_in_dir(os.path.join(cfg.DATA_DIR, "test_labeled"))
    _clear_jpgs_in_dir(
        os.path.join(cfg.DATA_DIR, cfg.TRAIN_PERTURB_DEBUG_SUBDIR)
    )

    right_lane_offset_px = (
        cfg.LANE_WIDTH_METERS * cfg.DATASET_RIGHT_LANE_LATERAL_FRAC * dw.px_per_m
    )

    margin = cfg.DATASET_MAP_MARGIN
    records: list[tuple[str, float, float, float, int]] = []

    train_y, test_y = dataset_train_test_y(
        int(num_train),
        int(num_test),
        size,
        margin,
        mix_train_test_geography=cfg.DATASET_MIX_TRAIN_TEST_GEOGRAPHY,
        seed=cfg.DATASET_SEED,
        uniform_along_road=cfg.DATASET_SAMPLE_UNIFORM_ALONG_ROAD,
        road_cs=dw.cs,
    )

    L_ramp_px = float(dw.offramp_total_arc_length_px())
    n_offramp_train, n_offramp_test = offramp_clean_counts_matching_main_spacing(
        L_ramp_px,
        int(num_train),
        int(num_test),
        size,
        float(margin),
        mix_train_test_geography=cfg.DATASET_MIX_TRAIN_TEST_GEOGRAPHY,
        cs=dw.cs,
        cap_train=int(cfg.DATASET_OFFRAMP_TRAIN_FRAMES),
        cap_test=int(cfg.DATASET_OFFRAMP_TEST_FRAMES),
        match_spacing=cfg.DATASET_OFFRAMP_MATCH_MAIN_SPACING,
    )

    # Right-lane center except roadkill detour; heading = road tangent (no lateral/yaw noise unless σ > 0).
    for i, yf in enumerate(train_y):
        road_x = dw.get_road_center(yf)
        dxdY = float(dw.cs(yf, nu=1))
        psi_road = _psi_from_dxdY(dxdY)
        lat_px = lateral_offset_px_avoid_roadkill(
            float(yf),
            right_lane_offset_px,
            for_training_labels=True,
            world_bgr=world,
            dw=dw,
            x_center=float(road_x),
            psi=psi_road,
        )
        view = get_perspective_view(
            world,
            yf,
            road_x,
            dxdY,
            lateral_offset_px=lat_px,
            yaw_offset_rad=0.0,
        )
        if view is None:
            continue
        rel_path = f"train/frame_{i:04d}.jpg"
        out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
        _draw_train_near_vehicle_row_debug_bgr(view)
        cv2.imwrite(out, view)
        kappa = signed_path_curvature(dw.cs, yf)
        records.append((rel_path, kappa, 0.0, 0.0, 0))

    perturb_train = cfg.DATASET_ALIGNED_PERTURB
    if perturb_train:
        debug_dir = os.path.join(cfg.DATA_DIR, cfg.TRAIN_PERTURB_DEBUG_SUBDIR)
        os.makedirs(debug_dir, exist_ok=True)
        rng = np.random.default_rng(cfg.DATASET_SEED)
        yaw_std = float(np.deg2rad(cfg.PERTURB_YAW_STD_DEG))
        for j, yf in enumerate(train_y):
            road_x = dw.get_road_center(yf)
            dxdY = float(dw.cs(yf, nu=1))
            psi_road = _psi_from_dxdY(dxdY)
            idx = num_train + j
            view, lat_m, yaw_rad = _sample_perturbed_perspective_view(
                world,
                yf,
                road_x,
                dxdY,
                lateral_offset_px_avoid_roadkill(
                    float(yf),
                    right_lane_offset_px,
                    for_training_labels=True,
                    world_bgr=world,
                    dw=dw,
                    x_center=float(road_x),
                    psi=psi_road,
                ),
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
            records.append((rel_path, kappa, lat_m, yaw_rad, 0))
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
                psi_road = _psi_from_dxdY(dxdY)
                idx = base_idx + k
                view, lat_m, yaw_rad = _sample_perturbed_perspective_view(
                    world,
                    yf,
                    road_x,
                    dxdY,
                    lateral_offset_px_avoid_roadkill(
                        float(yf),
                        right_lane_offset_px,
                        for_training_labels=True,
                        world_bgr=world,
                        dw=dw,
                        x_center=float(road_x),
                        psi=psi_road,
                    ),
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
                records.append((rel_path, kappa, lat_m, yaw_rad, 0))
                dbg = view.copy()
                annotate_perturb_debug_bgr(dbg, lat_m, yaw_rad, kappa)
                dbg_name = f"frame_{idx:04d}.jpg"
                cv2.imwrite(os.path.join(debug_dir, dbg_name), dbg)

    if (
        cfg.OFFRAMP_ENABLE
        and cfg.DATASET_OFFRAMP_LABELS_ENABLE
        and n_offramp_train > 0
    ):
        n_or = int(n_offramp_train)
        n_br = max(1, dw.offramp_num())
        or_pairs = offramp_train_u_rid_pairs_main_spacing(
            dw,
            size,
            margin,
            n_br,
            int(num_train),
            n_or,
            uniform_along_ramp=cfg.DATASET_SAMPLE_UNIFORM_ALONG_ROAD,
        )
        for i, (u, rid) in enumerate(or_pairs):
            ev = dw.offramp_bezier_evolution(u, rid)
            if ev is None:
                continue
            B, (fx, fy), kappa = ev
            px, py = float(B[0]), float(B[1])
            view = get_perspective_view_from_forward(
                world,
                px,
                py,
                fx,
                fy,
                lateral_offset_px=right_lane_offset_px,
                yaw_offset_rad=0.0,
            )
            if view is None:
                continue
            rel_path = f"train/offramp_{i:04d}.jpg"
            out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
            _draw_train_near_vehicle_row_debug_bgr(view)
            cv2.imwrite(out, view)
            records.append((rel_path, float(kappa), 0.0, 0.0, 1))
    # Test: same y grid shape as train (clean then aligned perturbed with matching indices).
    for i, yf in enumerate(test_y):
        road_x = dw.get_road_center(yf)
        dxdY = float(dw.cs(yf, nu=1))
        psi_road = _psi_from_dxdY(dxdY)
        lat_px = lateral_offset_px_avoid_roadkill(
            float(yf),
            right_lane_offset_px,
            for_training_labels=True,
            world_bgr=world,
            dw=dw,
            x_center=float(road_x),
            psi=psi_road,
        )
        view = get_perspective_view(
            world,
            yf,
            road_x,
            dxdY,
            lateral_offset_px=lat_px,
            yaw_offset_rad=0.0,
        )
        if view is None:
            continue
        rel_path = f"test/frame_{i:04d}.jpg"
        out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
        cv2.imwrite(out, view)
        kappa = signed_path_curvature(dw.cs, yf)
        records.append((rel_path, kappa, 0.0, 0.0, 0))

    perturb_test = cfg.DATASET_ALIGNED_PERTURB
    if perturb_test:
        rng_test = np.random.default_rng(
            cfg.DATASET_SEED + cfg.TEST_PERTURB_SEED_OFFSET
        )
        yaw_std = float(np.deg2rad(cfg.PERTURB_YAW_STD_DEG))
        for j, yf in enumerate(test_y):
            road_x = dw.get_road_center(yf)
            dxdY = float(dw.cs(yf, nu=1))
            psi_road = _psi_from_dxdY(dxdY)
            idx = int(num_test) + j
            view, lat_m, yaw_rad = _sample_perturbed_perspective_view(
                world,
                yf,
                road_x,
                dxdY,
                lateral_offset_px_avoid_roadkill(
                    float(yf),
                    right_lane_offset_px,
                    for_training_labels=True,
                    world_bgr=world,
                    dw=dw,
                    x_center=float(road_x),
                    psi=psi_road,
                ),
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
            records.append((rel_path, kappa, lat_m, yaw_rad, 0))

    if (
        cfg.OFFRAMP_ENABLE
        and cfg.DATASET_OFFRAMP_LABELS_ENABLE
        and n_offramp_test > 0
    ):
        n_ot = int(n_offramp_test)
        n_br = max(1, dw.offramp_num())
        u_by_rid: list[np.ndarray] = []
        for rid in range(n_br):
            u_by_rid.append(
                dw.offramp_u_samples_uniform_arc_length(n_ot, ramp_id=rid)
                if cfg.DATASET_SAMPLE_UNIFORM_ALONG_ROAD
                else np.linspace(0.05, 0.95, n_ot, dtype=np.float64)
            )
        for i in range(n_ot):
            rid = i % n_br
            u = float(u_by_rid[rid][i])
            ev = dw.offramp_bezier_evolution(u, rid)
            if ev is None:
                continue
            B, (fx, fy), kappa = ev
            px, py = float(B[0]), float(B[1])
            view = get_perspective_view_from_forward(
                world,
                px,
                py,
                fx,
                fy,
                lateral_offset_px=right_lane_offset_px,
                yaw_offset_rad=0.0,
            )
            if view is None:
                continue
            rel_path = f"test/offramp_{i:04d}.jpg"
            out = os.path.join(cfg.DATA_DIR, rel_path.replace("/", os.sep))
            cv2.imwrite(out, view)
            records.append((rel_path, float(kappa), 0.0, 0.0, 1))

    if records:
        kappas = np.array([k for _, k, _, _, _ in records], dtype=np.float64)
        scale = 1.0 / max(float(np.max(np.abs(kappas))), cfg.KAPPA_SCALE_EPS)
        rows = []
        for path, k, lat_m, yaw_rad, take_or in records:
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
                    "take_offramp": int(take_or),
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
        n_train = sum(1 for p, _, _, _, _ in records if p.startswith("train/"))
        n_test = sum(1 for p, _, _, _, _ in records if p.startswith("test/"))
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
        if cfg.OFFRAMP_ENABLE and cfg.DATASET_OFFRAMP_LABELS_ENABLE:
            nor_tr = sum(
                1
                for p, _, _, _, t in records
                if p.startswith("train/offramp") and int(t) == 1
            )
            nor_te = sum(
                1
                for p, _, _, _, t in records
                if p.startswith("test/offramp") and int(t) == 1
            )
            extra += (
                f"; off-ramp rows (take_offramp=1): "
                f"{nor_tr} train, {nor_te} test"
            )
        print(
            f"Data split complete. ({n_train} train, {n_test} test, labels in {labels_path!r}; "
            f"test previews with steering in data/test_labeled/{extra})"
        )
    else:
        print("Data split complete. (0 frames)")


if __name__ == "__main__":
    generate_data()
