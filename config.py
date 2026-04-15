"""
Central numerical (and related) hyperparameters for the driving sandbox.

Summary:

- **World:** S-curve from ``SPLINE_X_DELTAS_BOTTOM_TO_TOP`` (``generate_world``).
- **Dataset:** By default train = bottom BEV half, test = top half. Roadkill: ``ROADKILL_OBSTACLE_Y_PX`` defines
  supervised detours; ``ROADKILL_EVAL_ONLY_OBSTACLE_Y_PX`` is drawn and used in sim only (generalization).
  ``DATASET_MIX_TRAIN_TEST_GEOGRAPHY`` only shuffles row order within each half (train stays bottom BEV,
  test top). ``DATASET_SEED`` controls that shuffle when enabled. ``DATASET_SAMPLE_UNIFORM_ALONG_ROAD``
  chooses equal spacing along the main road (and along the off-ramp Bézier) vs uniform ``y`` / ``u``. Perturbed duplicate frames:
  ``DATASET_PERTURBATIONS_ENABLE`` plus ``PERTURB_*``. ``NUM_TRAIN_FRAMES`` / ``NUM_TEST_FRAMES`` are clean-grid
  counts; ``TOTAL_TRAIN_FRAMES`` / ``TOTAL_TEST_FRAMES`` are full split sizes after generation.
- **Model input:** ``PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY`` uses only the bottom half of each perspective
  crop (near-ego pixels), resized to ``CAMERA_IMAGE_SIZE``; otherwise the full crop is used.
- **Labels:** Steering is ``kappa / max|kappa|`` over all CSV rows (``generate_dataset``), including the
  optional off-ramp Bézier when ``OFFRAMP_ENABLE`` and ``DATASET_OFFRAMP_LABELS_ENABLE``. Off-ramp row counts
  can match main-road spacing (``DATASET_OFFRAMP_MATCH_MAIN_SPACING``). Column ``take_offramp`` (0 = main road, 1 = ramp) is an extra model input. Lateral/yaw recentering applies
  to perturbed rows when ``DATASET_PERTURBATIONS_ENABLE`` is true and ``PERTURB_*`` σ > 0.
- **Training:** MSE on ``DrivingNet`` channel 0 only with ``take_offramp`` concatenated to the head input; ``EPOCHS`` is in switches; see ``MODEL_USE_TRANSFORMER_HEAD``, ``LEARNING_RATE``.
- **Simulation:** ``MAX_SIM_SPEED`` (near the top) toggles live OpenCV windows and display delays (not MP4 I/O).
  Each ``simulate`` run always writes first-person + bird's-eye MP4s into ``runs/<timestamp>_sim/`` (filenames
  ``SIM_FP_VIDEO_FILENAME``, ``SIM_BEV_VIDEO_FILENAME``). ``SIM_YAW_RATE_GAIN`` from ``_compute_sim_yaw_rate_gain`` aligns
  ``psi += steering * gain * dt`` with curvature step semantics on the train/test ``y`` grid. Off-ramp **merges** (main → Bézier) require
  ``SIM_PROJECT_REF_ONTO_MAIN_ROAD`` and ``OFFRAMP_ENABLE``; at each branch, the sim merges only if
  ``take_offramp`` intent is on—either fixed ``SIM_TAKE_OFFRAMP`` for the whole run, or per-step upper-half
  geographic intent when ``SIM_TAKE_OFFRAMP_UPPER_HALF_NAV`` is true (with optional BEV nav text
  ``SIM_NAV_EXIT_INSTRUCTION_TEXT``).
"""
from __future__ import annotations

import json
import sys
from pickle import TRUE
import types

# --- Paths (relative to project root) ---
DATA_DIR = "data"
RUNS_DIR = "runs"
CHECKPOINT_FILENAME = "driving_net.pt"
# ``latest_checkpoint_path`` (simulate / eval): prefer ``runs/.../driving_net.pt`` unless ``data/`` copy
# is newer by at least this many seconds (avoids sync nudging ``data/`` ahead so sim artifacts land only in ``data/``).
CHECKPOINT_PREFER_DATA_IF_NEWER_BY_SEC = 10.0
LABELS_CSV = "labels.csv"
LABELS_CSV_ALT = "labels_new.csv"
LABELS_TMP = "labels.partial.tmp"

# --- Simulation wall-clock speed (``simulate.py``) ---
# True: no live OpenCV windows, zero ``waitKey`` delay (first-person + BEV MP4s are still written each run).
# False: use the ``_CUSTOM_SIM_*`` values below for realtime windows and delays.
MAX_SIM_SPEED = True

# Applied only when ``MAX_SIM_SPEED`` is False.
_CUSTOM_SIM_REALTIME_BEV = True
_CUSTOM_SIM_REALTIME_DRIVER_VIEW = True
_CUSTOM_SIM_REALTIME_BEV_WAIT_MS = 1
_CUSTOM_SIM_REALTIME_STEP_PAUSE_MS = 0
_CUSTOM_SIM_CUDNN_BENCHMARK = True
_CUSTOM_SIM_TORCH_COMPILE = False

if MAX_SIM_SPEED:
    SIM_REALTIME_BEV = False
    SIM_REALTIME_DRIVER_VIEW = False
    SIM_REALTIME_BEV_WAIT_MS = 0
    SIM_REALTIME_STEP_PAUSE_MS = 0
    SIM_CUDNN_BENCHMARK = True
    SIM_TORCH_COMPILE = False
else:
    SIM_REALTIME_BEV = _CUSTOM_SIM_REALTIME_BEV
    SIM_REALTIME_DRIVER_VIEW = _CUSTOM_SIM_REALTIME_DRIVER_VIEW
    SIM_REALTIME_BEV_WAIT_MS = _CUSTOM_SIM_REALTIME_BEV_WAIT_MS
    SIM_REALTIME_STEP_PAUSE_MS = _CUSTOM_SIM_REALTIME_STEP_PAUSE_MS
    SIM_CUDNN_BENCHMARK = _CUSTOM_SIM_CUDNN_BENCHMARK
    SIM_TORCH_COMPILE = _CUSTOM_SIM_TORCH_COMPILE

# --- Switches (feature toggles; adjust these first) ---
# Perspective: use only bottom half of warp before resize (train / eval / sim).
PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY = False
# Dataset: train rows always bottom BEV half, test always top half; when true, shuffle order within each half.
DATASET_MIX_TRAIN_TEST_GEOGRAPHY = False
# Dataset: if true, clean main-road samples are uniform in **arc length** along the centerline; off-ramp samples
# are uniform in arc length along the Bézier (between the usual ``u`` inset). If false, main uses uniform ``y``;
# ramp uses uniform ``u`` (not arc length).
DATASET_SAMPLE_UNIFORM_ALONG_ROAD = True
# Dataset: add aligned perturbed train/test frames when a perturb σ > 0 (see PERTURB_* below).
DATASET_PERTURBATIONS_ENABLE = True
# Model: true = Transformer head; false = Flatten + linear to MODEL_OUTPUT_DIM.
MODEL_USE_TRANSFORMER_HEAD = True
# Training: number of epochs per ``train.py`` run.
EPOCHS = 20
# Realtime BEV/driver windows: set by ``MAX_SIM_SPEED`` / ``_CUSTOM_SIM_*`` at top (MP4s are always on per sim run).
# World: draw a secondary off-ramp in the bottom half of the BEV; optional dataset labels + κ for it.
OFFRAMP_ENABLE = True
# Off-ramps: one **lane** wide on the map (half the main road´s paved width) and no centerline dashes.
# ``OFFRAMP_SINGLE_LANE_LATERAL_SCALE`` scales ego/camera offset from the ramp centerline vs ``SIM_EGO_LATERAL_OFFSET_M``.
OFFRAMP_SINGLE_LANE = True
OFFRAMP_SINGLE_LANE_LATERAL_SCALE = 0.5
# Simulation: DrivingNet ``take_offramp`` input (1 = ramp intent; 0 = stay on main).
# Ramp **kinematics** (actually merging onto the Bézier) use ``SIM_PROJECT_REF_ONTO_MAIN_ROAD`` and
# ``OFFRAMP_ENABLE``; at each branch, the sim merges only if intent is on (this flag when
# ``SIM_TAKE_OFFRAMP_UPPER_HALF_NAV`` is false, else upper-half geographic intent).
SIM_TAKE_OFFRAMP = False
# When True, ``simulate.run_simulation`` sets ``take_offramp`` from ego image-y each step:
# 1 while ``y <= WORLD_IMAGE_SIZE/2`` (upper half of the BEV), 0 otherwise, draws the nav cue, and **allows**
# a merge onto an off-ramp at that time (e.g. skip a lower-half exit, take exit 117 / ``OFFRAMP_EXTRA_BRANCH_Y_PX``).
# When False, ``take_offramp`` and merge-at-branch are fixed from ``SIM_TAKE_OFFRAMP`` for the whole rollout.
SIM_TAKE_OFFRAMP_UPPER_HALF_NAV = True
# Text shown in the realtime BEV nav box when upper-half intent is active.
SIM_NAV_EXIT_INSTRUCTION_TEXT = "take the next exit"
# Simulation: advance the centerline reference by arc length along the main spline (and off-ramp Bézier when
# a merge occurs). Merge uses ``SIM_TAKE_OFFRAMP`` or upper-half nav intent (see ``SIM_TAKE_OFFRAMP_UPPER_HALF_NAV``).
# False = free BC integration (no ramp path).
SIM_PROJECT_REF_ONTO_MAIN_ROAD = True
# After each step, blend this fraction of the main-road heading into integrated ``psi`` (for kinematics only).
SIM_BLEND_PSI_TO_MAIN_ROAD = 0.22
# ``DATASET_ALIGNED_PERTURB`` is computed later (depends on ``DATASET_PERTURBATIONS_ENABLE`` and ``PERTURB_*`` σ).

# --- Bird's-eye world (generate_world.py) ---
WORLD_IMAGE_SIZE = 1024
WORLD_METERS = 500.0
SPLINE_NUM_CONTROL_POINTS = 11
# X offsets from map center for spline control points, bottom → top along the road.
# Length must match SPLINE_NUM_CONTROL_POINTS. Order: bottom (large y) → top (small y).
# More points + alternating Δ → about twice as many bends as the old 6-point S-curve (stay ~±260 px).
# CubicSpline: dx/dy=0 at bottom (large y). Larger |Δ| = curvier road (stay within ~±260 px of center).
# First two non-center knots from the bottom are 0 so the **lower** part of the road stays straight;
# curvature starts at the next control (``104``).
SPLINE_X_DELTAS_BOTTOM_TO_TOP = (
    0,
    0,
    0,
    104,
    -52,
    100,
    -56,
    96,
    -52,
    82,
    0,
)
# Knot heights: fractions from top (0) to bottom (1) of the map. Uneven steps → bends irregularly
# along the road. Length must match SPLINE_NUM_CONTROL_POINTS; strictly increasing 0..1.
SPLINE_Y_FRACTIONS_TOP_TO_BOTTOM = (
    0.0,
    0.026,
    0.158,
    0.191,
    0.397,
    0.431,
    0.508,
    0.800,
    0.876,
    0.912,
    1.0,
)
ROAD_POLYLINE_SAMPLES = 2000
LANE_WIDTH_METERS = 4.0
DASH_LENGTH_METERS = 3.0
DASH_GAP_METERS = 3.0
WORLD_GREEN_BGR = (34, 139, 34)
WORLD_ROAD_BGR = (40, 40, 40)
ROAD_EDGE_THICKNESS = 1
# Roadkill splat(s) in the **right** lane at BEV row(s) ``y`` (pixels, top=0; traffic → smaller ``y``).
# **Training** (``generate_dataset`` lateral / crops): only ``ROADKILL_OBSTACLE_Y_PX`` — use a ``float`` or ``tuple``.
# **Map + simulation** also draw ``ROADKILL_EVAL_ONLY_OBSTACLE_Y_PX`` and apply that detour in ``simulate`` only
# (model never supervised on those rows; tests generalization).
ROADKILL_ENABLE = True
ROADKILL_OBSTACLE_Y_PX = (850.0,)
ROADKILL_EVAL_ONLY_OBSTACLE_Y_PX = (300.0,)
# Half-height (px) of full-strength detour over the splat; keep modest so most motion stays on eased ramps.
ROADKILL_DETOUR_CORE_HALF_PX = 20.0
# **Ungated / warp-fallback** approach only (sim+dataset use visibility-first when homography succeeds).
# Keep short so ``y``-only labels do not dodge long before the obstacle.
ROADKILL_APPROACH_LEAD_S = 0.0
ROADKILL_DETOUR_BLEND_APPROACH_S = 1.0
# After the splat is visible, blend ``w`` to 1.0 over this many ego car lengths before the core band.
ROADKILL_APPROACH_CORE_MERGE_CAR_LENGTHS = 2.5
# Larger → slower rise of visibility weight as the splat moves through the image (gentler steer in).
ROADKILL_VIS_SPLAT_SOFT_RAMP_FRAC = 1.35
# **Exit:** merge-back span along ``y`` = this × ``SIM_BEV_EGO_CAR_LENGTH_M`` × ``WORLD_IMAGE_SIZE / WORLD_METERS`` (default 4 × 3 m).
ROADKILL_DETOUR_EXIT_CAR_LENGTHS = 4.0
# Exit ramp: raise ``t`` to this power before raised-cosine (1 = symmetric; <1 = hold left longer).
ROADKILL_DETOUR_EXIT_EASE_POWER = 1.0
# If true, only the **approach** ramp is visibility-gated (soft); **exit** uses geometry so the merge
# can finish after the splat leaves the camera (right-lane vantage / same homography as ``perspective_camera_view``).
ROADKILL_DETOUR_ONLY_WHEN_VISIBLE = True
# Ellipse half-axes (px) for splat art; ``generate_world`` clamps so the across-road size stays inside
# one lane (see ``ROADKILL_ACROSS_MAX_FRAC_OF_LANE``).
ROADKILL_ALONG_ROAD_HALF_PX = 14.0
ROADKILL_ACROSS_ROAD_HALF_PX = 3.0
# Max fraction of ``LANE_WIDTH_METERS * px_per_m`` for the splat semi-axis across the road (right lane only).
ROADKILL_ACROSS_MAX_FRAC_OF_LANE = 0.38
# Max multiple of lane width for along-road semi-axis (elongated but not huge).
ROADKILL_ALONG_MAX_LANE_WIDTHS = 2.2
# Off-ramp geometry (see ``OFFRAMP_ENABLE`` in switches): branch on main centerline; image-row fraction
# (0 = top, 1 = bottom). Must be > 0.5 for bottom-half exit; ignored otherwise.
OFFRAMP_BRANCH_Y_FRAC = 0.78
# Additional exits at explicit BEV ``y`` rows (pixels). Default geometry matches ``OFFRAMP_TANGENT_CTRL_PX``
# / ``OFFRAMP_END_*``; use ``OFFRAMP_BRANCH_BEZIER_OVERRIDE_BY_Y_PX`` to customize a branch row.
# Sorted with the primary branch by ``y`` (descending): first encountered when driving from the map bottom.
OFFRAMP_EXTRA_BRANCH_Y_PX = (250.0,)
# Optional per branch row: ``y`` (px) → ``(tangent_ctrl_px, end_dx_px, end_dy_px)`` for a gentler/sharper peel.
# Longer ``tangent_ctrl`` + moderate ``end_dx`` yields a gradual exit toward +x (driver's right in BEV).
OFFRAMP_BRANCH_BEZIER_OVERRIDE_BY_Y_PX = {250.0: (185.0, 175.0, 295.0)}
# Quadratic Bézier from branch: control point lies ``OFFRAMP_TANGENT_CTRL_PX`` along the **main-road tangent**
# (traffic toward decreasing ``y``) so the merge is not a sharp kink; end vertex is branch + (Δx, Δy).
OFFRAMP_TANGENT_CTRL_PX = 95.0
OFFRAMP_END_DX_PX = 210.0
OFFRAMP_END_DY_PX = 265.0
# Main road only: BEV sign placed this many **meters** of centerline arc **before** the first off-ramp
# (travel toward decreasing ``y``). ``0`` disables.
OFFRAMP_ADVANCE_SIGN_ARC_METERS_BEFORE = 50.0
# Extra shift in BEV **+x** (pixels, right on the image) after the shoulder offset.
OFFRAMP_ADVANCE_SIGN_OFFSET_RIGHT_PX = 75.0
OFFRAMP_ADVANCE_SIGN_TEXT = "exit 114"
OFFRAMP_ADVANCE_SIGN_TEXT_SECOND = "150 ft"
# Second off-ramp (next merge toward decreasing ``y`` after the first). ``0`` disables.
OFFRAMP_ADVANCE_SIGN_SECOND_ARC_METERS_BEFORE = 50.0
OFFRAMP_ADVANCE_SIGN_SECOND_OFFSET_RIGHT_PX = 125.0
OFFRAMP_ADVANCE_SIGN_SECOND_TEXT = "exit 117"
OFFRAMP_ADVANCE_SIGN_SECOND_TEXT_SECOND = "150 ft"

# --- Camera / perspective (generate_dataset.py) ---
CAMERA_IMAGE_SIZE = 128
# BEV distance from camera to far edge of the source quad (px). Too small → almost no depth;
# top of the crop does not read as a horizon. ~200–300 is a typical tradeoff vs map bounds.
PERSPECTIVE_FAR_OFFSET_PX = 240.0
# Cross-track half-width of the BEV source quad at the far vs near edge (see ``perspective_camera``).
# The far row must span *more* BEV pixels than the near row: a short far segment stretched to the
# full image width over-magnifies distance (road looks wide at the top); a long near segment
# under-magnifies the ego (road looks narrow at the bottom). So FAR_HALF_WIDTH > NEAR_HALF_WIDTH.
PERSPECTIVE_FAR_HALF_WIDTH = 60.0
PERSPECTIVE_NEAR_HALF_WIDTH = 14.0
# Allow BEV source quad corners this far outside [0,w)×[0,h) before rejecting the warp.
# Near the map edge, a small heading change can spill ~1px past the border; without this,
# open-loop sim often stops after one step. Warp uses replicate padding for out-of-map samples.
PERSPECTIVE_SRC_MARGIN_PX = 12.0

# --- Dataset generation ---
# BEV ``y`` increases downward; geography split vs mix is ``DATASET_MIX_TRAIN_TEST_GEOGRAPHY`` (switches above).
DATASET_MAP_MARGIN = 80
# Clean grid size per split; with aligned perturbations on, total files = 2 × this (half clean, half perturbed).
NUM_TRAIN_FRAMES = 100
NUM_TEST_FRAMES = 100
# Extra train/test images on the off-ramp (``train/offramp_*.jpg``, ``test/offramp_*.jpg``) with
# ``take_offramp=1``; ignored unless ``OFFRAMP_ENABLE`` and ``DATASET_OFFRAMP_LABELS_ENABLE``.
DATASET_OFFRAMP_LABELS_ENABLE = True
# If true, off-ramp clean counts are ``min(cap below, …)`` so average BEV spacing on the ramp matches
# the main-road train/test grid (ramp is shorter than the main segment, so counts drop vs caps).
DATASET_OFFRAMP_MATCH_MAIN_SPACING = True
DATASET_OFFRAMP_TRAIN_FRAMES = NUM_TRAIN_FRAMES
DATASET_OFFRAMP_TEST_FRAMES = NUM_TEST_FRAMES
# Camera lateral (m) = LANE_WIDTH_METERS × fraction: from spline (lane divider) along driver's-right
# toward the outer edge. 0.5 = geometric center of the right lane; lower if the view hugs the outer edge.
DATASET_RIGHT_LANE_LATERAL_FRAC = 0.45
# Gaussian lateral (m) and yaw (deg) for perturbed views; BEV uses ``WORLD_IMAGE_SIZE / WORLD_METERS`` px/m.
# Aligned perturb duplicates require ``DATASET_PERTURBATIONS_ENABLE`` (switches above) and σ > 0 here.
PERTURB_LATERAL_STD_M = 2.2
PERTURB_YAW_STD_DEG = 0.0
# Steering recentering on perturbed rows: −GAIN_LAT·lat_m − GAIN_YAW·yaw_rad (after κ scaling).
TRAIN_PERTURB_RECENTER_GAIN_LAT = 0.35
TRAIN_PERTURB_RECENTER_GAIN_YAW = 2.0
TRAIN_PERTURB_VIEW_RETRIES = 15
# Extra perturbed train frames beyond the 1:1 grid (indices ``2*NUM_TRAIN_FRAMES ..``); ``y`` sampled
# from the train grid with replacement. This does *not* turn off aligned clean/perturbed pairs
# (``NUM_TRAIN_FRAMES`` clean then ``NUM_TRAIN_FRAMES`` perturbed); for clean-only train data set
# ``DATASET_PERTURBATIONS_ENABLE = False`` or both ``PERTURB_*`` σ to 0.
TRAIN_PERTURB_EXTRA_FRAMES = 0
# When true, ``generate_dataset`` adds aligned perturbed mates (and train extras); same gate as
# ``perturb_train`` / ``perturb_test`` in ``generate_dataset.py``.
DATASET_ALIGNED_PERTURB = bool(
    DATASET_PERTURBATIONS_ENABLE
    and (PERTURB_LATERAL_STD_M > 0.0 or PERTURB_YAW_STD_DEG > 0.0)
)
_DATASET_OFFRAMP_TRAIN_N = (
    int(DATASET_OFFRAMP_TRAIN_FRAMES)
    if (OFFRAMP_ENABLE and DATASET_OFFRAMP_LABELS_ENABLE)
    else 0
)
_DATASET_OFFRAMP_TEST_N = (
    int(DATASET_OFFRAMP_TEST_FRAMES)
    if (OFFRAMP_ENABLE and DATASET_OFFRAMP_LABELS_ENABLE)
    else 0
)
# Rows in ``labels.csv`` / image files per split after generation. ``NUM_TRAIN_FRAMES`` and
# ``NUM_TEST_FRAMES`` count **clean** grid positions only. Off-ramp terms use configured caps; with
# ``DATASET_OFFRAMP_MATCH_MAIN_SPACING`` true, ``generate_dataset`` may write fewer off-ramp rows.
TOTAL_TRAIN_FRAMES = int(
    NUM_TRAIN_FRAMES
    + (
        NUM_TRAIN_FRAMES + int(TRAIN_PERTURB_EXTRA_FRAMES)
        if DATASET_ALIGNED_PERTURB
        else 0
    )
    + _DATASET_OFFRAMP_TRAIN_N
)
TOTAL_TEST_FRAMES = int(
    NUM_TEST_FRAMES
    + (NUM_TEST_FRAMES if DATASET_ALIGNED_PERTURB else 0)
    + _DATASET_OFFRAMP_TEST_N
)
# Companion images for perturbed train views (lat/yaw/κ); not listed in labels.csv.
TRAIN_PERTURB_DEBUG_SUBDIR = "train_perturb_debug"
# RNG stream for test perturbation draws (independent of train; same σ).
TEST_PERTURB_SEED_OFFSET = 12345
DATASET_SEED = 42
CURVATURE_DENOM_EPS = 1e-12
KAPPA_SCALE_EPS = 1e-9
STEERING_CLIP_MIN = -1.0
STEERING_CLIP_MAX = 1.0

# CSV save retries (Windows file locks)
LABELS_SAVE_RETRIES = 20
LABELS_SAVE_RETRY_SLEEP_SEC = 0.15

# Overlay text (generate_dataset test_labeled, evaluate_test)
ANNOT_FONT_SCALE = 0.4
ANNOT_FONT_THICKNESS = 1
ANNOT_STEERING_POS = (4, 14)
EVAL_TARGET_TEXT_POS = (4, 14)
EVAL_PRED_TEXT_POS = (4, 28)
EVAL_OUTLINED_FONT_SCALE = 0.4
EVAL_OUTLINED_FONT_THICK = 1

# --- Model architecture (driving_model.py) ---
MODEL_CONV1_CHANNELS = 24
MODEL_CONV2_CHANNELS = 36
MODEL_KERNEL_SIZE = 5
MODEL_STRIDE = 2
MODEL_NUM_CONV_BLOCKS = 2


def _spatial_after_convs(
    image_size: int,
    n_layers: int,
    kernel: int,
    stride: int,
) -> int:
    """Side length of the feature map after ``n_layers`` conv blocks (square assumed)."""
    h = image_size
    for _ in range(n_layers):
        h = (h - kernel) // stride + 1
    return h


_MODEL_SPATIAL = _spatial_after_convs(
    CAMERA_IMAGE_SIZE, MODEL_NUM_CONV_BLOCKS, MODEL_KERNEL_SIZE, MODEL_STRIDE
)
MODEL_FLATTEN_DIM = MODEL_CONV2_CHANNELS * _MODEL_SPATIAL * _MODEL_SPATIAL
# CNN → ``AdaptiveAvgPool2d`` to this side length → ``token_grid²`` sequence tokens for the transformer head.
MODEL_TRANSFORMER_TOKEN_GRID = 7
MODEL_TRANSFORMER_D_MODEL = 128
MODEL_TRANSFORMER_NHEAD = 4
MODEL_TRANSFORMER_NUM_LAYERS = 2
MODEL_TRANSFORMER_FF_DIM = 256
MODEL_TRANSFORMER_DROPOUT = 0.1
# ``train.py`` / ``simulate.py`` / ``evaluate_test.py`` use **channel 0** as steering; channel 1 is unused in the loss.
MODEL_OUTPUT_DIM = 2

# --- Training (train.py) ---
BATCH_SIZE = 16
# Adam: ``1e-3`` often stalls near predicting mean steering; ``3e-4`` (or ``1e-4``) fits this task reliably.
LEARNING_RATE = 3e-4
# ``EPOCHS`` is in switches at top. ``CHECKPOINT_MIN_EPOCH`` (below) delays best-metric checkpoints.
# Used by ``reproducibility.set_global_seed`` and train ``DataLoader`` shuffle generator.
TRAIN_SEED = 42
# First 1..(N-1) epochs are warmup: no best-metric tracking, checkpoints, or best-loss coloring.
CHECKPOINT_MIN_EPOCH = 2
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def _compute_sim_yaw_rate_gain() -> float:
    """
    Match ``generate_dataset`` labels: steering is ``kappa / kappa_max`` over the same train/test
    ``y`` grid used when building ``labels.csv``.

    Step length in px is ``SIM_SPEED_M_S * SIM_DT * (WORLD_IMAGE_SIZE / WORLD_METERS)``. For small
    steps, ``d_psi = kappa * ds``; substituting ``kappa = steering * kappa_max`` and
    ``ds/dt = SIM_SPEED_M_S * px_per_m`` gives
    ``SIM_YAW_RATE_GAIN = kappa_max * SIM_SPEED_M_S * px_per_m`` so that
    ``psi += steering * SIM_YAW_RATE_GAIN * SIM_DT``.

    Curvature matches ``generate_dataset.signed_path_curvature`` (no import of ``generate_dataset``:
    that module loads ``config`` at import time and would recurse while ``SIM_YAW_RATE_GAIN`` is
    still being initialized).
    """
    import numpy as np

    from dataset_split import dataset_train_test_y
    from generate_world import DrivingWorld

    def _kappa_main_road(cs, y: float) -> float:
        d1 = float(cs(y, nu=1))
        d2 = float(cs(y, nu=2))
        denom = (1.0 + d1 * d1) ** 1.5
        if denom < CURVATURE_DENOM_EPS:
            return 0.0
        return d2 / denom

    dw = DrivingWorld()
    size = dw.size
    margin = DATASET_MAP_MARGIN
    train_y, test_y = dataset_train_test_y(
        int(NUM_TRAIN_FRAMES),
        int(NUM_TEST_FRAMES),
        size,
        margin,
        mix_train_test_geography=DATASET_MIX_TRAIN_TEST_GEOGRAPHY,
        seed=DATASET_SEED,
        uniform_along_road=DATASET_SAMPLE_UNIFORM_ALONG_ROAD,
        road_cs=dw.cs,
    )
    kmax = 0.0
    for yf in np.concatenate((train_y, test_y)):
        k = abs(float(_kappa_main_road(dw.cs, float(yf))))
        if k > kmax:
            kmax = k
    if OFFRAMP_ENABLE and DATASET_OFFRAMP_LABELS_ENABLE:
        kmax = max(kmax, float(dw.offramp_max_abs_curvature()))
    px_per_m = float(WORLD_IMAGE_SIZE / WORLD_METERS)
    return float(kmax * SIM_SPEED_M_S * px_per_m)


# --- Open-loop simulator (simulate.py) ---
SIM_SPEED_M_S = 40.0
SIM_DT = 0.05
# Heading rate (rad/s) per unit network output; derived from κ_max and speed (see ``_compute_sim_yaw_rate_gain``).
# Nudge upward slightly if behavioral cloning still under-steers in open loop. Computed **lazily** on first
# read (``__getattr__``) so ``generate_world`` can ``import config`` while this module is still loading.
# Maximum simulation steps per ``simulate.run_simulation`` (hard cap; early exit may stop sooner).
SIM_MAX_STEPS = 2000
# ``SIM_CUDNN_BENCHMARK`` / ``SIM_TORCH_COMPILE``: set by ``MAX_SIM_SPEED`` / ``_CUSTOM_SIM_*`` at top.
# End roll-out when the centerline reference reaches the top drivable band (``y <= DATASET_MAP_MARGIN``).
SIM_STOP_WHEN_REACHES_MAP_TOP = True
# Meters from centerline toward driver's right; must match ``generate_dataset`` camera offset
# (``LANE_WIDTH_METERS * DATASET_RIGHT_LANE_LATERAL_FRAC``). Pixels = this × ``px_per_m`` in sim.
SIM_EGO_LATERAL_OFFSET_M = LANE_WIDTH_METERS * DATASET_RIGHT_LANE_LATERAL_FRAC


def offramp_camera_lateral_offset_px(px_per_m: float) -> float:
    """Camera/ego lateral offset (px) from off-ramp centerline (dataset + sim)."""
    o = float(SIM_EGO_LATERAL_OFFSET_M) * float(px_per_m)
    if OFFRAMP_SINGLE_LANE:
        return float(o * OFFRAMP_SINGLE_LANE_LATERAL_SCALE)
    return o


# Live BEV / ``sim_path.png`` (written under ``runs/<timestamp>_sim/``): top-down ego footprint (meters, full scale on the map).
SIM_BEV_EGO_CAR_WIDTH_M = 1.5
SIM_BEV_EGO_CAR_LENGTH_M = 3.0
# If the true size in pixels is smaller than this length (px), scale the icon up uniformly
# (keeps 1.5 m × 3 m proportions; set0 to always use exact map scale).
SIM_BEV_EGO_CAR_MIN_DISPLAY_LEN_PX = 22.0
# Start as low as possible: try y = h-1, then move up until perspective warp fits.
SIM_START_MAX_INSET_PX = 200
# First-person MP4 written every ``simulate`` run under ``runs/<timestamp>_sim/`` (``CAMERA_IMAGE_SIZE``).
SIM_FP_VIDEO_FILENAME = "sim_first_person.mp4"
# Bird's-eye MP4 every run (``WORLD_IMAGE_SIZE`` square, trail + ego + optional nav box); same FPS as first-person.
SIM_BEV_VIDEO_FILENAME = "sim_bev.mp4"
# Playback speed matches one simulation step per frame (``1 / SIM_DT``).
SIM_VIDEO_FPS = 1.0 / SIM_DT
# ``SIM_REALTIME_BEV`` / ``SIM_REALTIME_DRIVER_VIEW``: set by ``MAX_SIM_SPEED`` / ``_CUSTOM_SIM_*`` at top.
SIM_REALTIME_BEV_WINDOW = "BEV — ego on map"
# Separate window: ego perspective crop (same preprocessing as the network: ``_fp_video_frame_bgr``).
SIM_REALTIME_DRIVER_WINDOW = "Driver — ego camera"
# Initial top-left corner (screen px) for the first live window; second is placed to the right with a gap.
SIM_REALTIME_WINDOW_ORIGIN_X = 40
SIM_REALTIME_WINDOW_ORIGIN_Y = 40
SIM_REALTIME_WINDOW_GAP_PX = 32
# ``SIM_REALTIME_BEV_WAIT_MS`` / ``SIM_REALTIME_STEP_PAUSE_MS``: set by ``MAX_SIM_SPEED`` / ``_CUSTOM_SIM_*`` at top.
# (``waitKey``: ``0`` = block until a key each step. Step pause does not change physics, only display pacing.)
# BEV on-map speed bar (drag): position / ``SIM_REALTIME_SPEED_TRACKBAR_CENTER`` scales arc-length step per
# roll-out iteration (100 = nominal ``SIM_SPEED_M_S * SIM_DT`` in px). Names are legacy from cv2 trackbars.
SIM_REALTIME_SPEED_TRACKBAR_MAX = 200
SIM_REALTIME_SPEED_TRACKBAR_DEFAULT = 100
SIM_REALTIME_SPEED_TRACKBAR_CENTER = 100
SIM_REALTIME_SPEED_SCALE_MIN = 0.05
SIM_REALTIME_SPEED_SCALE_MAX = 2.0

# --- Dummy data (generate_dummy_data.py) ---
DUMMY_NUM_SAMPLES = 100
DUMMY_STEERING_MIN = -1.0
DUMMY_STEERING_MAX = 1.0


def __getattr__(name: str) -> float:
    if name == "SIM_YAW_RATE_GAIN":
        v = float(_compute_sim_yaw_rate_gain())
        globals()["SIM_YAW_RATE_GAIN"] = v
        return v
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def to_json_snapshot() -> dict:
    """Public constants from this module as JSON-serializable dict (for training run logs)."""
    getattr(sys.modules[__name__], "SIM_YAW_RATE_GAIN")
    snap: dict = {}
    for name in sorted(globals()):
        if name.startswith("_"):
            continue
        val = globals()[name]
        if callable(val) or isinstance(val, types.ModuleType):
            continue
        try:
            json.dumps(val)
            snap[name] = val
        except TypeError:
            if isinstance(val, tuple):
                snap[name] = list(val)
            else:
                snap[name] = repr(val)
    return snap
