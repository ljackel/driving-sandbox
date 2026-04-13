"""
Central numerical (and related) hyperparameters for the driving sandbox.

Summary:

- **World:** S-curve from ``SPLINE_X_DELTAS_BOTTOM_TO_TOP`` (``generate_world``).
- **Dataset:** Train = bottom BEV half (large ``y``); test = top half (small ``y``). See ``NUM_TRAIN_FRAMES``, ``NUM_TEST_FRAMES``, perturb settings.
- **Model input:** ``PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY`` uses only the bottom half of each perspective
  crop (near-ego pixels), resized to ``CAMERA_IMAGE_SIZE``; otherwise the full crop is used.
- **Labels:** Steering is ``kappa / max|kappa|`` over all CSV rows, minus lateral/yaw recentering on
  perturbed rows (``generate_dataset``).
- **Training:** MSE on ``DrivingNet`` channel 0 only; see ``LEARNING_RATE``, ``EPOCHS``.
- **Simulation:** ``SIM_YAW_RATE_GAIN`` from ``_compute_sim_yaw_rate_gain`` aligns ``psi += steering * gain * dt``
  with curvature step semantics on the train/test ``y`` grid. Optional first-person MP4: ``SIM_FP_VIDEO_*``.
"""
from __future__ import annotations

import json
import types

# --- Paths (relative to project root) ---
DATA_DIR = "data"
RUNS_DIR = "runs"
CHECKPOINT_FILENAME = "driving_net.pt"
LABELS_CSV = "labels.csv"
LABELS_CSV_ALT = "labels_new.csv"
LABELS_TMP = "labels.partial.tmp"

# --- Bird's-eye world (generate_world.py) ---
WORLD_IMAGE_SIZE = 1024
WORLD_METERS = 500.0
SPLINE_NUM_CONTROL_POINTS = 6
# X offsets from map center for spline control points, bottom → top along the road.
# Length must match SPLINE_NUM_CONTROL_POINTS. Order: bottom (large y) → top (small y).
# S-curve strength: x offsets (px from map center), bottom → top; first value 0 = straight start.
# CubicSpline: dx/dy=0 at bottom (large y). Larger |Δ| = curvier road (stay within ~±260 px of center).
SPLINE_X_DELTAS_BOTTOM_TO_TOP = (0, 116, -104, 84, -76, 0)
ROAD_POLYLINE_SAMPLES = 2000
LANE_WIDTH_METERS = 4.0
DASH_LENGTH_METERS = 3.0
DASH_GAP_METERS = 3.0
WORLD_GREEN_BGR = (34, 139, 34)
WORLD_ROAD_BGR = (40, 40, 40)
ROAD_EDGE_THICKNESS = 1

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
# If true: for ``train`` / ``evaluate_test`` / ``simulate``, crop each perspective image to its
# bottom half (nearest the vehicle), then resize to ``CAMERA_IMAGE_SIZE``. If false: use the full square.
PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY = True

# --- Dataset generation ---
# BEV ``y`` increases downward. Train samples the bottom half (large ``y``); test the top half
# (small ``y``), disjoint at the midline. With perturbations on: each split gets one clean + one
# aligned perturbed frame per sampled ``y`` (``PERTURB_*``). ``TRAIN_PERTURB_EXTRA_FRAMES`` adds
# extra perturbed train frames at random train ``y``; test perturb RNG uses ``TEST_PERTURB_SEED_OFFSET``.
DATASET_MAP_MARGIN = 80
NUM_TRAIN_FRAMES = 1000
NUM_TEST_FRAMES = 100
# Camera lateral (m) = LANE_WIDTH_METERS × fraction: from spline (lane divider) along driver's-right
# toward the outer edge. 0.5 = geometric center of the right lane; lower if the view hugs the outer edge.
DATASET_RIGHT_LANE_LATERAL_FRAC = 0.45
# Road alignment: generate_dataset uses yaw_offset_rad=0 so forward axis matches the path tangent.
# Single Gaussian lateral (m) and yaw (deg) for all perturbed train/test views; set yaw σ to 0 for lateral-only.
# Lateral is lat_m * (WORLD_IMAGE_SIZE / WORLD_METERS) BEV px/m (~2.05).
PERTURB_LATERAL_STD_M = 2.2
PERTURB_YAW_STD_DEG = 0.0
# Legacy aliases (same values as PERTURB_*).
TRAIN_PERTURB_LATERAL_STD_M = PERTURB_LATERAL_STD_M
TRAIN_PERTURB_YAW_STD_DEG = PERTURB_YAW_STD_DEG
TEST_PERTURB_LATERAL_STD_M = PERTURB_LATERAL_STD_M
TEST_PERTURB_YAW_STD_DEG = PERTURB_YAW_STD_DEG
# Added to steering for perturbed rows: −GAIN_LAT·lat_m − GAIN_YAW·yaw_rad (same scale as κ after scaling).
TRAIN_PERTURB_RECENTER_GAIN_LAT = 0.35
TRAIN_PERTURB_RECENTER_GAIN_YAW = 2.0
TRAIN_PERTURB_VIEW_RETRIES = 30
# Extra perturbed frames: same Gaussian lateral/yaw and backoff as aligned perturbed; ``y`` drawn
# uniformly at random from the train ``y`` grid (with replacement). 0 = only NUM_TRAIN aligned.
TRAIN_PERTURB_EXTRA_FRAMES = 4000
# Companion images for perturbed train views (lat/yaw/κ on image); not listed in labels.csv.
TRAIN_PERTURB_DEBUG_SUBDIR = "train_perturb_debug"
# RNG stream for test perturbation draws (same σ as train; independent noise).
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
# Width of both fully connected hidden layers after the conv stack.
MODEL_FC_HIDDEN_DIM = 1000
# ``train.py`` / ``simulate.py`` / ``evaluate_test.py`` use **channel 0** as steering; channel 1 is unused in the loss.
MODEL_OUTPUT_DIM = 2

# --- Training (train.py) ---
BATCH_SIZE = 16
# Adam: ``1e-3`` often stalls near predicting mean steering; ``3e-4`` (or ``1e-4``) fits this task reliably.
LEARNING_RATE = 3e-4
# Increase when the dataset grows (e.g. many extra perturb frames); ``CHECKPOINT_MIN_EPOCH`` delays best-metric checkpoints.
EPOCHS = 200
# Used by ``reproducibility.set_global_seed`` and train ``DataLoader`` shuffle generator.
TRAIN_SEED = 42
# First 1..(N-1) epochs are warmup: no best-metric tracking, checkpoints, or best-loss coloring.
CHECKPOINT_MIN_EPOCH = 11
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
    """
    import numpy as np

    from generate_dataset import signed_path_curvature
    from generate_world import DrivingWorld

    dw = DrivingWorld()
    size = dw.size
    margin = DATASET_MAP_MARGIN
    half = size // 2
    train_y = np.linspace(
        float(size - margin),
        float(half) + 1.0,
        int(NUM_TRAIN_FRAMES),
        dtype=np.float64,
    )
    test_y = np.linspace(
        float(half),
        float(margin),
        int(NUM_TEST_FRAMES),
        dtype=np.float64,
    )
    kmax = 0.0
    for yf in np.concatenate((train_y, test_y)):
        k = abs(float(signed_path_curvature(dw.cs, float(yf))))
        if k > kmax:
            kmax = k
    px_per_m = float(WORLD_IMAGE_SIZE / WORLD_METERS)
    return float(kmax * SIM_SPEED_M_S * px_per_m)


# --- Open-loop simulator (simulate.py) ---
SIM_SPEED_M_S = 20.0
SIM_DT = 0.05
# Heading rate (rad/s) per unit network output; derived from κ_max and speed (see ``_compute_sim_yaw_rate_gain``).
# Nudge upward slightly if behavioral cloning still under-steers in open loop.
SIM_YAW_RATE_GAIN = _compute_sim_yaw_rate_gain()
SIM_MAX_STEPS = 200_000
# Meters from centerline toward driver's right; must match ``generate_dataset`` camera offset
# (``LANE_WIDTH_METERS * DATASET_RIGHT_LANE_LATERAL_FRAC``). Pixels = this × ``px_per_m`` in sim.
SIM_EGO_LATERAL_OFFSET_M = LANE_WIDTH_METERS * DATASET_RIGHT_LANE_LATERAL_FRAC
# Start as low as possible: try y = h-1, then move up until perspective warp fits.
SIM_START_MAX_INSET_PX = 200
# First-person video from ``simulate.run_simulation`` (same resolution as ``CAMERA_IMAGE_SIZE``).
SIM_FP_VIDEO_ENABLE = True
SIM_FP_VIDEO_FILENAME = "sim_first_person.mp4"
# Playback speed matches one simulation step per frame (``1 / SIM_DT``).
SIM_VIDEO_FPS = 1.0 / SIM_DT

# --- Dummy data (generate_dummy_data.py) ---
DUMMY_NUM_SAMPLES = 100
DUMMY_STEERING_MIN = -1.0
DUMMY_STEERING_MAX = 1.0


def to_json_snapshot() -> dict:
    """Public constants from this module as JSON-serializable dict (for training run logs)."""
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
