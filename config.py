"""
Central numerical (and related) hyperparameters for the driving sandbox.
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
# Gentle S: small +/− x offsets (px from map center); first value at bottom stays0 for straight start.
# CubicSpline: dx/dy=0 at bottom (large y). Sharper example: (0, 150, -100, 50, -200, 0)
SPLINE_X_DELTAS_BOTTOM_TO_TOP = (0, 22, -18, 14, -10, 0)
ROAD_POLYLINE_SAMPLES = 2000
LANE_WIDTH_METERS = 4.0
DASH_LENGTH_METERS = 3.0
DASH_GAP_METERS = 3.0
WORLD_GREEN_BGR = (34, 139, 34)
WORLD_ROAD_BGR = (40, 40, 40)
ROAD_EDGE_THICKNESS = 1

# --- Camera / perspective (generate_dataset.py) ---
CAMERA_IMAGE_SIZE = 128
PERSPECTIVE_FAR_OFFSET_PX = 20.0
PERSPECTIVE_FAR_HALF_WIDTH = 10.0
PERSPECTIVE_NEAR_HALF_WIDTH = 60.0
# Allow BEV source quad corners this far outside [0,w)×[0,h) before rejecting the warp.
# Near the map edge, a small heading change can spill ~1px past the border; without this,
# open-loop sim often stops after one step. Warp uses replicate padding for out-of-map samples.
PERSPECTIVE_SRC_MARGIN_PX = 12.0

# --- Dataset generation ---
DATASET_MAP_MARGIN = 80
TEST_Y_STEP = 10
# Samples along y on the map. Train images = NUM_TRAIN_FRAMES clean + NUM_TRAIN_FRAMES perturbed
# when any perturbation σ > 0 below (same y grid, different lateral/yaw).
NUM_TRAIN_FRAMES = 1000
# Camera lateral (m) = LANE_WIDTH_METERS × fraction: from spline (lane divider) along driver's-right
# toward the outer edge. 0.5 = geometric center of the right lane; lower if the view hugs the outer edge.
DATASET_RIGHT_LANE_LATERAL_FRAC = 0.45
# Road alignment: generate_dataset uses yaw_offset_rad=0 so forward axis matches the path tangent.
# Second half of train set (when σ > 0): extra samples with Gaussian lateral (m, along lane-right)
# and optional yaw noise (deg). Labels use κ·scale minus recentering terms below.
TRAIN_PERTURB_LATERAL_STD_M = 0.25
TRAIN_PERTURB_YAW_STD_DEG = 0.0
# Added to steering for perturbed rows: −GAIN_LAT·lat_m − GAIN_YAW·yaw_rad (same scale as κ after scaling).
TRAIN_PERTURB_RECENTER_GAIN_LAT = 0.35
TRAIN_PERTURB_RECENTER_GAIN_YAW = 2.0
TRAIN_PERTURB_VIEW_RETRIES = 30
# Companion images for perturbed train views (lat/yaw/κ on image); not listed in labels.csv.
TRAIN_PERTURB_DEBUG_SUBDIR = "train_perturb_debug"
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
MODEL_OUTPUT_DIM = 2

# --- Training (train.py) ---
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 100
# Used by ``reproducibility.set_global_seed`` and train ``DataLoader`` shuffle generator.
TRAIN_SEED = 42
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# --- Open-loop simulator (simulate.py) ---
SIM_SPEED_M_S = 20.0
SIM_DT = 0.05
# Maps network steering [-1, 1] to heading rate (rad/s); tune for stable turns.
SIM_YAW_RATE_GAIN = 2.0
SIM_MAX_STEPS = 200_000
# Meters from centerline toward driver's right; must match ``generate_dataset`` camera offset
# (``LANE_WIDTH_METERS * DATASET_RIGHT_LANE_LATERAL_FRAC``). Pixels = this × ``px_per_m`` in sim.
SIM_EGO_LATERAL_OFFSET_M = LANE_WIDTH_METERS * DATASET_RIGHT_LANE_LATERAL_FRAC
# Start as low as possible: try y = h-1, then move up until perspective warp fits.
SIM_START_MAX_INSET_PX = 200

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
