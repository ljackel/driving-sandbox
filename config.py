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
SPLINE_X_DELTAS_BOTTOM_TO_TOP = (0, 150, -100, 50, -200, 0)
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
# Meters from centerline to right-lane center (lane width is LANE_WIDTH_METERS).
RIGHT_LANE_OFFSET_METERS = 2.0

# --- Dataset generation ---
DATASET_MAP_MARGIN = 80
TEST_Y_STEP = 10
NUM_TRAIN_FRAMES = 1000
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
    h = image_size
    for _ in range(n_layers):
        h = (h - kernel) // stride + 1
    return h


_MODEL_SPATIAL = _spatial_after_convs(
    CAMERA_IMAGE_SIZE, MODEL_NUM_CONV_BLOCKS, MODEL_KERNEL_SIZE, MODEL_STRIDE
)
MODEL_FLATTEN_DIM = MODEL_CONV2_CHANNELS * _MODEL_SPATIAL * _MODEL_SPATIAL
MODEL_OUTPUT_DIM = 2

# --- Training (train.py) ---
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 100
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# --- Open-loop simulator (simulate.py) ---
SIM_SPEED_M_S = 20.0
SIM_DT = 0.05
# Maps network steering [-1, 1] to heading rate (rad/s); tune for stable turns.
SIM_YAW_RATE_GAIN = 2.0
SIM_MAX_STEPS = 200_000
# Lateral offset from centerline in pixels (+ = driver right in image coords).
SIM_EGO_LATERAL_OFFSET_PX = 0.0

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
