"""
Train ``DrivingNet`` on ``data/labels.csv`` (MSE on output **channel 0** vs steering targets, with ``take_offramp`` input).

Uses rows with ``train/`` prefix for training and ``test/`` for validation (geographic or mixed road
sampling per ``generate_dataset`` / ``DATASET_MIX_TRAIN_TEST_GEOGRAPHY``). Checkpoints on best test
loss after ``CHECKPOINT_MIN_EPOCH`` (or best train loss if no test rows). Hyperparameters in ``config``.
"""
import json
import math
import os
import shutil
import time
from datetime import datetime

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import config as cfg
from data_loader import (
    DrivingDataset,
    prepare_perspective_pil_for_model,
    resolve_labels_csv_path,
)
from generate_world import DrivingWorld
from reproducibility import set_global_seed

set_global_seed(cfg.TRAIN_SEED)

from driving_model import DrivingNet
from run_architecture import write_architecture_artifacts


def _write_latest_run_pointer(repo_root: str, run_dir: str, status: str) -> None:
    """``runs/LATEST_RUN.txt``: absolute run path + ``in_progress`` / ``complete`` (gitignored with ``runs/``)."""
    p = os.path.join(repo_root, cfg.RUNS_DIR, "LATEST_RUN.txt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(run_dir) + "\n")
        f.write(status + "\n")

_RED = "\033[31m"
_GRN = "\033[32m"
_RST = "\033[0m"


def _clone_state_dict_cpu(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _fmt_loss_colored(value: float, *, color: str) -> str:
    """Format `value` with optional ANSI color (no extra label text)."""
    return f"{color}{value:.4f}{_RST}"


# 1. Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = cfg.BATCH_SIZE
learning_rate = cfg.LEARNING_RATE
epochs = cfg.EPOCHS

# 2. Image preprocessing (optional bottom-half crop near ego, then square resize — see ``config``)
transform = transforms.Compose(
    [
        transforms.Lambda(prepare_perspective_pil_for_model),
        transforms.Resize((cfg.CAMERA_IMAGE_SIZE, cfg.CAMERA_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.NORMALIZE_MEAN, cfg.NORMALIZE_STD),
    ]
)

# 3. Load Dataset
dataset = DrivingDataset(
    csv_file=resolve_labels_csv_path(),
    root_dir=cfg.DATA_DIR,
    transform=transform,
    path_prefix="train/",
)
_train_gen = torch.Generator().manual_seed(cfg.TRAIN_SEED)
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=_train_gen,
)

test_dataset = DrivingDataset(
    csv_file=resolve_labels_csv_path(),
    root_dir=cfg.DATA_DIR,
    transform=transform,
    path_prefix="test/",
)
test_loader = (
    DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if len(test_dataset) > 0
    else None
)

# 4. Initialize Model, Loss, and Optimizer (MSE on steering head only: outputs[:, 0]; intent concat)
model = DrivingNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Timestamped run directory (weights + metrics + config snapshot)
_root = os.path.dirname(os.path.abspath(__file__))
_run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(_root, cfg.RUNS_DIR, _run_stamp)
os.makedirs(run_dir, exist_ok=True)
_write_latest_run_pointer(_root, run_dir, "in_progress")
write_architecture_artifacts(run_dir, model)
_onnx_p = os.path.join(run_dir, "driving_net.onnx")
print(
    f"Wrote architecture: {os.path.join(run_dir, 'architecture.md')!r}, "
    f"{os.path.join(run_dir, 'architecture.png')!r}, "
    f"Netron ONNX {_onnx_p!r} (if export succeeded; else see onnx_export_error.txt)"
)

# 5. The Training Loop
print(
    f"Dataset: {len(dataset)} train examples, {len(test_dataset)} test examples "
    f"(from {resolve_labels_csv_path()!r})"
)
print(f"Starting training on {device}...")
print(f"Run artifacts directory: {os.path.abspath(run_dir)!r}")
print(
    f"Latest-run pointer: {os.path.join(_root, cfg.RUNS_DIR, 'LATEST_RUN.txt')!r} "
    "(path + status; ends with 'complete' when training finishes)"
)
if test_loader is not None:
    print(
        f"Warmup: epochs 1..{cfg.CHECKPOINT_MIN_EPOCH - 1} skip best-test tracking; "
        f"checkpoints when test improves (epoch >= {cfg.CHECKPOINT_MIN_EPOCH})."
    )
else:
    print(
        f"Warmup: epochs 1..{cfg.CHECKPOINT_MIN_EPOCH - 1} skip best-train tracking; "
        f"checkpoints when train improves (epoch >= {cfg.CHECKPOINT_MIN_EPOCH})."
    )
model.train()

metrics = []
best_state: dict | None = None
best_epoch: int | None = None
best_test_loss_tracked = float("inf")
lowest_train_loss = float("inf")
running_min_test_loss = float("inf")
running_min_test_epoch: int | None = None
checkpoint_fallback_last_epoch = False

_train_wall_start = time.perf_counter()
for epoch in range(epochs):
    train_sse = 0.0
    train_n = 0
    for images, labels, take_or in train_loader:
        images, labels = images.to(device), labels.to(device)
        take_or = take_or.to(device)

        # Forward pass (no .squeeze(): batch size 1 would drop the batch dim and break [:, 0])
        outputs = model(images, take_or)
        loss = criterion(outputs[:, 0], labels)  # steering channel 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_sse += loss.item() * labels.size(0)
        train_n += labels.size(0)

    train_loss = train_sse / train_n if train_n > 0 else 0.0
    epoch_1based = epoch + 1
    past_warmup = epoch_1based >= cfg.CHECKPOINT_MIN_EPOCH
    lowest_train_loss = min(lowest_train_loss, train_loss)
    train_loss_is_best = math.isclose(train_loss, lowest_train_loss, rel_tol=0.0, abs_tol=1e-9)
    train_loss_str = (
        _fmt_loss_colored(train_loss, color=_RED)
        if train_loss_is_best and past_warmup
        else f"{train_loss:.4f}"
    )

    test_loss_val = None
    improved_checkpoint = False
    if test_loader is not None:
        model.eval()
        test_sse = 0.0
        test_n = 0
        with torch.no_grad():
            for images, labels, take_or in test_loader:
                images, labels = images.to(device), labels.to(device)
                take_or = take_or.to(device)
                outputs = model(images, take_or)
                batch_loss = criterion(outputs[:, 0], labels)
                test_sse += batch_loss.item() * labels.size(0)
                test_n += labels.size(0)
        test_loss_val = test_sse / test_n if test_n > 0 else 0.0
        model.train()
        if test_loss_val < running_min_test_loss:
            running_min_test_loss = test_loss_val
            running_min_test_epoch = epoch_1based
        test_new_best = False
        if past_warmup:
            prev_best_test = best_test_loss_tracked
            test_new_best = test_loss_val < prev_best_test
            if test_new_best:
                best_test_loss_tracked = test_loss_val
                best_epoch = epoch_1based
        improved_checkpoint = test_new_best
        if improved_checkpoint:
            best_state = _clone_state_dict_cpu(model)
        test_loss_str = (
            _fmt_loss_colored(test_loss_val, color=_GRN)
            if test_new_best
            else f"{test_loss_val:.4f}"
        )
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_str}, "
            f"Test Loss: {test_loss_str}"
            + ("  [new best test -> checkpoint]" if improved_checkpoint else "")
        )
    else:
        train_new_best = False
        if past_warmup:
            prev_best_train = best_test_loss_tracked
            train_new_best = train_loss < prev_best_train
            if train_new_best:
                best_test_loss_tracked = train_loss
                best_epoch = epoch_1based
        improved_checkpoint = train_new_best
        if improved_checkpoint:
            best_state = _clone_state_dict_cpu(model)
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_str}, "
            "Test Loss: n/a (no test split)"
            + ("  [new best train -> checkpoint]" if improved_checkpoint else "")
        )

    row = {
        "epoch": epoch + 1,
        "train_loss": float(train_loss),
        "test_loss": float(test_loss_val) if test_loss_val is not None else None,
        "lowest_train_loss_so_far": float(lowest_train_loss),
        "is_best_checkpoint": improved_checkpoint,
    }
    if test_loader is not None:
        row["lowest_test_loss_so_far"] = float(running_min_test_loss)
        row["epoch_with_lowest_test_so_far"] = running_min_test_epoch
    metrics.append(row)

_train_wall_sec = time.perf_counter() - _train_wall_start
_per_epoch_sec = _train_wall_sec / float(epochs) if epochs > 0 else 0.0
print(
    f"Training time: {_train_wall_sec:.2f}s ({_train_wall_sec / 60.0:.2f} min); "
    f"{_per_epoch_sec:.2f}s per epoch ({epochs} epochs)."
)

run_weights = os.path.join(run_dir, cfg.CHECKPOINT_FILENAME)
data_checkpoint = os.path.join(cfg.DATA_DIR, cfg.CHECKPOINT_FILENAME)
os.makedirs(cfg.DATA_DIR, exist_ok=True)
if best_state is None:
    checkpoint_fallback_last_epoch = True
    best_state = _clone_state_dict_cpu(model)
    print(
        f"Warning: no best-metric checkpoint with epoch >= {cfg.CHECKPOINT_MIN_EPOCH}; "
        "saving last epoch weights."
    )
torch.save(best_state, run_weights)
shutil.copy2(run_weights, data_checkpoint)

_report_metric = float(best_test_loss_tracked)
if (
    test_loader is not None
    and not math.isfinite(_report_metric)
    and math.isfinite(running_min_test_loss)
):
    _report_metric = float(running_min_test_loss)

training_log = {
    "started_at": _run_stamp,
    "device": str(device),
    "train_samples": len(dataset),
    "test_samples": len(test_dataset),
    "epochs": epochs,
    "train_wall_time_sec": float(_train_wall_sec),
    "train_time_per_epoch_sec": float(_per_epoch_sec),
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "train_seed": cfg.TRAIN_SEED,
    "checkpoint_min_epoch": cfg.CHECKPOINT_MIN_EPOCH,
    "best_checkpoint_epoch": best_epoch,
    "best_metric_at_checkpoint": _report_metric,
    "checkpoint_criterion": "min_test_loss" if test_loader is not None else "min_train_loss",
    "checkpoint_fallback_last_epoch": checkpoint_fallback_last_epoch,
    "metrics": metrics,
}
with open(os.path.join(run_dir, "training_log.json"), "w", encoding="utf-8") as f:
    json.dump(training_log, f, indent=2)

with open(os.path.join(run_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
    json.dump(cfg.to_json_snapshot(), f, indent=2)

shutil.copy2(os.path.join(_root, "config.py"), os.path.join(run_dir, "config.py"))

bev_path = os.path.join(run_dir, "world_bev.png")
cv2.imwrite(bev_path, DrivingWorld().image)

_write_latest_run_pointer(_root, run_dir, "complete")

_crit = "test" if test_loader is not None else "train"
if checkpoint_fallback_last_epoch:
    print(
        f"Training complete. Last-epoch weights (epoch {epochs}) saved under {os.path.abspath(run_dir)!r} "
        f"and {os.path.abspath(data_checkpoint)!r} (no improvement with epoch >= {cfg.CHECKPOINT_MIN_EPOCH}). "
        f"Best {_crit} metric seen: {_report_metric:.4f} at epoch {best_epoch}. "
        f"Bird's-eye map: {bev_path!r}."
    )
else:
    print(
        f"Training complete. Best-by-{_crit} weights from epoch {best_epoch} "
        f"(metric={_report_metric:.4f}) saved under {os.path.abspath(run_dir)!r} "
        f"and {os.path.abspath(data_checkpoint)!r} for evaluate_test.py. "
        f"Bird's-eye map: {bev_path!r}."
    )
