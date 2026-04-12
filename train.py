import json
import os
import shutil
from datetime import datetime

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import config as cfg
from data_loader import DrivingDataset
from generate_world import DrivingWorld
from reproducibility import set_global_seed

set_global_seed(cfg.TRAIN_SEED)

from driving_model import DrivingNet


def _labels_csv_path() -> str:
    """Prefer data/labels.csv; if labels_new.csv is newer (e.g. CSV was locked), use it."""
    a = os.path.join(cfg.DATA_DIR, cfg.LABELS_CSV)
    b = os.path.join(cfg.DATA_DIR, cfg.LABELS_CSV_ALT)
    have_a, have_b = os.path.isfile(a), os.path.isfile(b)
    if have_a and have_b and os.path.getmtime(b) > os.path.getmtime(a):
        return b
    if have_a:
        return a
    if have_b:
        return b
    return a


# 1. Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = cfg.BATCH_SIZE
learning_rate = cfg.LEARNING_RATE
epochs = cfg.EPOCHS

# 2. Image Preprocessing (Standard for PyTorch)
transform = transforms.Compose(
    [
        transforms.Resize((cfg.CAMERA_IMAGE_SIZE, cfg.CAMERA_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.NORMALIZE_MEAN, cfg.NORMALIZE_STD),
    ]
)

# 3. Load Dataset
dataset = DrivingDataset(
    csv_file=_labels_csv_path(),
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
    csv_file=_labels_csv_path(),
    root_dir=cfg.DATA_DIR,
    transform=transform,
    path_prefix="test/",
)
test_loader = (
    DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if len(test_dataset) > 0
    else None
)

# 4. Initialize Model, Loss, and Optimizer
model = DrivingNet().to(device)
criterion = nn.MSELoss()  # Mean Squared Error is standard for steering (regression)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Timestamped run directory (weights + metrics + config snapshot)
_root = os.path.dirname(os.path.abspath(__file__))
_run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(_root, cfg.RUNS_DIR, _run_stamp)
os.makedirs(run_dir, exist_ok=True)

# 5. The Training Loop
print(f"Starting training on {device}...")
print(f"Run artifacts directory: {run_dir!r}")
model.train()

metrics = []

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images).squeeze()
        # Ensure labels and outputs are the same shape
        loss = criterion(outputs[:, 0], labels)  # Only training for steering [0]

        # Backward pass (The "Learning" part)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    test_loss_val = None
    if test_loader is not None:
        model.eval()
        test_sse = 0.0
        test_n = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                batch_loss = criterion(outputs[:, 0], labels)
                test_sse += batch_loss.item() * labels.size(0)
                test_n += labels.size(0)
        test_loss_val = test_sse / test_n if test_n > 0 else 0.0
        model.train()
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss_val:.4f}"
        )
    else:
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
            "Test Loss: n/a (no test split)"
        )

    metrics.append(
        {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "test_loss": float(test_loss_val) if test_loss_val is not None else None,
        }
    )

run_weights = os.path.join(run_dir, cfg.CHECKPOINT_FILENAME)
torch.save(model.state_dict(), run_weights)

data_checkpoint = os.path.join(cfg.DATA_DIR, cfg.CHECKPOINT_FILENAME)
os.makedirs(cfg.DATA_DIR, exist_ok=True)
shutil.copy2(run_weights, data_checkpoint)

training_log = {
    "started_at": _run_stamp,
    "device": str(device),
    "train_samples": len(dataset),
    "test_samples": len(test_dataset),
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "train_seed": cfg.TRAIN_SEED,
    "metrics": metrics,
}
with open(os.path.join(run_dir, "training_log.json"), "w", encoding="utf-8") as f:
    json.dump(training_log, f, indent=2)

with open(os.path.join(run_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
    json.dump(cfg.to_json_snapshot(), f, indent=2)

shutil.copy2(os.path.join(_root, "config.py"), os.path.join(run_dir, "config.py"))

bev_path = os.path.join(run_dir, "world_bev.png")
cv2.imwrite(bev_path, DrivingWorld().image)

print(
    f"Training complete. Weights and logs saved under {run_dir!r} "
    f"(also copied weights to {data_checkpoint!r} for evaluate_test.py). "
    f"Bird's-eye map: {bev_path!r}."
)
