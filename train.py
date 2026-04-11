import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import DrivingDataset
from driving_model import DrivingNet


def _labels_csv_path() -> str:
    """Prefer data/labels.csv; if labels_new.csv is newer (e.g. CSV was locked), use it."""
    a = os.path.join("data", "labels.csv")
    b = os.path.join("data", "labels_new.csv")
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
batch_size = 16
learning_rate = 0.001
epochs = 100

# 2. Image Preprocessing (Standard for PyTorch)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. Load Dataset
dataset = DrivingDataset(
    csv_file=_labels_csv_path(),
    root_dir="data",
    transform=transform,
    path_prefix="train/",
)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_dataset = DrivingDataset(
    csv_file=_labels_csv_path(),
    root_dir="data",
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
criterion = nn.MSELoss() # Mean Squared Error is standard for steering (regression)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. The Training Loop
print(f"Starting training on {device}...")
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images).squeeze()
        # Ensure labels and outputs are the same shape
        loss = criterion(outputs[:, 0], labels) # Only training for steering [0]
        
        # Backward pass (The "Learning" part)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
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
        test_loss = test_sse / test_n if test_n > 0 else 0.0
        model.train()
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}"
        )
    else:
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
            "Test Loss: n/a (no test split)"
        )

checkpoint_path = os.path.join("data", "driving_net.pt")
torch.save(model.state_dict(), checkpoint_path)
print(f"Training complete. Weights saved to {checkpoint_path!r}")