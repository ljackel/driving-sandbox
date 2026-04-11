import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg

# 1. Setup Device (Single-Stream)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
else:
    print(f"Using device: {device}")

# 2. Define a Simple Driving Model (CNN based)
class DrivingNet(nn.Module):
    def __init__(self):
        super(DrivingNet, self).__init__()
        k = cfg.MODEL_KERNEL_SIZE
        s = cfg.MODEL_STRIDE
        self.features = nn.Sequential(
            nn.Conv2d(3, cfg.MODEL_CONV1_CHANNELS, kernel_size=k, stride=s),
            nn.ReLU(),
            nn.Conv2d(
                cfg.MODEL_CONV1_CHANNELS,
                cfg.MODEL_CONV2_CHANNELS,
                kernel_size=k,
                stride=s,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.controller = nn.Linear(
            cfg.MODEL_FLATTEN_DIM, cfg.MODEL_OUTPUT_DIM
        )

    def forward(self, x):
        x = self.features(x)
        return self.controller(x)

# 3. Initialize Model and Move to GPU
model = DrivingNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
criterion = nn.MSELoss()

print("Model is ready for single-stream training.")
