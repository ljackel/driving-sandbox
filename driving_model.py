import torch
import torch.nn as nn
import torch.optim as optim

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
        # Simplified: takes an image, outputs steering/throttle
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        # 128×128 → 62×62 → 29×29 spatial after two 5×5 stride-2 convs; 36 channels
        self.controller = nn.Linear(36 * 29 * 29, 2)  # [Steering, Throttle]

    def forward(self, x):
        x = self.features(x)
        return self.controller(x)

# 3. Initialize Model and Move to GPU
model = DrivingNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Model is ready for single-stream training.")
print("This is wonderful")
