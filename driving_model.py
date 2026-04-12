import torch
import torch.nn as nn

import config as cfg


class DrivingNet(nn.Module):
    """CNN + two-layer MLP: normalized RGB crops → 2-D vector (steering uses index 0)."""

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

        self.head = nn.Sequential(
            nn.Linear(cfg.MODEL_FLATTEN_DIM, cfg.MODEL_FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.MODEL_FC_HIDDEN_DIM, cfg.MODEL_OUTPUT_DIM),
        )

    def forward(self, x):
        """
        Run the network on a batch of NCHW tensors.

        Args:
            x: Input images shaped ``(N, 3, H, W)`` with ``H, W`` matching ``CAMERA_IMAGE_SIZE``.

        Returns:
            Tensor of shape ``(N, MODEL_OUTPUT_DIM)``.
        """
        x = self.features(x)
        return self.head(x)
