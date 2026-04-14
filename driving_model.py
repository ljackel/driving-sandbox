import torch
import torch.nn as nn

import config as cfg


class DrivingNet(nn.Module):
    """
    CNN backbone, then either a **Transformer** encoder over spatial tokens or a **two-hidden-layer MLP**
    (controlled by ``MODEL_USE_TRANSFORMER_HEAD``).

    **Steering** is **channel 0**; training and simulation use only that channel (channel 1 is unused).
    """

    def __init__(self):
        super().__init__()

        self.use_transformer = cfg.MODEL_USE_TRANSFORMER_HEAD

        k = cfg.MODEL_KERNEL_SIZE
        s = cfg.MODEL_STRIDE

        self.backbone = nn.Sequential(
            nn.Conv2d(3, cfg.MODEL_CONV1_CHANNELS, kernel_size=k, stride=s),
            nn.ReLU(),
            nn.Conv2d(
                cfg.MODEL_CONV1_CHANNELS,
                cfg.MODEL_CONV2_CHANNELS,
                kernel_size=k,
                stride=s,
            ),
            nn.ReLU(),
        )

        if self.use_transformer:
            p = int(cfg.MODEL_TRANSFORMER_TOKEN_GRID)
            self.token_grid = p
            self.num_tokens = p * p
            self.pool = nn.AdaptiveAvgPool2d((p, p))

            c_in = cfg.MODEL_CONV2_CHANNELS
            d = cfg.MODEL_TRANSFORMER_D_MODEL
            self.input_proj = nn.Linear(c_in, d)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=cfg.MODEL_TRANSFORMER_NHEAD,
                dim_feedforward=cfg.MODEL_TRANSFORMER_FF_DIM,
                dropout=cfg.MODEL_TRANSFORMER_DROPOUT,
                activation="relu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                enc_layer,
                num_layers=cfg.MODEL_TRANSFORMER_NUM_LAYERS,
            )
            self.head = nn.Linear(d, cfg.MODEL_OUTPUT_DIM)
            self.mlp_head = None
        else:
            self.token_grid = 0
            self.num_tokens = 0
            self.pool = None
            self.input_proj = None
            self.pos_embed = None
            self.encoder = None
            self.head = None
            h = cfg.MODEL_FC_HIDDEN_DIM
            self.mlp_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(cfg.MODEL_FLATTEN_DIM, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, cfg.MODEL_OUTPUT_DIM),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the network on a batch of NCHW tensors.

        Args:
            x: Input images shaped ``(N, 3, H, W)`` with ``H, W`` matching ``CAMERA_IMAGE_SIZE``.

        Returns:
            Tensor of shape ``(N, MODEL_OUTPUT_DIM)``; steering for loss/sim is ``[..., 0]``.
        """
        x = self.backbone(x)
        if self.use_transformer:
            x = self.pool(x)
            b, c, _, _ = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.input_proj(x)
            x = x + self.pos_embed
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.head(x)
        return self.mlp_head(x)
