"""
Write architecture diagram + parameter counts into a training ``runs/<stamp>/`` directory.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from driving_model import DrivingNet


class _DrivingNetOnnxFriendly(nn.Module):
    """
    Same parameters as a transformer ``DrivingNet``; ``forward`` matches training except
    ``AdaptiveAvgPool2d`` is replaced by bilinear ``interpolate`` to ``(p, p)`` so the legacy
    ONNX exporter accepts the graph (Netron visualization only—pooling math differs slightly).
    """

    def __init__(self, src: DrivingNet) -> None:
        super().__init__()
        if not src.use_transformer:
            raise ValueError("Use DrivingNet directly when not using the transformer head.")
        self.backbone = src.backbone
        self.p = int(src.token_grid)
        self.input_proj = src.input_proj
        self.pos_embed = src.pos_embed
        self.encoder = src.encoder
        self.head = src.head

    def forward(self, x: torch.Tensor, take_offramp: torch.Tensor) -> torch.Tensor:
        if take_offramp.dim() == 1:
            take_offramp = take_offramp.unsqueeze(1)
        take_offramp = take_offramp.to(dtype=x.dtype)
        x = self.backbone(x)
        x = F.interpolate(
            x,
            size=(self.p, self.p),
            mode="bilinear",
            align_corners=False,
        )
        _b, _c, _h, _w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = torch.cat([x, take_offramp], dim=1)
        return self.head(x)


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def driving_net_parameter_rows(model: DrivingNet) -> tuple[list[tuple[str, int]], int, int]:
    """Named blocks, total params, trainable params."""
    if model.use_transformer:
        rows: list[tuple[str, int]] = [
            ("backbone (2× Conv2d + ReLU)", sum(p.numel() for p in model.backbone.parameters())),
            (f"adaptive_avg_pool ({model.token_grid}×{model.token_grid})", 0),
            ("input_proj (Linear)", sum(p.numel() for p in model.input_proj.parameters())),
            ("pos_embed (learned)", int(model.pos_embed.numel())),
            (
                f"transformer_encoder ({cfg.MODEL_TRANSFORMER_NUM_LAYERS} layers)",
                sum(p.numel() for p in model.encoder.parameters()),
            ),
            (
                "head (Linear; in = pooled ‖ take_offramp)",
                sum(p.numel() for p in model.head.parameters()),
            ),
        ]
    else:
        rows = [
            ("backbone (2× Conv2d + ReLU)", sum(p.numel() for p in model.backbone.parameters())),
            ("Flatten (no params)", 0),
            (
                "Linear (flattened ‖ take_offramp → out)",
                sum(p.numel() for p in model.mlp_lin.parameters()),
            ),
        ]
    total = sum(n for _, n in rows)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return rows, total, trainable


def _mermaid_block(model: DrivingNet) -> str:
    h = cfg.CAMERA_IMAGE_SIZE
    c1, c2 = cfg.MODEL_CONV1_CHANNELS, cfg.MODEL_CONV2_CHANNELS
    k, s = cfg.MODEL_KERNEL_SIZE, cfg.MODEL_STRIDE
    out = cfg.MODEL_OUTPUT_DIM
    if not model.use_transformer:
        fh = cfg.MODEL_FLATTEN_DIM
        f1 = fh + 1
        return f"""flowchart TB
  IMG["RGB image (N, 3, {h}, {h})"]
  OFFRAMP["**Off-ramp control**<br/>tensor `take_offramp` (N, 1)<br/>values 0 or 1 — **not** image pixels<br/>train: `labels.csv`; sim: `SIM_TAKE_OFFRAMP` / `SIM_TAKE_OFFRAMP_UPPER_HALF_NAV`"]
  subgraph CNN["CNN backbone"]
    C1["Conv2d 3→{c1}, k={k}, s={s} + ReLU"]
    C2["Conv2d {c1}→{c2}, k={k}, s={s} + ReLU"]
  end
  FL["Flatten → (N, {fh})"]
  CAT["`torch.cat` (vision feats, take_offramp) → (N, {f1})"]
  OUT["Linear (N, {f1})→(N, {out})<br/>steering = [:, 0]"]
  IMG --> C1 --> C2 --> FL --> CAT --> OUT
  OFFRAMP --> CAT
"""
    p = model.token_grid
    t = model.num_tokens
    d = cfg.MODEL_TRANSFORMER_D_MODEL
    L = cfg.MODEL_TRANSFORMER_NUM_LAYERS
    nh = cfg.MODEL_TRANSFORMER_NHEAD
    ff = cfg.MODEL_TRANSFORMER_FF_DIM
    return f"""flowchart TB
  IMG["RGB image (N, 3, {h}, {h})"]
  OFFRAMP["**Off-ramp control**<br/>tensor `take_offramp` (N, 1)<br/>values 0 or 1 — **not** image pixels<br/>train: `labels.csv`; sim: `SIM_TAKE_OFFRAMP` / `SIM_TAKE_OFFRAMP_UPPER_HALF_NAV`"]
  subgraph CNN["CNN backbone"]
    C1["Conv2d 3→{c1}, k={k}, s={s} + ReLU"]
    C2["Conv2d {c1}→{c2}, k={k}, s={s} + ReLU"]
  end
  POOL["AdaptiveAvgPool2d → ({p}×{p}) feature map"]
  TOK["Reshape → (N, {t}, {c2}) tokens"]
  PROJ["Linear {c2}→{d} + pos_embed ({t}×{d})"]
  subgraph TX["Transformer"]
    ENC["Encoder ×{L}: d={d}, heads={nh}, FFN={ff}"]
  end
  POOL2["Mean over tokens → (N, {d})"]
  CAT["`torch.cat` (pooled vector, take_offramp) → (N, {d}+1)"]
  HEAD["Linear (N, {d}+1)→(N, {out})<br/>steering = [:, 0]"]
  IMG --> C1 --> C2 --> POOL --> TOK --> PROJ --> ENC --> POOL2 --> CAT --> HEAD
  OFFRAMP --> CAT
"""


def _write_architecture_png(path: str, model: DrivingNet, rows: list[tuple[str, int]]) -> None:
    """Left-to-right backbone blocks + off-ramp control box merging into the final Linear."""
    labels = [f"{name}\n({_fmt_int(n)} params)" if n else f"{name}\n(no params)" for name, n in rows]
    nbox = len(labels)
    fig, ax = plt.subplots(figsize=(16, 5.2))
    ax.set_xlim(0, max(8, nbox * 1.2 + 1))
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    total = sum(n for _, n in rows)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fig.suptitle(
        f"DrivingNet — {_fmt_int(total)} parameters ({_fmt_int(trainable)} trainable)\n"
        r"Off-ramp: scalar $take\_offramp$ (N,1) is concatenated with vision features before the final Linear",
        fontsize=11,
        fontweight="bold",
    )
    for i, label in enumerate(labels):
        x = 0.8 + i * 1.15
        rect = mpatches.FancyBboxPatch(
            (x, 1.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#333",
            facecolor="#e8f4fc" if i % 2 == 0 else "#f0f8e8",
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.5,
            1.5,
            label,
            ha="center",
            va="center",
            fontsize=8,
            linespacing=1.15,
        )
        if i < nbox - 1:
            ax.annotate(
                "",
                xy=(x + 1.15, 1.5),
                xytext=(x + 1.0, 1.5),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
            )
    if nbox >= 1:
        x_last = 0.8 + (nbox - 1) * 1.15
        cx_last = x_last + 0.5
        ix = max(0.35, x_last - 1.05)
        iy = 0.12
        iw, ih = 1.15, 0.88
        intent = mpatches.FancyBboxPatch(
            (ix, iy),
            iw,
            ih,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.4,
            edgecolor="#b35900",
            facecolor="#ffe8d4",
        )
        ax.add_patch(intent)
        ax.text(
            ix + iw / 2,
            iy + ih / 2,
            "Off-ramp control\n`take_offramp` (N, 1)\nnot image pixels\n→ concat before Linear",
            ha="center",
            va="center",
            fontsize=7,
            linespacing=1.1,
        )
        ax.annotate(
            "",
            xy=(cx_last, 1.0),
            xytext=(ix + iw / 2, iy + ih),
            arrowprops=dict(
                arrowstyle="->",
                color="#b35900",
                lw=1.6,
                connectionstyle="arc3,rad=0.12",
            ),
        )
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_drivingnet_onnx(model: DrivingNet, onnx_path: str) -> tuple[bool, str]:
    """
    Export ``DrivingNet`` to ONNX for [Netron](https://github.com/lutzroeder/netron).

    Two graph inputs: ``image`` (N, 3, H, H), ``take_offramp`` (N, 1). Output: ``out`` (N, MODEL_OUTPUT_DIM).

    Temporarily moves the model to CPU for export, then restores device and ``train()`` / ``eval()``
    so callers (e.g. ``train.py``) are not left with weights on CPU while batches are on CUDA.

    Returns:
        ``(True, "")`` on success, or ``(False, error message)``.
    """
    orig_device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    h = int(cfg.CAMERA_IMAGE_SIZE)
    cpu = torch.device("cpu")
    try:
        m = model.to(cpu)
        export_body: nn.Module = (
            _DrivingNetOnnxFriendly(m) if m.use_transformer else m
        )
        dummy_x = torch.randn(1, 3, h, h, device=cpu)
        dummy_take = torch.zeros(1, 1, device=cpu)
        export_kw: dict = dict(
            model=export_body,
            args=(dummy_x, dummy_take),
            f=onnx_path,
            input_names=["image", "take_offramp"],
            output_names=["out"],
            dynamic_axes={
                "image": {0: "batch"},
                "take_offramp": {0: "batch"},
                "out": {0: "batch"},
            },
            opset_version=17,
        )
        try:
            torch.onnx.export(**export_kw, dynamo=False)
        except TypeError:
            torch.onnx.export(**export_kw)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        model.to(orig_device)
        model.train(was_training)


def write_architecture_artifacts(run_dir: str, model: nn.Module) -> None:
    """
    Write ``architecture.md`` (Mermaid + table), ``architecture.png``, and ``driving_net.onnx``
    (for Netron) under ``run_dir``.

    Expects a ``DrivingNet`` instance; raises ``TypeError`` otherwise.
    """
    if not isinstance(model, DrivingNet):
        raise TypeError(f"Expected DrivingNet, got {type(model).__name__}")
    rows, total, trainable = driving_net_parameter_rows(model)
    md_path = os.path.join(run_dir, "architecture.md")
    png_path = os.path.join(run_dir, "architecture.png")

    head_name = "Transformer head" if model.use_transformer else "Linear readout (no FC hidden)"
    lines: list[str] = [
        "# DrivingNet architecture",
        "",
        f"- **Head:** {head_name} (`MODEL_USE_TRANSFORMER_HEAD={model.use_transformer}`)",
        "",
        f"- **Total parameters:** {_fmt_int(total)}",
        f"- **Trainable parameters:** {_fmt_int(trainable)}",
        "",
        "## Parameter breakdown",
        "",
        "| Block | Parameters |",
        "| --- | ---: |",
    ]
    for name, n in rows:
        lines.append(f"| {name} | {_fmt_int(n)} |")
    lines.extend(
        [
            "",
            "### Off-ramp control input `take_offramp`",
            "",
            "Not part of the image tensor. A per-batch scalar **(N, 1)** with values **0 or 1** is **concatenated** to the vision vector before the final `Linear` (`DrivingNet.forward`). The **Mermaid diagram** and **`architecture.png`** both show this side input merging at the `torch.cat` step.",
            "",
            "- **Training:** `take_offramp` column in `labels.csv` (from `generate_dataset`: main rows = 0, `offramp_*.jpg` = 1).",
            "- **Simulation:** per-step `take_offramp` tensor from `SIM_TAKE_OFFRAMP` or geographic `SIM_TAKE_OFFRAMP_UPPER_HALF_NAV` in `config.py` (second argument to `model(...)`); merge-at-branch follows the same intent (see `simulate.py`).",
            "- **Omit / `None`:** treated as all zeros (main road).",
            "",
            "## Diagram (Mermaid)",
            "",
            "Render this block in a Mermaid-capable viewer (e.g. GitHub, VS Code preview).",
            "",
            "```mermaid",
            _mermaid_block(model).rstrip(),
            "```",
            "",
        ]
    )

    onnx_path = os.path.join(run_dir, "driving_net.onnx")
    onnx_ok, onnx_err = export_drivingnet_onnx(model, onnx_path)
    if onnx_ok:
        pool_note = (
            ""
            if not model.use_transformer
            else (
                "\n\nWith the **transformer** head, ONNX uses **bilinear resize** to the token grid "
                "instead of `AdaptiveAvgPool2d` (legacy exporter limitation). Graph topology and weights "
                "match; use this file for **visualization**, not bit-identical inference."
            )
        )
        lines.extend(
            [
                "## Netron",
                "",
                f"Open **`{os.path.basename(onnx_path)}`** from this folder in [Netron](https://netron.app) "
                "(desktop app or browser). Install **`onnx`** if export failed (`pip install onnx`).",
                "",
                "- Inputs: **`image`** `(batch, 3, H, H)`, **`take_offramp`** `(batch, 1)` — off-ramp control is **not** part of the image tensor.",
                "- Output: **`out`**; steering is `[:, 0]`.",
                pool_note,
                "",
            ]
        )
    else:
        err_path = os.path.join(run_dir, "onnx_export_error.txt")
        with open(err_path, "w", encoding="utf-8") as ef:
            ef.write(onnx_err)
        lines.extend(
            [
                "## Netron",
                "",
                f"ONNX export failed ({onnx_err}). Details: `{os.path.basename(err_path)}`. "
                "Try `pip install onnx` and re-run training, or upgrade PyTorch.",
                "",
            ]
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _write_architecture_png(png_path, model, rows)
