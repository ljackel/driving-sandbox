"""
Write architecture diagram + parameter counts into a training ``runs/<stamp>/`` directory.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn as nn

import config as cfg
from driving_model import DrivingNet


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
  INTENT["take_offramp (N, 1)<br/>0 = main road, 1 = off-ramp<br/>CSV column in train; SIM_TAKE_OFFRAMP in sim"]
  subgraph CNN["CNN backbone"]
    C1["Conv2d 3→{c1}, k={k}, s={s} + ReLU"]
    C2["Conv2d {c1}→{c2}, k={k}, s={s} + ReLU"]
  end
  FL["Flatten → (N, {fh})"]
  CAT["torch.cat (feats, take_offramp) → (N, {f1})"]
  OUT["Linear (N, {f1})→(N, {out})<br/>steering = [:, 0]"]
  IMG --> C1 --> C2 --> FL --> CAT --> OUT
  INTENT --> CAT
"""
    p = model.token_grid
    t = model.num_tokens
    d = cfg.MODEL_TRANSFORMER_D_MODEL
    L = cfg.MODEL_TRANSFORMER_NUM_LAYERS
    nh = cfg.MODEL_TRANSFORMER_NHEAD
    ff = cfg.MODEL_TRANSFORMER_FF_DIM
    return f"""flowchart TB
  IMG["RGB image (N, 3, {h}, {h})"]
  INTENT["take_offramp (N, 1)<br/>0 = main road, 1 = off-ramp<br/>CSV column in train; SIM_TAKE_OFFRAMP in sim"]
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
  CAT["torch.cat (pooled, take_offramp) → (N, {d}+1)"]
  HEAD["Linear (N, {d}+1)→(N, {out})<br/>steering = [:, 0]"]
  IMG --> C1 --> C2 --> POOL --> TOK --> PROJ --> ENC --> POOL2 --> CAT --> HEAD
  INTENT --> CAT
"""


def _write_architecture_png(path: str, model: DrivingNet, rows: list[tuple[str, int]]) -> None:
    """Simple left-to-right block diagram with per-block parameter counts."""
    labels = [f"{name}\n({_fmt_int(n)} params)" if n else f"{name}\n(no params)" for name, n in rows]
    nbox = len(labels)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, max(8, nbox * 1.2 + 1))
    ax.set_ylim(0, 3)
    ax.axis("off")
    total = sum(n for _, n in rows)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fig.suptitle(
        f"DrivingNet — {_fmt_int(total)} parameters ({_fmt_int(trainable)} trainable)",
        fontsize=12,
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
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_architecture_artifacts(run_dir: str, model: nn.Module) -> None:
    """
    Write ``architecture.md`` (Mermaid + table) and ``architecture.png`` under ``run_dir``.

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
            "### Intent input `take_offramp`",
            "",
            "Not part of the image tensor. A per-batch scalar **(N, 1)** in **{0, 1}** is **concatenated** to the pooled vector before the final `Linear` (see `DrivingNet.forward`).",
            "",
            "- **Training:** `take_offramp` column in `labels.csv` (from `generate_dataset`: main rows = 0, `offramp_*.jpg` = 1).",
            "- **Simulation:** `SIM_TAKE_OFFRAMP` in `config.py` → tensor passed as the second argument to `model(...)`.",
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
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _write_architecture_png(png_path, model, rows)
