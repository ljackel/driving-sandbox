"""
Write side-by-side **clean vs aligned-perturbed** train frames plus an amplified diff panel.

Only pairs ``frame_{i:04d}.jpg`` with ``frame_{NUM_TRAIN_FRAMES + i:04d}.jpg`` for ``i`` in
``[0, NUM_TRAIN_FRAMES)`` are scanned (the 1:1 clean/perturbed grid). Extra frames from
``TRAIN_PERTURB_EXTRA_FRAMES`` are not paired here. Ranks pairs by mean absolute pixel difference
(perturbation backoff can make some pairs nearly identical). Output: ``data/perturb_compare/``.

Run from project root: ``python show_perturb_pairs.py``
"""
from __future__ import annotations

import os

import cv2
import numpy as np

import config as cfg


def _load_pair(train_dir: str, i: int, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    clean_p = os.path.join(train_dir, f"frame_{i:04d}.jpg")
    pert_p = os.path.join(train_dir, f"frame_{n + i:04d}.jpg")
    a = cv2.imread(clean_p)
    b = cv2.imread(pert_p)
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    return a, b


def _composite(clean: np.ndarray, pert: np.ndarray, scale: float = 8.0) -> np.ndarray:
    d = cv2.absdiff(clean, pert)
    d = np.clip(d.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    h, w = clean.shape[:2]
    bar = np.zeros((24, w * 3, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        "clean",
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        "perturbed (same y)",
        (w + 4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        f"absdiff x{scale:.0f}",
        (2 * w + 4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    row = np.hstack((clean, pert, d))
    return np.vstack((bar, row))


def main() -> None:
    n = cfg.NUM_TRAIN_FRAMES
    train_dir = os.path.join(cfg.DATA_DIR, "train")
    out_dir = os.path.join(cfg.DATA_DIR, "perturb_compare")
    os.makedirs(out_dir, exist_ok=True)

    scores: list[tuple[float, int]] = []
    step = max(1, n // 400)
    for i in range(0, n, step):
        pair = _load_pair(train_dir, i, n)
        if pair is None:
            continue
        clean, pert = pair
        mean_diff = float(np.mean(cv2.absdiff(clean, pert)))
        scores.append((mean_diff, i))

    scores.sort(key=lambda t: t[0], reverse=True)
    top = scores[:12] if scores else []

    if not top:
        print(
            f"No pairs found under {train_dir!r}. "
            f"Expected frame_0000..frame_{n - 1:04d}.jpg and "
            f"frame_{n:04d}..frame_{2 * n - 1:04d}.jpg. Run generate_dataset.py first."
        )
        return

    print(f"Mean abs diff (0-255 per channel): best={top[0][0]:.4f}, worst in scan={top[-1][0]:.4f}")
    for rank, (mean_diff, i) in enumerate(top):
        pair = _load_pair(train_dir, i, n)
        if pair is None:
            continue
        clean, pert = pair
        comp = _composite(clean, pert, scale=8.0)
        out_path = os.path.join(out_dir, f"pair_rank{rank:02d}_idx{i:04d}_meandiff_{mean_diff:.2f}.png")
        cv2.imwrite(out_path, comp)
        print(f"  wrote {out_path}")

    dbg_dir = os.path.join(cfg.DATA_DIR, cfg.TRAIN_PERTURB_DEBUG_SUBDIR)
    if os.path.isdir(dbg_dir):
        print(
            f"Annotated perturb debug images (lat/yaw/kappa) are in {dbg_dir!r} "
            f"(same filenames as perturbed train frames)."
        )


if __name__ == "__main__":
    main()
