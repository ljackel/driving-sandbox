"""
Evaluate a trained DrivingNet on the held-out test split (data/test/).
Compare predicted steering (channel 0) to CSV ground truth.
Writes data/test_pred/*.jpg with target (white) and prediction (yellow) overlaid.
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import config as cfg
from data_loader import DrivingDataset
from driving_model import DrivingNet


def _labels_csv_path() -> str:
    """Return ``data/labels.csv`` or ``labels_new.csv`` if the alternate file is newer."""
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


def _put_outlined_bgr(
    img, text, org_xy, color_bgr, scale=0.4, thick=1
):
    """Draw ``text`` on ``img`` with a black outline for readability (in-place)."""
    x, y = org_xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    for dx, dy in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
    ):
        cv2.putText(
            img,
            text,
            (x + dx, y + dy),
            font,
            scale,
            (0, 0, 0),
            thick + 1,
            cv2.LINE_AA,
        )
    cv2.putText(img, text, (x, y), font, scale, color_bgr, thick, cv2.LINE_AA)


def main():
    """Load ``data/driving_net.pt``, score the test split, write annotated images to ``data/test_pred/``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(cfg.DATA_DIR, cfg.CHECKPOINT_FILENAME)
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(
            f"Missing {checkpoint_path!r}. Run train.py first to produce weights."
        )

    transform = transforms.Compose(
        [
            transforms.Resize((cfg.CAMERA_IMAGE_SIZE, cfg.CAMERA_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.NORMALIZE_MEAN, cfg.NORMALIZE_STD),
        ]
    )

    csv_path = _labels_csv_path()
    test_ds = DrivingDataset(
        csv_file=csv_path,
        root_dir=cfg.DATA_DIR,
        transform=transform,
        path_prefix="test/",
    )
    if len(test_ds) == 0:
        raise SystemExit(
            f"No test rows in {csv_path!r}. Regenerate data with generate_dataset.py."
        )

    loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = DrivingNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out = model(images).squeeze()
            pred = out[:, 0].detach().cpu()
            preds.append(pred)
            gts.append(labels)

    pred = torch.cat(preds).numpy()
    gt = torch.cat(gts).numpy()
    err = pred - gt

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    max_abs = float(np.max(np.abs(err)))

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    print(f"Test samples: {len(test_ds)}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Max |error|: {max_abs:.6f}")
    print(f"R^2:  {r2:.6f}")
    print()
    print("Sample (path | ground_truth | predicted | error)")
    df = test_ds.driving_labels
    for i in range(min(10, len(pred))):
        path = df.iloc[i, 0]
        print(
            f"  {path:32s}  gt={gt[i]:+.4f}  pred={pred[i]:+.4f}  err={err[i]:+.4f}"
        )

    out_dir = os.path.join(cfg.DATA_DIR, "test_pred")
    os.makedirs(out_dir, exist_ok=True)
    root = test_ds.root_dir
    white_bgr = (255, 255, 255)
    yellow_bgr = (0, 255, 255)
    for i in range(len(pred)):
        rel = df.iloc[i, 0]
        src = os.path.join(root, rel.replace("/", os.sep))
        bgr = cv2.imread(src)
        if bgr is None:
            continue
        _put_outlined_bgr(
            bgr,
            f"target: {gt[i]:+.4f}",
            cfg.EVAL_TARGET_TEXT_POS,
            white_bgr,
            scale=cfg.EVAL_OUTLINED_FONT_SCALE,
            thick=cfg.EVAL_OUTLINED_FONT_THICK,
        )
        _put_outlined_bgr(
            bgr,
            f"pred:   {pred[i]:+.4f}",
            cfg.EVAL_PRED_TEXT_POS,
            yellow_bgr,
            scale=cfg.EVAL_OUTLINED_FONT_SCALE,
            thick=cfg.EVAL_OUTLINED_FONT_THICK,
        )
        base = os.path.basename(rel)
        cv2.imwrite(os.path.join(out_dir, base), bgr)
    print()
    print(f"Annotated test images saved to {out_dir!r} (target=white, pred=yellow).")


if __name__ == "__main__":
    main()
