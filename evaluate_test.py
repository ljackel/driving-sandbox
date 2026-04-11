"""
Evaluate a trained DrivingNet on the held-out test split (data/test/).
Compare predicted steering (channel 0) to CSV ground truth.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import DrivingDataset
from driving_model import DrivingNet


def _labels_csv_path() -> str:
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join("data", "driving_net.pt")
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(
            f"Missing {checkpoint_path!r}. Run train.py first to produce weights."
        )

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    csv_path = _labels_csv_path()
    test_ds = DrivingDataset(
        csv_file=csv_path,
        root_dir="data",
        transform=transform,
        path_prefix="test/",
    )
    if len(test_ds) == 0:
        raise SystemExit(
            f"No test rows in {csv_path!r}. Regenerate data with generate_dataset.py."
        )

    loader = DataLoader(test_ds, batch_size=16, shuffle=False)

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


if __name__ == "__main__":
    main()
