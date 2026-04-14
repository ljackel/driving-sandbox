import os
import re

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import config as cfg


def resolve_labels_csv_path() -> str:
    """
    Path to the active labels file: ``labels.csv``, or ``labels_new.csv`` if it exists and is newer
    (e.g. Windows lock on the primary file during generation).
    """
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


def count_train_test_examples() -> tuple[int, int]:
    """Counts of ``train/`` and ``test/`` rows in the active labels CSV (no image loading)."""
    path = resolve_labels_csv_path()
    df = pd.read_csv(path)
    n_train = int(df["image_path"].str.startswith("train/").sum())
    n_test = int(df["image_path"].str.startswith("test/").sum())
    return n_train, n_test


_FRAME_IDX_RE = re.compile(r"frame_(\d+)\.", re.IGNORECASE)


def train_perturb_stats_from_labels() -> tuple[int, int, float]:
    """
    From the active labels CSV: total ``train/`` rows, how many are perturbation views (frame index
    ``>= NUM_TRAIN_FRAMES``, matching ``generate_dataset`` naming), and that fraction of the train split.
    """
    df = pd.read_csv(resolve_labels_csv_path())
    train = df[df["image_path"].str.startswith("train/")]
    n = len(train)
    clean_n = int(cfg.NUM_TRAIN_FRAMES)
    n_pert = 0
    for raw in train["image_path"]:
        norm = str(raw).replace("\\", "/")
        base = norm.rsplit("/", 1)[-1]
        m = _FRAME_IDX_RE.search(base)
        if m is not None and int(m.group(1)) >= clean_n:
            n_pert += 1
    frac = float(n_pert) / float(n) if n else 0.0
    return n, n_pert, frac


def prepare_perspective_pil_for_model(im: Image.Image) -> Image.Image:
    """
    Optionally crop to the bottom half of the perspective image (rows nearest the vehicle).

    Used before ``Resize`` to ``CAMERA_IMAGE_SIZE`` in ``train.py``, ``evaluate_test.py``. When
    ``PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY`` is false, returns ``im`` unchanged.
    """
    if not cfg.PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY:
        return im
    w, h = im.size
    return im.crop((0, h // 2, w, h))


class DrivingDataset(Dataset):
    """
    PyTorch dataset: ``image_path``, ``steering``, and optional ``take_offramp`` from ``labels.csv``.

    Steering matches ``generate_dataset`` (scaled κ; recentering on perturbed rows when enabled in
    ``config``), clipped to
    ``[STEERING_CLIP_MIN, STEERING_CLIP_MAX]``. Missing ``take_offramp`` is treated as 0 (main road).
    """

    def __init__(self, csv_file, root_dir, transform=None, path_prefix=None):
        """
        Load labels and optionally filter by path prefix (train vs test split).

        Args:
            csv_file: Path to CSV with ``image_path``, ``steering``, optional ``take_offramp``.
            root_dir: Base directory prepended to each relative ``image_path``.
            transform: Optional torchvision-style transform applied to PIL RGB images.
            path_prefix: If set, keep only rows whose ``image_path`` starts with this
                (e.g. ``'train/'`` or ``'test/'``).
        """
        df = pd.read_csv(csv_file)
        if path_prefix is not None:
            df = df[df["image_path"].str.startswith(path_prefix)].reset_index(
                drop=True
            )
        self.driving_labels = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Number of rows after optional ``path_prefix`` filtering."""
        return len(self.driving_labels)

    def __getitem__(self, idx):
        """
        Load one image, steering scalar, and off-ramp intent (0 = main, 1 = ramp).

        Args:
            idx: Row index into the filtered dataframe.

        Returns:
            Tuple ``(image, steering, take_offramp)`` — all ``float32`` tensors; intent shape ``()``.
        """
        row = self.driving_labels.iloc[idx]
        img_name = os.path.join(self.root_dir, str(row["image_path"]))
        image = Image.open(img_name).convert("RGB")

        steering = torch.tensor(float(row["steering"]), dtype=torch.float32)
        if "take_offramp" in self.driving_labels.columns:
            take = float(row["take_offramp"])
        else:
            take = 0.0
        take_offramp = torch.tensor(take, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, steering, take_offramp