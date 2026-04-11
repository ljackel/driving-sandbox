import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

class DrivingDataset(Dataset):
    """PyTorch dataset reading ``image_path`` and ``steering`` rows from a CSV."""

    def __init__(self, csv_file, root_dir, transform=None, path_prefix=None):
        """
        Load labels and optionally filter by path prefix (train vs test split).

        Args:
            csv_file: Path to CSV with columns ``image_path``, ``steering``.
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
        Load one image and steering scalar.

        Args:
            idx: Row index into the filtered dataframe.

        Returns:
            Tuple ``(image, steering)`` where ``image`` is transformed tensor and
            ``steering`` is ``float32`` scalar tensor.
        """
        img_name = os.path.join(self.root_dir, self.driving_labels.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # 2. Get the steering angle or control command
        steering = self.driving_labels.iloc[idx, 1]
        steering = torch.tensor(float(steering), dtype=torch.float32)

        # 3. Apply transformations (like resizing)
        if self.transform:
            image = self.transform(image)

        return image, steering