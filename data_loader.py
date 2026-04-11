import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

class DrivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, path_prefix=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations (image_path, steering).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            path_prefix (string, optional): If set, keep only rows whose image_path starts with this (e.g. "train/").
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
        # Tells PyTorch how many total images are in your sandbox
        return len(self.driving_labels)

    def __getitem__(self, idx):
        # 1. Get image name from the CSV
        img_name = os.path.join(self.root_dir, self.driving_labels.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # 2. Get the steering angle or control command
        steering = self.driving_labels.iloc[idx, 1]
        steering = torch.tensor(float(steering), dtype=torch.float32)

        # 3. Apply transformations (like resizing)
        if self.transform:
            image = self.transform(image)

        return image, steering