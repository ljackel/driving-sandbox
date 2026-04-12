"""Generate random RGB crops and a matching labels CSV under ``dummy_data/`` (no functions)."""
import os

import numpy as np
import pandas as pd
from PIL import Image

import config as cfg
from reproducibility import set_global_seed

set_global_seed(cfg.TRAIN_SEED)

# 1. Setup directories
data_dir = "dummy_data"
images_dir = os.path.join(data_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# 2. Parameters
num_samples = cfg.DUMMY_NUM_SAMPLES
image_size = (cfg.CAMERA_IMAGE_SIZE, cfg.CAMERA_IMAGE_SIZE)
data_records = []

print(f"Generating {num_samples} dummy images...")

for i in range(num_samples):
    random_array = np.random.randint(
        0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8
    )
    img = Image.fromarray(random_array)
    img_filename = f"frame_{i:04d}.jpg"
    img.save(os.path.join(images_dir, img_filename))
    steering_angle = np.random.uniform(
        cfg.DUMMY_STEERING_MIN, cfg.DUMMY_STEERING_MAX
    )
    data_records.append({"image_path": img_filename, "steering": steering_angle})

# 3. Save the labels to CSV
df = pd.DataFrame(data_records)
df.to_csv(os.path.join(data_dir, "labels.csv"), index=False)

print(f"Success! Created {num_samples} images in '{images_dir}' and 'labels.csv'.")
