import os
import pandas as pd
import numpy as np
from PIL import Image

# 1. Setup directories
data_dir = "dummy_data"
images_dir = os.path.join(data_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# 2. Parameters
num_samples = 100
image_size = (128, 128)
data_records = []

print(f"Generating {num_samples} dummy images...")

for i in range(num_samples):
    # Create a random image (3 channels: R, G, B)
    random_array = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
    img = Image.fromarray(random_array)
    
    # Save image
    img_filename = f"frame_{i:04d}.jpg"
    img.save(os.path.join(images_dir, img_filename))
    
    # Generate a dummy steering angle between -1.0 (left) and 1.0 (right)
    steering_angle = np.random.uniform(-1.0, 1.0)
    
    # Store the filename and the label
    data_records.append({"image_path": img_filename, "steering": steering_angle})

# 3. Save the labels to CSV
df = pd.DataFrame(data_records)
df.to_csv(os.path.join(data_dir, "labels.csv"), index=False)

print(f"✅ Success! Created {num_samples} images in '{images_dir}' and 'labels.csv'.")