import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_cartoon_sandbox(image_size=1024):
    """
    Generates a cartoon, bird's eye view of a 1km square world.
    Assumptions:
    - 1 pixel = 0.9765 meters (1000m / 1024px)
    - World is 1km square.
    - Lanes are 4m wide (approx. 4.1 px).
    - Road is curvey.
    """
    # 1. Configuration for GPU usage (or CPU if no GPU found)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating world on: {device}...")

    # 2. Convert meters to pixel dimensions (approx.)
    world_meters = 1000
    pixels_per_meter = image_size / world_meters
    lane_width_px = int(4 * pixels_per_meter)
    total_road_width = lane_width_px * 2

    # 3. Create the Flat Green World
    # Start on CPU (Numpy is often faster for initial grid creation)
    world = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    world[:] = (34, 139, 34) # Cartoon Green Color (RGB)

    # 4. Generate the Curvy Road Path (using Splines)
    # Define keypoints (Control Points) for the curve (bottom to top)
    num_points = 10
    # Control points along increasing y (CubicSpline requires strictly increasing x)
    x_points = np.random.normal(image_size // 2, 100, num_points)
    y_points = np.linspace(0, image_size, num_points)

    # Use a Cubic Spline to interpolate the curve
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(y_points, x_points)
    
    # Generate the smooth path (Y values 0-1024)
    y_new = np.linspace(0, image_size, image_size * 2) # Doubled for smooth lines
    x_new = cs(y_new)
    
    # Convert path to integer pixel coordinates
    points = np.vstack((x_new, y_new)).T.astype(np.int32)
    
    # 5. Draw the Black Road Base
    # Use OpenCV to draw the smooth curve (on CPU first, simpler)
    cv2.polylines(world, [points], False, (50, 50, 50), thickness=total_road_width)
    
    # 6. Draw the White Dashed Line (Center)
    dash_length = int(8 * pixels_per_meter) # 8 meters long
    gap_length = int(5 * pixels_per_meter) # 5 meters gap
    
    # Loop through the path to create dashes
    for i in range(0, len(points) - dash_length, dash_length + gap_length):
        start_pt = tuple(points[i])
        end_pt = tuple(points[i + dash_length // 2])
        cv2.line(world, start_pt, end_pt, (255, 255, 255), thickness=1)

    # 7. Optional: The Cartoon-ify Filter (CPU, using OpenCV)
    # Simple median blur can give it that clean "cartoon" look
    world = cv2.medianBlur(world, 5)

    # 8. Push to GPU (Verify the Handshake)
    # We turn the image into a PyTorch Tensor on your NVIDIA GPU
    world_tensor = torch.from_numpy(world).to(device)
    print(f"World generated and moved to GPU: {world_tensor.shape}")

    return world # Return the numpy image for easy viewing

# Run the generator
if __name__ == "__main__":
    cartoon_world = create_cartoon_sandbox(image_size=1024)
    
    # View the generated image (using Matplotlib)
    plt.figure(figsize=(10, 10))
    plt.imshow(cartoon_world)
    plt.title("Generated Birds-Eye Sandbox (1km x 1km)")
    plt.axis('off')
    plt.show()