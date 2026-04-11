import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def create_cartoon_sandbox(image_size=1024):
    # 1. New Scale: 500 meters / 1024 pixels
    world_meters = 500 
    pixels_per_meter = image_size / world_meters
    
    lane_width_px = int(4 * pixels_per_meter)      # ~8 pixels
    dash_length = int(3 * pixels_per_meter)        # 3m dash
    dash_gap = int(3 * pixels_per_meter)           # 3m gap

    # 2. Create Green World
    world = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    world[:] = (34, 139, 34) # Forest Green

    # 3. Generate the Road Path (6 control points, bottom → top in list order)
    # CubicSpline requires strictly increasing independent variable → y must increase (0 = top)
    y_points = np.linspace(0, image_size, 6)
    x_bottom_to_top = np.array([
        image_size // 2,                    # Bottom center
        image_size // 2 + 150,
        image_size // 2 - 100,
        image_size // 2 + 50,
        image_size // 2 - 200,
        image_size // 2,                    # Top center
    ], dtype=np.float64)
    x_points = x_bottom_to_top[::-1]

    cs = CubicSpline(y_points, x_points)
    y_new = np.linspace(0, image_size, 2000) 
    x_new = cs(y_new)
    points = np.vstack((x_new, y_new)).T.astype(np.int32)
    
    # 4. Draw Black Road (16px total width for 2 lanes)
    cv2.polylines(world, [points], False, (40, 40, 40), thickness=lane_width_px * 2)
    
    # 5. Draw White Dashed Center Line
    # We iterate through the points and draw every other segment
    for i in range(0, len(points) - dash_length, dash_length + dash_gap):
        pt1 = tuple(points[i])
        pt2 = tuple(points[i + dash_length])
        cv2.line(world, pt1, pt2, (255, 255, 255), thickness=1)

    return world

if __name__ == "__main__":
    world_img = create_cartoon_sandbox()
    
    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(world_img)
    plt.title("0.5km World (High Res Road)")
    plt.axis('off')
    plt.show()
