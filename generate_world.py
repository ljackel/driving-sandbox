import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class DrivingWorld:
    def __init__(self, image_size=1024, world_meters=500):
        self.size = image_size
        self.px_per_m = image_size / world_meters
        
        # Cursor's Fix: Strictly increasing y for the spline (0 to 1024)
        self.y_pts = np.linspace(0, image_size, 6)
        x_bottom_to_top = np.array([
            image_size // 2, 150 + image_size // 2, 
            image_size // 2 - 100, image_size // 2 + 50, 
            image_size // 2 - 200, image_size // 2
        ], dtype=np.float64)
        self.x_pts = x_bottom_to_top[::-1]
        
        self.cs = CubicSpline(self.y_pts, self.x_pts)
        self.image = self.create_map()

    def get_road_center(self, y):
        """Returns the X coordinate for any given Y."""
        return float(self.cs(y))

    def create_map(self):
        world = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        world[:] = (34, 139, 34) # Green
        
        y_new = np.linspace(0, self.size, 2000) 
        x_new = self.cs(y_new)
        points = np.vstack((x_new, y_new)).T.astype(np.int32)
        
        lane_px = int(4 * self.px_per_m)
        cv2.polylines(world, [points], False, (40, 40, 40), thickness=lane_px * 2)
        
        dash_len = int(3 * self.px_per_m)
        dash_gap = int(3 * self.px_per_m)
        for i in range(0, len(points) - dash_len, dash_len + dash_gap):
            pt1 = tuple(points[i])
            pt2 = tuple(points[i + dash_len])
            cv2.line(world, pt1, pt2, (255, 255, 255), thickness=1)
        return world

if __name__ == "__main__":
    world = DrivingWorld()
    plt.imshow(world.image)
    plt.show()
