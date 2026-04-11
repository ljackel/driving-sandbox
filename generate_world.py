import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import config as cfg


class DrivingWorld:
    def __init__(
        self,
        image_size=cfg.WORLD_IMAGE_SIZE,
        world_meters=cfg.WORLD_METERS,
    ):
        self.size = image_size
        self.px_per_m = image_size / world_meters

        self.y_pts = np.linspace(0, image_size, cfg.SPLINE_NUM_CONTROL_POINTS)
        c = image_size // 2
        x_bottom_to_top = np.array(
            [c + d for d in cfg.SPLINE_X_DELTAS_BOTTOM_TO_TOP], dtype=np.float64
        )
        self.x_pts = x_bottom_to_top[::-1]

        self.cs = CubicSpline(self.y_pts, self.x_pts)
        self.image = self.create_map()

    def get_road_center(self, y):
        """Returns the X coordinate for any given Y."""
        return float(self.cs(y))

    def create_map(self):
        world = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        world[:] = cfg.WORLD_GREEN_BGR

        y_new = np.linspace(0, self.size, cfg.ROAD_POLYLINE_SAMPLES)
        x_new = self.cs(y_new)
        points = np.vstack((x_new, y_new)).T.astype(np.int32)

        lane_px = int(cfg.LANE_WIDTH_METERS * self.px_per_m)
        cv2.polylines(
            world,
            [points],
            False,
            cfg.WORLD_ROAD_BGR,
            thickness=lane_px * 2,
        )

        dash_len = int(cfg.DASH_LENGTH_METERS * self.px_per_m)
        dash_gap = int(cfg.DASH_GAP_METERS * self.px_per_m)
        for i in range(0, len(points) - dash_len, dash_len + dash_gap):
            pt1 = tuple(points[i])
            pt2 = tuple(points[i + dash_len])
            cv2.line(
                world,
                pt1,
                pt2,
                (255, 255, 255),
                thickness=cfg.ROAD_EDGE_THICKNESS,
            )
        return world


if __name__ == "__main__":
    world = DrivingWorld()
    plt.imshow(cv2.cvtColor(world.image, cv2.COLOR_BGR2RGB))
    plt.show()
