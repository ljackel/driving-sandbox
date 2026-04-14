import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import config as cfg


class DrivingWorld:
    """
    Top-down raster map: cubic-spline centerline from ``SPLINE_X_DELTAS_BOTTOM_TO_TOP``, two-lane road,
    dashed center marking. Curvature of this spline drives dataset steering labels after global scaling.
    """

    def __init__(
        self,
        image_size=cfg.WORLD_IMAGE_SIZE,
        world_meters=cfg.WORLD_METERS,
    ):
        """
        Build the spline road and render ``self.image`` (BGR, ``image_size`` square).

        Args:
            image_size: Map resolution in pixels (width and height).
            world_meters: Physical extent represented by one edge of the map (meters).
        """
        self.size = image_size
        self.px_per_m = image_size / world_meters

        fr = cfg.SPLINE_Y_FRACTIONS_TOP_TO_BOTTOM
        if len(fr) != cfg.SPLINE_NUM_CONTROL_POINTS:
            raise ValueError(
                "SPLINE_Y_FRACTIONS_TOP_TO_BOTTOM length must match SPLINE_NUM_CONTROL_POINTS"
            )
        self.y_pts = np.asarray(fr, dtype=np.float64) * float(image_size)
        if np.any(np.diff(self.y_pts) <= 0):
            raise ValueError("SPLINE_Y_FRACTIONS_TOP_TO_BOTTOM must be strictly increasing")
        c = image_size // 2
        x_bottom_to_top = np.array(
            [c + d for d in cfg.SPLINE_X_DELTAS_BOTTOM_TO_TOP], dtype=np.float64
        )
        self.x_pts = x_bottom_to_top[::-1]

        # y_pts increase downward; sim starts at large y. Clamp dx/dy=0 at bottom so the road
        # begins straight up (constant x while moving toward smaller y).
        self.cs = CubicSpline(
            self.y_pts,
            self.x_pts,
            bc_type=("not-a-knot", (1, 0.0)),
        )
        self.image = self.create_map()

    def get_road_center(self, y):
        """
        Return the road centerline x-coordinate (pixels) at image row ``y``.

        Args:
            y: Vertical image coordinate (pixels, origin top-left).
        """
        return float(self.cs(y))

    def create_map(self):
        """Paint grass, a thick gray polyline for pavement, and white dash segments."""
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
        # Orientation: top of image is y=0 (small y); blue stripe marks that edge for debugging.
        cv2.line(
            world,
            (0, 0),
            (self.size - 1, 0),
            (255, 0, 0),
            thickness=4,
        )
        return world


if __name__ == "__main__":
    world = DrivingWorld()
    plt.imshow(cv2.cvtColor(world.image, cv2.COLOR_BGR2RGB))
    plt.show()
