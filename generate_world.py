import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import config as cfg


class DrivingWorld:
    """
    Top-down raster map: cubic-spline centerline from ``SPLINE_X_DELTAS_BOTTOM_TO_TOP``, two-lane road,
    dashed center marking, optional bottom-half off-ramp (``OFFRAMP_*``). Curvature of the **main**
    spline drives dataset steering labels after global scaling.
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

    @staticmethod
    def _densify_polyline(pts: np.ndarray, n_target: int) -> np.ndarray:
        """Resample open polyline to ``n_target`` points along piecewise-linear arc length."""
        if len(pts) < 2 or n_target < 2:
            return pts.astype(np.int32)
        p = pts.astype(np.float64)
        seg = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] <= 0:
            return pts.astype(np.int32)
        t = np.linspace(0.0, s[-1], n_target)
        xi = np.interp(t, s, p[:, 0])
        yi = np.interp(t, s, p[:, 1])
        return np.column_stack((xi, yi)).astype(np.int32)

    def _offramp_centerline_points(self) -> np.ndarray | None:
        """
        Dense integer polyline ``(N, 2)`` as ``(x, y)`` for the off-ramp, or ``None`` if disabled.

        Quadratic Bézier from the branch: the first control point lies on the main-road tangent
        (traffic toward decreasing ``y``) so departure is smooth, not a sharp kink.
        """
        if not cfg.OFFRAMP_ENABLE:
            return None
        y0 = float(np.clip(cfg.OFFRAMP_BRANCH_Y_FRAC, 0.0, 1.0)) * float(self.size)
        if y0 <= 0.5 * float(self.size):
            return None
        p0 = np.array([float(self.cs(y0)), y0], dtype=np.float64)
        dxdy = float(self.cs(y0, nu=1))
        # Unit tangent for motion with decreasing y (toward map top), along the centerline.
        t = np.array([-dxdy, -1.0], dtype=np.float64)
        t_norm = float(np.linalg.norm(t))
        if t_norm <= 1e-9:
            t = np.array([0.0, -1.0], dtype=np.float64)
        else:
            t /= t_norm
        L = float(cfg.OFFRAMP_TANGENT_CTRL_PX)
        p1 = p0 + L * t
        p2 = p0 + np.array(
            [float(cfg.OFFRAMP_END_DX_PX), float(cfg.OFFRAMP_END_DY_PX)], dtype=np.float64
        )
        n_bez = 96
        u = np.linspace(0.0, 1.0, n_bez, dtype=np.float64)
        omu = 1.0 - u
        pts = (
            (omu[:, np.newaxis] ** 2) * p0
            + (2.0 * omu * u)[:, np.newaxis] * p1
            + (u[:, np.newaxis] ** 2) * p2
        )
        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(self.size - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(self.size - 1))
        return self._densify_polyline(pts.astype(np.float64), max(120, n_bez * 4))

    @staticmethod
    def _draw_road_polyline(
        world: np.ndarray,
        points: np.ndarray,
        *,
        lane_px: int,
        dash_len: int,
        dash_gap: int,
    ) -> None:
        """Draw gray pavement and white dashes along one open polyline."""
        if len(points) < 2:
            return
        cv2.polylines(
            world,
            [points],
            False,
            cfg.WORLD_ROAD_BGR,
            thickness=lane_px * 2,
        )
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

    def create_map(self):
        """Paint grass, a thick gray polyline for pavement, and white dash segments."""
        world = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        world[:] = cfg.WORLD_GREEN_BGR

        y_new = np.linspace(0, self.size, cfg.ROAD_POLYLINE_SAMPLES)
        x_new = self.cs(y_new)
        points = np.vstack((x_new, y_new)).T.astype(np.int32)

        lane_px = int(cfg.LANE_WIDTH_METERS * self.px_per_m)
        dash_len = int(cfg.DASH_LENGTH_METERS * self.px_per_m)
        dash_gap = int(cfg.DASH_GAP_METERS * self.px_per_m)

        self._draw_road_polyline(world, points, lane_px=lane_px, dash_len=dash_len, dash_gap=dash_gap)

        off = self._offramp_centerline_points()
        if off is not None and len(off) >= 2:
            self._draw_road_polyline(world, off, lane_px=lane_px, dash_len=dash_len, dash_gap=dash_gap)
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
