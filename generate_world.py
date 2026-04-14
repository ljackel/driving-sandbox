from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import config as cfg
from perspective_camera import bev_point_to_camera_uv, perspective_camera_homography


def _as_roadkill_y_tuple(v: object) -> tuple[float, ...]:
    """Normalize a config roadkill-Y value (``float``, ``int``, or sequence)."""
    if isinstance(v, (int, float)):
        return (float(v),)
    if v is None:
        return ()
    return tuple(float(y) for y in v)


def _roadkill_obstacle_y_rows_training() -> tuple[float, ...]:
    """Rows used for **dataset** lateral reference (labels / train crops)."""
    return _as_roadkill_y_tuple(cfg.ROADKILL_OBSTACLE_Y_PX)


def _roadkill_obstacle_y_rows_eval_only() -> tuple[float, ...]:
    """Extra splat rows: on the map and in **sim** lateral only, not in training labels."""
    return _as_roadkill_y_tuple(getattr(cfg, "ROADKILL_EVAL_ONLY_OBSTACLE_Y_PX", ()))


def _roadkill_obstacle_y_rows_all() -> tuple[float, ...]:
    """All drawn splats and full **sim** detour geometry."""
    return _roadkill_obstacle_y_rows_training() + _roadkill_obstacle_y_rows_eval_only()


def _raised_cosine01(t: float) -> float:
    """Hann ease 0→1; zero slope at both ends (gentler lane-change than polynomial ease)."""
    t = float(np.clip(t, 0.0, 1.0))
    return float(0.5 - 0.5 * np.cos(np.pi * t))


def _splat_visibility_soft_weight(
    uv: tuple[float, float],
    cam_s: float,
    bottom_half_only: bool,
) -> float:
    """
    Fade detour in as the hazard occupies more of the forward view (avoids a one-frame step when
    ``ROADKILL_DETOUR_ONLY_WHEN_VISIBLE`` is on). ``v`` increases downward; near-ego is larger ``v``.
    """
    u, v = float(uv[0]), float(uv[1])
    m = 0.06 * cam_s
    if u < -m or u > cam_s + m or v < -m or v > cam_s + m:
        return 0.0
    if bottom_half_only and v < 0.5 * cam_s:
        return 0.0
    if bottom_half_only:
        v0, v1 = 0.5 * cam_s, cam_s
    else:
        v0, v1 = 0.1 * cam_s, cam_s
    denom = max(v1 - v0, 1e-6)
    v_frac = float(getattr(cfg, "ROADKILL_VIS_SPLAT_SOFT_RAMP_FRAC", 1.28))
    t_v = (v - v0) / max(v_frac * denom, 1e-6)
    f_v = _raised_cosine01(t_v)
    u_mid = 0.5 * cam_s
    half = 0.5 * cam_s
    t_u = 1.0 - abs(u - u_mid) / half
    f_u = _raised_cosine01(t_u)
    return float(np.clip(f_v * f_u, 0.0, 1.0))


def _roadkill_exit_ramp_length_px() -> float:
    """BEV ``y`` span to merge back to the right lane: ``ROADKILL_DETOUR_EXIT_CAR_LENGTHS`` × ego length (m) → px."""
    px_per_m = float(cfg.WORLD_IMAGE_SIZE) / float(cfg.WORLD_METERS)
    car_m = float(cfg.SIM_BEV_EGO_CAR_LENGTH_M)
    n = float(cfg.ROADKILL_DETOUR_EXIT_CAR_LENGTHS)
    return max(1e-3, n * car_m * px_per_m)


def _roadkill_approach_merge_before_core_px() -> float:
    """Last stretch before the core: blend visibility-driven ``w`` up to 1.0 (car lengths × ego, map px)."""
    px_per_m = float(cfg.WORLD_IMAGE_SIZE) / float(cfg.WORLD_METERS)
    car_m = float(cfg.SIM_BEV_EGO_CAR_LENGTH_M)
    n = float(cfg.ROADKILL_APPROACH_CORE_MERGE_CAR_LENGTHS)
    return max(1e-3, n * car_m * px_per_m)


def _roadkill_left_lane_blend_weight_at_row(y_row: float, y0: float) -> float:
    """
    Blend 0..1 for a **single** obstacle centered at ``y0``.

    Approach ramp uses ``SIM_SPEED_M_S`` and ``ROADKILL_DETOUR_BLEND_APPROACH_S`` (time → distance in ``y``).
    Exit ramp length is ``ROADKILL_DETOUR_EXIT_CAR_LENGTHS`` × ``SIM_BEV_EGO_CAR_LENGTH_M`` on the map scale.
    """
    hc = float(cfg.ROADKILL_DETOUR_CORE_HALF_PX)
    px_per_m = float(cfg.WORLD_IMAGE_SIZE) / float(cfg.WORLD_METERS)
    v = float(cfg.SIM_SPEED_M_S)
    bl_in = v * float(cfg.ROADKILL_DETOUR_BLEND_APPROACH_S) * px_per_m
    bl_out = _roadkill_exit_ramp_length_px()
    lead_px = v * float(cfg.ROADKILL_APPROACH_LEAD_S) * px_per_m
    c_lo = y0 - hc
    c_hi = y0 + hc
    e_hi = c_hi + bl_in + max(0.0, lead_px)
    e_lo = c_lo - bl_out
    if y_row > e_hi or y_row < e_lo:
        return 0.0
    if c_lo <= y_row <= c_hi:
        return 1.0
    if y_row > c_hi:
        span = e_hi - c_hi
        if span <= 1e-9:
            return 0.0
        t_lin = float(np.clip((e_hi - y_row) / span, 0.0, 1.0))
        return _raised_cosine01(t_lin)
    span = c_lo - e_lo
    if span <= 1e-9:
        return 0.0
    t_lin = float(np.clip((y_row - e_lo) / span, 0.0, 1.0))
    # Power 1 = symmetric ease; <1 lingers longer in the detour before finishing the return.
    p = float(getattr(cfg, "ROADKILL_DETOUR_EXIT_EASE_POWER", 1.0))
    t_ease = float(np.clip(t_lin, 0.0, 1.0) ** p)
    return _raised_cosine01(t_ease)


def roadkill_left_lane_blend_weight_training(y_row: float) -> float:
    """
    Blend for **dataset** generation: ``ROADKILL_OBSTACLE_Y_PX`` only (supervised detours).
    """
    if not cfg.ROADKILL_ENABLE:
        return 0.0
    w = 0.0
    for y0 in _roadkill_obstacle_y_rows_training():
        w = max(w, _roadkill_left_lane_blend_weight_at_row(y_row, y0))
    return w


def roadkill_left_lane_blend_weight(y_row: float) -> float:
    """
    Blend for **simulation** (and any caller needing full map behavior): training rows plus
    ``ROADKILL_EVAL_ONLY_OBSTACLE_Y_PX``. Uses **max** over all such rows.
    """
    if not cfg.ROADKILL_ENABLE:
        return 0.0
    w = 0.0
    for y0 in _roadkill_obstacle_y_rows_all():
        w = max(w, _roadkill_left_lane_blend_weight_at_row(y_row, y0))
    return w


def roadkill_splat_center_bev_xy(dw: DrivingWorld, y_obs: float) -> tuple[float, float]:
    """Right-lane splat reference point in BEV px (matches drawing / ``DATASET_RIGHT_LANE_LATERAL_FRAC``)."""
    y0 = float(y_obs)
    xc = float(dw.cs(y0))
    dxdY = float(dw.cs(y0, nu=1))
    norm = float(np.hypot(dxdY, 1.0))
    if norm < 1e-9:
        return xc, y0
    fx, fy = -dxdY / norm, -1.0 / norm
    rx, ry = -fy, fx
    lane_off = float(
        cfg.LANE_WIDTH_METERS * cfg.DATASET_RIGHT_LANE_LATERAL_FRAC * dw.px_per_m
    )
    return float(xc + rx * lane_off), float(y0 + ry * lane_off)


def _camera_uv_visible(
    u: float,
    v: float,
    cam_s: float,
    bottom_half_only: bool,
) -> bool:
    if u < 0.0 or u >= cam_s or v < 0.0 or v >= cam_s:
        return False
    if bottom_half_only and v < 0.5 * cam_s:
        return False
    return True


def _roadkill_blend_weight_visible_gated(
    y_row: float,
    world_bgr: np.ndarray,
    dw: DrivingWorld,
    x_center: float,
    psi: float,
    right_lane_offset_px: float,
    *,
    for_training_labels: bool,
) -> float:
    """
    **Approach** (before the core): lateral weight follows a **soft visibility** curve as soon as the
    splat appears in the **right-lane** camera; only in the last ``ROADKILL_APPROACH_CORE_MERGE_CAR_LENGTHS``
    car lengths is ``w`` blended up to 1 so the ego clears the obstacle. **Core** uses full ``w``.
    **Exit** is geometry-only so the return can finish after the splat leaves the FOV.
    """
    rows = (
        _roadkill_obstacle_y_rows_training()
        if for_training_labels
        else _roadkill_obstacle_y_rows_all()
    )
    fx = float(np.cos(psi))
    fy = float(np.sin(psi))
    rx = float(-np.sin(psi))
    ry = float(np.cos(psi))
    f = np.array([fx, fy], dtype=np.float32)
    rvec = np.array([rx, ry], dtype=np.float32)
    near_c = np.array([float(x_center), float(y_row)], dtype=np.float32) + rvec * float(
        right_lane_offset_px
    )
    M = perspective_camera_homography(world_bgr, near_c, f, rvec)
    if M is None:
        return (
            roadkill_left_lane_blend_weight_training(y_row)
            if for_training_labels
            else roadkill_left_lane_blend_weight(y_row)
        )

    cam_s = float(cfg.CAMERA_IMAGE_SIZE)
    bh = bool(cfg.PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY)
    y_rf = float(y_row)
    bl_out = _roadkill_exit_ramp_length_px()
    merge_px = _roadkill_approach_merge_before_core_px()
    w_max = 0.0
    for y_obs in rows:
        wi = _roadkill_left_lane_blend_weight_at_row(y_row, y_obs)
        hc = float(cfg.ROADKILL_DETOUR_CORE_HALF_PX)
        y0 = float(y_obs)
        c_lo = y0 - hc
        c_hi = y0 + hc
        e_lo = c_lo - bl_out
        in_core = c_lo <= y_rf <= c_hi
        in_exit_ramp = e_lo <= y_rf < c_lo
        ox, oy = roadkill_splat_center_bev_xy(dw, y_obs)
        uv = bev_point_to_camera_uv(M, ox, oy)
        if in_core:
            wi_eff = wi
        elif in_exit_ramp:
            wi_eff = wi
        elif y_rf > c_hi:
            vis_w = 0.0
            if uv is not None and _camera_uv_visible(uv[0], uv[1], cam_s, bh):
                vis_w = _splat_visibility_soft_weight((uv[0], uv[1]), cam_s, bh)
            if y_rf >= c_hi + merge_px:
                wi_eff = vis_w
            else:
                t_m = float(np.clip((y_rf - c_hi) / max(merge_px, 1e-6), 0.0, 1.0))
                geom_pull = _raised_cosine01(1.0 - t_m)
                wi_eff = (1.0 - geom_pull) * vis_w + geom_pull * 1.0
        else:
            wi_eff = wi
        w_max = max(w_max, wi_eff)
    return w_max


def lateral_offset_px_avoid_roadkill(
    y_row: float,
    right_lane_center_offset_px: float,
    *,
    for_training_labels: bool = False,
    world_bgr: np.ndarray | None = None,
    dw: DrivingWorld | None = None,
    x_center: float | None = None,
    psi: float | None = None,
    on_main_road: bool = True,
) -> float:
    """
    Signed lateral offset (px) along driver's right: nominal right-lane center, or mirrored left-lane
    when passing the roadkill band (smooth ramps).

    Roadkill splats live on the **main** road only; set ``on_main_road=False`` for off-ramp poses
    (matches ``generate_dataset`` off-ramp crops, which use a fixed right-lane offset).

    If ``for_training_labels`` is true (``generate_dataset`` main road), only obstacles in
    ``ROADKILL_OBSTACLE_Y_PX`` contribute. If false (default, ``simulate``), eval-only rows are included too.

    When ``ROADKILL_DETOUR_ONLY_WHEN_VISIBLE`` and ``world_bgr``, ``dw``, ``x_center``, ``psi`` are set,
    the **approach** follows a soft splat visibility curve in the camera (bottom half if
    ``PERSPECTIVE_INPUT_BOTTOM_HALF_ONLY``), then blends to full ``w`` just before the core; the **exit**
    ramp is geometry-only so the return can finish after the hazard leaves the FOV.
    """
    r = float(right_lane_center_offset_px)
    if not cfg.ROADKILL_ENABLE:
        return r
    if not on_main_road:
        return r
    use_vis = bool(getattr(cfg, "ROADKILL_DETOUR_ONLY_WHEN_VISIBLE", True))
    if (
        use_vis
        and world_bgr is not None
        and dw is not None
        and x_center is not None
        and psi is not None
    ):
        w = _roadkill_blend_weight_visible_gated(
            float(y_row),
            world_bgr,
            dw,
            float(x_center),
            float(psi),
            r,
            for_training_labels=for_training_labels,
        )
    elif for_training_labels:
        w = roadkill_left_lane_blend_weight_training(y_row)
    else:
        w = roadkill_left_lane_blend_weight(y_row)
    return (1.0 - w) * r + w * (-r)


def _bezier_quadratic_xy_d1_d2(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    u: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadratic Bézier ``B(u)`` and parametric derivatives ``B'(u)``, ``B''(u)`` (column vectors)."""
    u = float(np.clip(u, 0.0, 1.0))
    omu = 1.0 - u
    B = omu**2 * p0 + 2.0 * omu * u * p1 + u**2 * p2
    d1 = -2.0 * omu * p0 + 2.0 * (1.0 - 2.0 * u) * p1 + 2.0 * u * p2
    d2 = 2.0 * p0 - 4.0 * p1 + 2.0 * p2
    return B, d1, d2


def _bezier_quadratic_xy_batch(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """``B(u)`` for each entry in 1-D ``u``; returns ``(len(u), 2)``."""
    u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
    omu = 1.0 - u
    return (omu * omu) * p0 + (2.0 * omu * u) * p1 + (u * u) * p2


class DrivingWorld:
    """
    Top-down raster map: cubic-spline centerline from ``SPLINE_X_DELTAS_BOTTOM_TO_TOP``, two-lane road,
    dashed center marking, optional off-ramps (``OFFRAMP_*``; primary branch + ``OFFRAMP_EXTRA_BRANCH_Y_PX``),
    optional right-lane roadkill
    obstacle(s) (``ROADKILL_*``; map shows training + eval-only rows). Curvature of the **main** spline drives
    dataset steering labels after global scaling; dataset lateral uses only ``ROADKILL_OBSTACLE_Y_PX``.
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

    def offramp_branch_y_pxs(self) -> tuple[float, ...]:
        """
        Branch rows on the main centerline (BEV ``y``, descending = first merge when driving from
        the bottom of the map).
        """
        if not cfg.OFFRAMP_ENABLE:
            return tuple()
        ys: list[float] = []
        y_main = float(np.clip(cfg.OFFRAMP_BRANCH_Y_FRAC, 0.0, 1.0)) * float(
            self.size
        )
        if y_main > 0.5 * float(self.size):
            ys.append(float(y_main))
        seen = set(ys)
        for yx in getattr(cfg, "OFFRAMP_EXTRA_BRANCH_Y_PX", ()):
            yf = float(np.clip(float(yx), 1.0, float(self.size) - 1.0))
            if yf not in seen:
                ys.append(yf)
                seen.add(yf)
        ys.sort(reverse=True)
        return tuple(ys)

    def offramp_num(self) -> int:
        return len(self.offramp_branch_y_pxs())

    def _offramp_bezier_scalar_params_at(
        self, y_branch: float
    ) -> tuple[float, float, float]:
        """``(tangent_ctrl_px, end_dx_px, end_dy_px)`` for branch row ``y_branch`` (defaults + optional override)."""
        L = float(cfg.OFFRAMP_TANGENT_CTRL_PX)
        edx = float(cfg.OFFRAMP_END_DX_PX)
        edy = float(cfg.OFFRAMP_END_DY_PX)
        y0 = float(np.clip(float(y_branch), 1.0, float(self.size) - 1.0))
        ovr = getattr(cfg, "OFFRAMP_BRANCH_BEZIER_OVERRIDE_BY_Y_PX", None)
        if ovr:
            for yk, triple in ovr.items():
                if abs(float(yk) - y0) <= 0.51:
                    L = float(triple[0])
                    edx = float(triple[1])
                    edy = float(triple[2])
                    break
        return L, edx, edy

    def _offramp_bezier_controls_at(
        self, y_branch: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Bézier ``p0, p1, p2`` for one branch row ``y_branch`` on the main centerline."""
        if not cfg.OFFRAMP_ENABLE:
            return None
        y0 = float(np.clip(y_branch, 1.0, float(self.size) - 1.0))
        p0 = np.array([float(self.cs(y0)), y0], dtype=np.float64)
        dxdy = float(self.cs(y0, nu=1))
        t = np.array([-dxdy, -1.0], dtype=np.float64)
        t_norm = float(np.linalg.norm(t))
        if t_norm <= 1e-9:
            t = np.array([0.0, -1.0], dtype=np.float64)
        else:
            t /= t_norm
        L, edx, edy = self._offramp_bezier_scalar_params_at(y0)
        p1 = p0 + L * t
        p2 = p0 + np.array([edx, edy], dtype=np.float64)
        return p0, p1, p2

    def _offramp_bezier_controls(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Controls for ramp index ``0`` (legacy), or ``None`` if there are no branches.
        """
        branches = self.offramp_branch_y_pxs()
        if not branches:
            return None
        return self._offramp_bezier_controls_at(branches[0])

    def _offramp_bezier_polyline_int(
        self, ctrl: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Dense integer polyline for one Bézier."""
        p0, p1, p2 = ctrl
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

    def offramp_bezier_evolution(
        self, u: float, ramp_id: int = 0
    ) -> tuple[np.ndarray, tuple[float, float], float] | None:
        """
        Position on off-ramp ``ramp_id``, unit forward ``(fx, fy)`` in BEV (+x right, +y down), signed κ.

        Curvature uses the standard parametric formula in image coordinates (matches ``generate_dataset``
        scaling when combined with main-road κ in one global ``max|κ|``).
        """
        branches = self.offramp_branch_y_pxs()
        if ramp_id < 0 or ramp_id >= len(branches):
            return None
        ctrl = self._offramp_bezier_controls_at(branches[ramp_id])
        if ctrl is None:
            return None
        p0, p1, p2 = ctrl
        B, d1, d2 = _bezier_quadratic_xy_d1_d2(p0, p1, p2, u)
        xp, yp = float(d1[0]), float(d1[1])
        xpp, ypp = float(d2[0]), float(d2[1])
        vnorm = float(np.hypot(xp, yp))
        if vnorm <= cfg.CURVATURE_DENOM_EPS:
            return B.astype(np.float64), (0.0, -1.0), 0.0
        fx, fy = xp / vnorm, yp / vnorm
        denom = vnorm**3
        if denom < cfg.CURVATURE_DENOM_EPS:
            kappa = 0.0
        else:
            kappa = (xp * ypp - yp * xpp) / denom
        return B.astype(np.float64), (float(fx), float(fy)), float(kappa)

    def offramp_u_samples_uniform_arc_length(
        self,
        n: int,
        *,
        u_lo: float = 0.05,
        u_hi: float = 0.95,
        n_fine: int | None = None,
        ramp_id: int = 0,
    ) -> np.ndarray:
        """
        ``n`` values of Bézier parameter ``u``, approximately **uniform in Euclidean arc length** between
        ``u_lo`` and ``u_hi`` (same inset band as legacy ``linspace`` sampling).

        Matches the spirit of ``dataset_split._y_samples_uniform_arc_length`` for the main spline.
        """
        branches = self.offramp_branch_y_pxs()
        if ramp_id < 0 or ramp_id >= len(branches) or n <= 0:
            return np.zeros((0,), dtype=np.float64)
        ctrl = self._offramp_bezier_controls_at(branches[ramp_id])
        if ctrl is None:
            return np.zeros((0,), dtype=np.float64)
        p0, p1, p2 = ctrl
        u_lo = float(np.clip(u_lo, 0.0, 1.0))
        u_hi = float(np.clip(u_hi, 0.0, 1.0))
        if u_hi <= u_lo + 1e-12:
            return np.full((n,), u_lo, dtype=np.float64)
        nf = int(n_fine) if n_fine is not None else max(500, int(n) * 100)
        u_fine = np.linspace(u_lo, u_hi, nf, dtype=np.float64)
        B = _bezier_quadratic_xy_batch(p0, p1, p2, u_fine)
        seg = np.sqrt(np.sum(np.diff(B, axis=0) ** 2, axis=1))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        s_tot = float(s[-1])
        if (not np.isfinite(s_tot)) or s_tot <= 0.0:
            return np.linspace(u_lo, u_hi, n, dtype=np.float64)
        if n == 1:
            return np.array(
                [float(np.interp(0.5 * s_tot, s, u_fine))], dtype=np.float64
            )
        targets = np.linspace(0.0, s_tot, n, dtype=np.float64)
        return np.interp(targets, s, u_fine).astype(np.float64)

    def offramp_arc_length_px(
        self,
        u_lo: float = 0.05,
        u_hi: float = 0.95,
        *,
        n_fine: int = 2000,
        ramp_id: int = 0,
    ) -> float:
        """Pixel arc length along one off-ramp Bézier between ``u_lo`` and ``u_hi`` (dataset inset band)."""
        branches = self.offramp_branch_y_pxs()
        if ramp_id < 0 or ramp_id >= len(branches):
            return 0.0
        ctrl = self._offramp_bezier_controls_at(branches[ramp_id])
        if ctrl is None:
            return 0.0
        u_lo = float(np.clip(u_lo, 0.0, 1.0))
        u_hi = float(np.clip(u_hi, 0.0, 1.0))
        if u_hi <= u_lo + 1e-12:
            return 0.0
        p0, p1, p2 = ctrl
        u_fine = np.linspace(u_lo, u_hi, int(n_fine), dtype=np.float64)
        B = _bezier_quadratic_xy_batch(p0, p1, p2, u_fine)
        return float(np.sum(np.sqrt(np.sum(np.diff(B, axis=0) ** 2, axis=1))))

    def offramp_total_arc_length_px(
        self,
        u_lo: float = 0.05,
        u_hi: float = 0.95,
        *,
        n_fine: int = 2000,
    ) -> float:
        """Sum of inset-band arc lengths over all off-ramps (dataset spacing heuristic)."""
        return float(
            sum(
                self.offramp_arc_length_px(
                    u_lo, u_hi, n_fine=n_fine, ramp_id=rid
                )
                for rid in range(self.offramp_num())
            )
        )

    def offramp_max_abs_curvature(self, n: int = 128) -> float:
        """Max ``|κ|`` over all Béziers (for ``SIM_YAW_RATE_GAIN`` vs dataset κ scaling)."""
        if self.offramp_num() <= 0:
            return 0.0
        ugrid = np.linspace(0.02, 0.98, int(n), dtype=np.float64)
        m = 0.0
        for rid in range(self.offramp_num()):
            for u in ugrid:
                ev = self.offramp_bezier_evolution(float(u), rid)
                if ev is None:
                    continue
                m = max(m, abs(float(ev[2])))
        return float(m)

    def offramp_step_arc_px(
        self, u: float, ds: float, ramp_id: int = 0
    ) -> tuple[float, float, float] | None:
        """
        Advance the Bézier parameter by arc length ``ds`` (pixels in BEV).

        Returns ``(u_new, x, y)`` on the ramp centerline, or ``None`` if the off-ramp is disabled.
        """
        branches = self.offramp_branch_y_pxs()
        if ramp_id < 0 or ramp_id >= len(branches):
            return None
        ctrl = self._offramp_bezier_controls_at(branches[ramp_id])
        if ctrl is None:
            return None
        p0, p1, p2 = ctrl
        u = float(np.clip(u, 0.0, 1.0))
        _, d1, _ = _bezier_quadratic_xy_d1_d2(p0, p1, p2, u)
        vnorm = float(np.hypot(float(d1[0]), float(d1[1])))
        if vnorm < cfg.CURVATURE_DENOM_EPS:
            return None
        u_new = float(np.clip(u + float(ds) / vnorm, 0.0, 1.0))
        B, _, _ = _bezier_quadratic_xy_d1_d2(p0, p1, p2, u_new)
        return u_new, float(B[0]), float(B[1])

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

        for yb in self.offramp_branch_y_pxs():
            ctrl = self._offramp_bezier_controls_at(yb)
            if ctrl is None:
                continue
            off = self._offramp_bezier_polyline_int(ctrl)
            if len(off) >= 2:
                self._draw_road_polyline(
                    world, off, lane_px=lane_px, dash_len=dash_len, dash_gap=dash_gap
                )
        self._draw_roadkill_obstacles(world)
        # Orientation: top of image is y=0 (small y); blue stripe marks that edge for debugging.
        cv2.line(
            world,
            (0, 0),
            (self.size - 1, 0),
            (255, 0, 0),
            thickness=4,
        )
        return world

    def _draw_roadkill_obstacles(self, world: np.ndarray) -> None:
        """Flattened red splat(s) in the right lane at each ``ROADKILL_OBSTACLE_Y_PX`` row."""
        if not cfg.ROADKILL_ENABLE:
            return
        for y_row in _roadkill_obstacle_y_rows_all():
            self._draw_roadkill_obstacle_at(world, float(y_row))

    def _draw_roadkill_obstacle_at(self, world: np.ndarray, y0: float) -> None:
        """Single splat in the right lane (pavement) at BEV row ``y0``."""
        size = int(self.size)
        y0i = int(np.clip(int(round(y0)), 4, size - 5))
        xc = float(self.cs(float(y0)))
        dxdY = float(self.cs(float(y0), nu=1))
        norm = float(np.hypot(dxdY, 1.0))
        if norm < 1e-9:
            return
        fx, fy = -dxdY / norm, -1.0 / norm
        rx, ry = -fy, fx
        lane_off = float(
            cfg.LANE_WIDTH_METERS * cfg.DATASET_RIGHT_LANE_LATERAL_FRAC * self.px_per_m
        )
        cx = int(np.clip(int(round(xc + rx * lane_off)), 4, size - 5))
        cy = int(np.clip(int(round(float(y0i) + ry * lane_off)), 4, size - 5))
        lane_w_px = float(cfg.LANE_WIDTH_METERS * self.px_per_m)
        b_cap = max(2.0, float(cfg.ROADKILL_ACROSS_MAX_FRAC_OF_LANE) * lane_w_px)
        a_cap = max(4.0, float(cfg.ROADKILL_ALONG_MAX_LANE_WIDTHS) * lane_w_px)
        b = int(
            round(
                max(2.0, min(float(cfg.ROADKILL_ACROSS_ROAD_HALF_PX), b_cap))
            )
        )
        a = int(
            round(
                max(4.0, min(float(cfg.ROADKILL_ALONG_ROAD_HALF_PX), a_cap))
            )
        )
        ang = float(np.degrees(np.arctan2(fy, fx)))
        # Red splat (high contrast on gray pavement)
        cv2.ellipse(
            world,
            (cx, cy),
            (a, max(2, b)),
            ang,
            0,
            360,
            (55, 55, 165),
            -1,
            cv2.LINE_AA,
        )
        # Brighter smear along forward direction
        cx2 = int(cx - 0.32 * a * fx)
        cy2 = int(cy - 0.32 * a * fy)
        cv2.ellipse(
            world,
            (cx2, cy2),
            (max(3, a // 2), max(2, b // 2)),
            ang,
            0,
            360,
            (85, 85, 235),
            -1,
            cv2.LINE_AA,
        )
        # Darker streak (smear core)
        p0 = (
            int(cx - 0.5 * a * fx + 2.0 * rx),
            int(cy - 0.5 * a * fy + 2.0 * ry),
        )
        p1 = (
            int(cx + 0.5 * a * fx + 2.0 * rx),
            int(cy + 0.5 * a * fy + 2.0 * ry),
        )
        streak_t = max(1, min(3, int(round(0.35 * lane_w_px))))
        cv2.line(world, p0, p1, (35, 35, 110), streak_t, cv2.LINE_AA)
        # Highlight glint
        cv2.ellipse(
            world,
            (cx - int(0.15 * a * fx), cy - int(0.15 * a * fy)),
            (max(2, a // 5), max(2, b // 3)),
            ang + 12.0,
            0,
            360,
            (130, 130, 255),
            -1,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    world = DrivingWorld()
    plt.imshow(cv2.cvtColor(world.image, cv2.COLOR_BGR2RGB))
    plt.show()
