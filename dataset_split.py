"""
Train/test road row sampling (shared by ``generate_dataset``, ``config._compute_sim_yaw_rate_gain``, ``simulate``).

No import-time side effects (no global RNG seeding).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _y_samples_uniform_arc_length(
    cs: Any,
    y_top: float,
    y_bottom: float,
    n: int,
    *,
    n_fine: int | None = None,
) -> np.ndarray:
    """
    ``n`` sample row indices ``y`` along the centerline ``x = cs(y)``, from ``y_bottom`` (map bottom,
    larger ``y``) toward ``y_top`` (smaller ``y``), **uniform in Euclidean arc length** along the polyline.

    ``y_bottom`` and ``y_top`` need not be ordered; the segment is the inclusive range between them.
    """
    if n <= 0:
        return np.array([], dtype=np.float64)
    a = float(min(y_top, y_bottom))
    b = float(max(y_top, y_bottom))
    y_hi = b
    y_lo = a
    if n_fine is None:
        n_fine = max(2000, n * 150)
    y_fine = np.linspace(y_hi, y_lo, int(n_fine))
    x_fine = cs(y_fine)
    dy = np.diff(y_fine)
    dx = np.diff(x_fine)
    seg_len = np.sqrt(dx * dx + dy * dy)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    s_tot = float(s[-1])
    if (not np.isfinite(s_tot)) or s_tot <= 0.0:
        return np.linspace(y_hi, y_lo, n, dtype=np.float64)
    if n == 1:
        return np.array([0.5 * (y_hi + y_lo)], dtype=np.float64)
    targets = np.linspace(0.0, s_tot, n)
    y_samples = np.interp(targets, s, y_fine).astype(np.float64)
    return y_samples


def dataset_train_test_y(
    num_train: int,
    num_test: int,
    size: int,
    margin: int,
    *,
    mix_train_test_geography: bool,
    seed: int,
    uniform_along_road: bool = False,
    road_cs: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Road centerline row indices ``y`` (pixels; BEV +y is downward).

    Args:
        num_train: Number of train clean samples (before optional perturbed duplicates).
        num_test: Number of test clean samples.
        size: Map side length in pixels.
        margin: Inset from top/bottom edges (``DATASET_MAP_MARGIN``).
        mix_train_test_geography: If false, train spans only the bottom half
            ``(size//2 + 1 .. size - margin]`` and test only the top half ``[margin .. size//2]``.
            If true, ``num_train + num_test`` positions along the full usable span
            and randomly split (reproducible ``seed``); both splits see top and bottom curvature.
        seed: RNG seed when ``mix_train_test_geography`` is true.
        uniform_along_road: If true, space samples by **arc length** along ``x = cs(y)``; if false,
            **even spacing in ``y``** (legacy ``linspace``).
        road_cs: Centerline spline ``x(y)`` (``DrivingWorld.cs``). If ``uniform_along_road`` and
            this is ``None``, a ``DrivingWorld`` is constructed to obtain ``cs``.

    Returns:
        ``(train_y, test_y)`` as ``float64`` 1-D arrays (unordered if mixed).
    """
    cs = road_cs
    if uniform_along_road and cs is None:
        from generate_world import DrivingWorld

        cs = DrivingWorld().cs

    if not mix_train_test_geography:
        half = size // 2
        y_tr_hi = float(size - margin)
        y_tr_lo = float(half) + 1.0
        y_te_hi = float(half)
        y_te_lo = float(margin)
        if uniform_along_road:
            train_y = _y_samples_uniform_arc_length(cs, y_tr_lo, y_tr_hi, int(num_train))
            test_y = _y_samples_uniform_arc_length(cs, y_te_lo, y_te_hi, int(num_test))
        else:
            train_y = np.linspace(y_tr_hi, y_tr_lo, int(num_train), dtype=np.float64)
            test_y = np.linspace(y_te_hi, y_te_lo, int(num_test), dtype=np.float64)
        return train_y, test_y

    n_pool = int(num_train) + int(num_test)
    y_pool_hi = float(size - margin)
    y_pool_lo = float(margin)
    if uniform_along_road:
        y_pool = _y_samples_uniform_arc_length(cs, y_pool_lo, y_pool_hi, n_pool)
    else:
        y_pool = np.linspace(y_pool_hi, y_pool_lo, n_pool, dtype=np.float64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pool)
    return y_pool[perm[: int(num_train)]], y_pool[perm[int(num_train) :]]


def centerline_arc_length_between_rows(
    cs,
    y_a: float,
    y_b: float,
    *,
    n_fine: int | None = None,
) -> float:
    """
    Euclidean arc length (pixels) along ``x = cs(y)`` between two rows; ``y_a``, ``y_b`` order-free.
    """
    lo = float(min(y_a, y_b))
    hi = float(max(y_a, y_b))
    if hi <= lo + 1e-12:
        return 0.0
    nf = int(n_fine) if n_fine is not None else max(2000, 500)
    y_fine = np.linspace(lo, hi, nf, dtype=np.float64)
    x_fine = cs(y_fine)
    return float(np.sum(np.sqrt(np.diff(x_fine) ** 2 + np.diff(y_fine) ** 2)))


def offramp_clean_counts_matching_main_spacing(
    L_ramp_px: float,
    num_train: int,
    num_test: int,
    size: int,
    margin: float,
    *,
    mix_train_test_geography: bool,
    cs,
    cap_train: int,
    cap_test: int,
    match_spacing: bool,
) -> tuple[int, int]:
    """
    Off-ramp clean frame counts for train / test.

    When ``match_spacing`` is false, returns ``cap_train``, ``cap_test``. When true, each count is
    ``min(cap, max(1, round(L_ramp / delta)))`` with ``delta`` the mean centerline spacing of the
    corresponding main-road clean grid (or full-road pool spacing when geography is mixed).
    """
    ct = max(0, int(cap_train))
    ce = max(0, int(cap_test))
    if L_ramp_px <= 0.0 or not np.isfinite(L_ramp_px):
        return 0, 0
    if not match_spacing:
        return ct, ce

    if not mix_train_test_geography:
        half = size // 2
        y_tr_hi = float(size - margin)
        y_tr_lo = float(half) + 1.0
        y_te_hi = float(half)
        y_te_lo = float(margin)
        L_tr = centerline_arc_length_between_rows(cs, y_tr_lo, y_tr_hi)
        L_te = centerline_arc_length_between_rows(cs, y_te_lo, y_te_hi)
        d_tr = L_tr / max(int(num_train) - 1, 1)
        d_te = L_te / max(int(num_test) - 1, 1)
        n_tr = max(1, int(round(L_ramp_px / d_tr)))
        n_te = max(1, int(round(L_ramp_px / d_te)))
        return (min(ct, n_tr) if ct > 0 else 0, min(ce, n_te) if ce > 0 else 0)

    y_pool_hi = float(size - margin)
    y_pool_lo = float(margin)
    L_full = centerline_arc_length_between_rows(cs, y_pool_lo, y_pool_hi)
    n_pool = int(num_train) + int(num_test)
    d_pool = L_full / max(n_pool - 1, 1)
    n_r = max(1, int(round(L_ramp_px / d_pool)))
    return (min(ct, n_r) if ct > 0 else 0, min(ce, n_r) if ce > 0 else 0)
