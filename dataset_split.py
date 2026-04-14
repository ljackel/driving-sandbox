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


def train_bev_y_band(size: int, margin: int) -> tuple[float, float]:
    """
    Train split uses only the **bottom** BEV half (+y down): rows in ``(size//2 + 1 .. size - margin]``.
    """
    half = int(size) // 2
    return float(half) + 1.0, float(size - margin)


def test_bev_y_band(size: int, margin: int) -> tuple[float, float]:
    """Test split uses the **top** BEV half: rows in ``[margin .. size//2]``."""
    half = int(size) // 2
    return float(margin), float(half)


def pose_row_in_train_bev_band(py: float, size: int, margin: int) -> bool:
    """True if BEV row ``py`` lies in the train (bottom-half) band."""
    y_lo, y_hi = train_bev_y_band(size, margin)
    return y_lo <= float(py) <= y_hi


def main_train_mean_arc_spacing_px(
    cs: Any,
    size: int,
    margin: int,
    num_main_train: int,
) -> float:
    """Mean Euclidean spacing (px) between main-road train samples in the bottom-half band."""
    y_tr_lo, y_tr_hi = train_bev_y_band(size, margin)
    L_tr = centerline_arc_length_between_rows(cs, y_tr_lo, y_tr_hi)
    return float(L_tr) / max(int(num_main_train) - 1, 1)


def _offramp_longest_u_interval_in_train_band(
    dw: Any,
    rid: int,
    size: int,
    margin: int,
    u_lo: float = 0.05,
    u_hi: float = 0.95,
    *,
    nf: int = 400,
) -> tuple[float, float] | None:
    """
    Longest contiguous ``u``-subinterval of ``[u_lo, u_hi]`` where the Bézier center ``y`` lies in
    the train (bottom) BEV band; ``None`` if none.
    """
    yb_lo, yb_hi = train_bev_y_band(size, margin)
    u_f = np.linspace(float(u_lo), float(u_hi), int(nf), dtype=np.float64)
    ok = np.zeros(int(nf), dtype=bool)
    for i, u in enumerate(u_f):
        ev = dw.offramp_bezier_evolution(float(u), rid)
        if ev is None:
            continue
        py = float(ev[0][1])
        ok[i] = yb_lo <= py <= yb_hi
    if not np.any(ok):
        return None
    padded = np.concatenate([[False], ok, [False]])
    step = np.diff(padded.astype(np.int8))
    starts = np.where(step == 1)[0]
    ends = np.where(step == -1)[0]
    lengths = ends - starts
    j = int(np.argmax(lengths))
    i0 = int(starts[j])
    i1 = int(ends[j]) - 1
    return float(u_f[i0]), float(u_f[i1])


def _fit_counts_to_cap(n_ideal: list[int], cap_total: int) -> list[int]:
    """Nonnegative integers summing to at most ``cap_total``, at least one when ideal>0 where possible."""
    n_br = len(n_ideal)
    total = int(sum(n_ideal))
    if cap_total <= 0 or total == 0:
        return [0] * n_br
    n_per = [int(x) for x in n_ideal]
    if sum(n_per) <= cap_total:
        return n_per
    # scale down; keep at least 1 for any ramp that had positive ideal count
    scl = cap_total / float(sum(n_per))
    n_per = [max(0 if n_ideal[i] == 0 else 1, int(round(n_per[i] * scl))) for i in range(n_br)]
    while sum(n_per) > cap_total:
        order = sorted(
            range(n_br),
            key=lambda i: (n_per[i], n_ideal[i]),
            reverse=True,
        )
        reduced = False
        for i in order:
            if n_per[i] > (1 if n_ideal[i] > 0 else 0):
                n_per[i] -= 1
                reduced = True
                break
        if not reduced:
            break
    return n_per


def offramp_train_u_rid_pairs_main_spacing(
    dw: Any,
    size: int,
    margin: int,
    n_br: int,
    num_main_train: int,
    cap_total: int,
    *,
    uniform_along_ramp: bool,
) -> list[tuple[float, int]]:
    """
    ``(u, ramp_id)`` in round-robin order: on each ramp, ``u`` is **uniform in arc length** (or linear
    in ``u``) along the longest in-train segment, with step length matching the main-road train mean
    spacing (same ``δ`` as ``main_train_mean_arc_spacing_px``). Total count is capped by ``cap_total``.
    """
    if cap_total <= 0 or n_br <= 0:
        return []
    delta = main_train_mean_arc_spacing_px(dw.cs, size, margin, num_main_train)
    if not np.isfinite(delta) or delta <= 1e-12:
        delta = 1.0
    n_ideal: list[int] = []
    u_seg: list[tuple[float, float] | None] = []
    for rid in range(n_br):
        seg = _offramp_longest_u_interval_in_train_band(dw, rid, size, margin)
        if seg is None:
            u_seg.append(None)
            n_ideal.append(0)
            continue
        ua, ub = seg
        u_seg.append((ua, ub))
        ell = float(dw.offramp_arc_length_px(ua, ub, ramp_id=rid))
        n_ideal.append(max(1, int(round(ell / delta))) if ell > 1e-9 else 1)
    if sum(n_ideal) == 0:
        return []
    n_per = _fit_counts_to_cap(n_ideal, cap_total)
    u_arrays: list[np.ndarray] = []
    for rid in range(n_br):
        if n_per[rid] <= 0 or u_seg[rid] is None:
            u_arrays.append(np.zeros((0,), dtype=np.float64))
            continue
        ua, ub = u_seg[rid]
        if uniform_along_ramp:
            arr = dw.offramp_u_samples_uniform_arc_length(
                n_per[rid], u_lo=ua, u_hi=ub, ramp_id=rid
            )
        else:
            arr = np.linspace(ua, ub, n_per[rid], dtype=np.float64)
        u_arrays.append(arr)
    pairs: list[tuple[float, int]] = []
    idx = [0] * n_br
    while True:
        moved = False
        for rid in range(n_br):
            if idx[rid] < len(u_arrays[rid]):
                pairs.append((float(u_arrays[rid][idx[rid]]), rid))
                idx[rid] += 1
                moved = True
        if not moved:
            break
    return pairs


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

    Train samples always lie in the **bottom** half ``(size//2 + 1 .. size - margin]``; test samples
    always in the **top** half ``[margin .. size//2]``. ``mix_train_test_geography`` only shuffles
    order within each band (reproducible ``seed``), so training never uses the upper half of the map.

    Args:
        num_train: Number of train clean samples (before optional perturbed duplicates).
        num_test: Number of test clean samples.
        size: Map side length in pixels.
        margin: Inset from top/bottom edges (``DATASET_MAP_MARGIN``).
        mix_train_test_geography: If true, shuffle train ``y`` and test ``y`` separately (same bands).
        seed: RNG seed when ``mix_train_test_geography`` is true.
        uniform_along_road: If true, space samples by **arc length** along ``x = cs(y)``; if false,
            **even spacing in ``y``** (legacy ``linspace``).
        road_cs: Centerline spline ``x(y)`` (``DrivingWorld.cs``). If ``uniform_along_road`` and
            this is ``None``, a ``DrivingWorld`` is constructed to obtain ``cs``.

    Returns:
        ``(train_y, test_y)`` as ``float64`` 1-D arrays (order may be shuffled when mixed).
    """
    cs = road_cs
    if uniform_along_road and cs is None:
        from generate_world import DrivingWorld

        cs = DrivingWorld().cs

    y_tr_lo, y_tr_hi = train_bev_y_band(size, margin)
    y_te_lo, y_te_hi = test_bev_y_band(size, margin)
    if uniform_along_road:
        train_y = _y_samples_uniform_arc_length(cs, y_tr_lo, y_tr_hi, int(num_train))
        test_y = _y_samples_uniform_arc_length(cs, y_te_lo, y_te_hi, int(num_test))
    else:
        train_y = np.linspace(y_tr_hi, y_tr_lo, int(num_train), dtype=np.float64)
        test_y = np.linspace(y_te_hi, y_te_lo, int(num_test), dtype=np.float64)
    if mix_train_test_geography:
        rng = np.random.default_rng(seed)
        train_y = rng.permutation(train_y)
        test_y = rng.permutation(test_y)
    return train_y, test_y


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
    ``min(cap, max(1, round(L_ramp_sum / delta)))`` with ``delta`` the mean main-road **train**-band
    spacing and ``L_ramp_sum`` the sum of inset-band arc lengths over all ramps (budget for interleaved output).
    """
    ct = max(0, int(cap_train))
    ce = max(0, int(cap_test))
    if L_ramp_px <= 0.0 or not np.isfinite(L_ramp_px):
        return 0, 0
    if not match_spacing:
        return ct, ce

    y_tr_lo, y_tr_hi = train_bev_y_band(size, margin)
    y_te_lo, y_te_hi = test_bev_y_band(size, margin)
    L_tr = centerline_arc_length_between_rows(cs, y_tr_lo, y_tr_hi)
    L_te = centerline_arc_length_between_rows(cs, y_te_lo, y_te_hi)
    d_tr = L_tr / max(int(num_train) - 1, 1)
    d_te = L_te / max(int(num_test) - 1, 1)
    n_tr = max(1, int(round(L_ramp_px / d_tr)))
    n_te = max(1, int(round(L_ramp_px / d_te)))
    _ = mix_train_test_geography  # train/test bands fixed; mix only shuffles row order in grids
    return (min(ct, n_tr) if ct > 0 else 0, min(ce, n_te) if ce > 0 else 0)
