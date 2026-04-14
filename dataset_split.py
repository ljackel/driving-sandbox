"""
Train/test road row sampling (shared by ``generate_dataset``, ``config._compute_sim_yaw_rate_gain``, ``simulate``).

No import-time side effects (no global RNG seeding).
"""
from __future__ import annotations

import numpy as np


def dataset_train_test_y(
    num_train: int,
    num_test: int,
    size: int,
    margin: int,
    *,
    mix_train_test_geography: bool,
    seed: int,
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
            If true, ``num_train + num_test`` positions are equally spaced along the full usable span
            and randomly split (reproducible ``seed``); both splits see top and bottom curvature.

    Returns:
        ``(train_y, test_y)`` as ``float64`` 1-D arrays (unordered if mixed).
    """
    if not mix_train_test_geography:
        half = size // 2
        train_y = np.linspace(
            float(size - margin),
            float(half) + 1.0,
            int(num_train),
            dtype=np.float64,
        )
        test_y = np.linspace(
            float(half),
            float(margin),
            int(num_test),
            dtype=np.float64,
        )
        return train_y, test_y

    n_pool = int(num_train) + int(num_test)
    y_pool = np.linspace(
        float(size - margin),
        float(margin),
        n_pool,
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pool)
    return y_pool[perm[: int(num_train)]], y_pool[perm[int(num_train) :]]
