"""
BEV → square camera perspective warp shared by ``generate_dataset`` and ``simulate``.
"""

from __future__ import annotations

import cv2
import numpy as np

import config as cfg

# Try shorter look-ahead if the far line would leave the map (e.g. poses near the top edge).
_FAR_SCALE_STEPS = (1.0, 0.75, 0.6, 0.45, 0.33, 0.25, 0.18, 0.12)


def perspective_camera_view(
    world_bgr: np.ndarray,
    near_c: np.ndarray,
    f: np.ndarray,
    r: np.ndarray,
) -> np.ndarray | None:
    """
    Warp a forward-facing ``CAMERA_IMAGE_SIZE`` square from the BEV map.

    Uses an adaptive far distance (up to ``PERSPECTIVE_FAR_OFFSET_PX``) so the source quad stays
    in bounds. Keep ``PERSPECTIVE_FAR_HALF_WIDTH`` > ``PERSPECTIVE_NEAR_HALF_WIDTH`` so the warp
    does not over-magnify the top of the image (otherwise the road looks wider at the horizon).

    Args:
        world_bgr: Full map (BGR), shape ``(h, w, 3)``.
        near_c: Camera reference in pixel coords, shape ``(2,)`` float32.
        f: Unit forward in image plane, shape ``(2,)`` float32.
        r: Unit right, shape ``(2,)`` float32.

    Returns:
        BGR square or ``None`` if no scale keeps the quad in the map.
    """
    h, w = world_bgr.shape[:2]
    mrg = float(cfg.PERSPECTIVE_SRC_MARGIN_PX)
    cam_s = float(cfg.CAMERA_IMAGE_SIZE)
    dst = np.float32([[0, 0], [cam_s, 0], [cam_s, cam_s], [0, cam_s]])
    wh = cfg.CAMERA_IMAGE_SIZE

    far_w = float(cfg.PERSPECTIVE_FAR_HALF_WIDTH)
    near_w = float(cfg.PERSPECTIVE_NEAR_HALF_WIDTH)
    far_target = float(cfg.PERSPECTIVE_FAR_OFFSET_PX)

    for scale in _FAR_SCALE_STEPS:
        far_off = far_target * scale
        far_c = near_c + f * float(far_off)
        tl = far_c - r * far_w
        tr = far_c + r * far_w
        br = near_c + r * near_w
        bl = near_c - r * near_w
        src = np.float32([tl, tr, br, bl])
        if (
            (src[:, 0] < -mrg).any()
            or (src[:, 0] >= w + mrg).any()
            or (src[:, 1] < -mrg).any()
            or (src[:, 1] >= h + mrg).any()
        ):
            continue
        m = cv2.getPerspectiveTransform(src, dst)
        view = cv2.warpPerspective(
            world_bgr,
            m,
            (wh, wh),
            borderMode=cv2.BORDER_REPLICATE,
        )
        return view
    return None
