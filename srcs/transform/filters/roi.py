from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from srcs.cli.Transformation import TransformConfig
except Exception:
    import sys
    from pathlib import Path

    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from srcs.cli.Transformation import TransformConfig


def apply_roi_filter(
    rgb: np.ndarray, contour: Optional[np.ndarray], cfg: TransformConfig
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    if contour is None:
        return rgb, None, None

    x, y, w, h = cv2.boundingRect(contour)
    roi_img = rgb[y : y + h, x : x + w]

    # Resize to standard size (padding to keep aspect ratio)
    H, W = cfg.roi_size
    if roi_img.size == 0:
        return rgb, None, None

    # Letterbox to target size
    scale = min(W / max(w, 1), H / max(h, 1))
    nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
    resized = cv2.resize(roi_img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), dtype=resized.dtype)
    oy, ox = (H - nh) // 2, (W - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized

    # ROI visualization on original
    vis = rgb.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return canvas, vis, (x, y, w, h)
