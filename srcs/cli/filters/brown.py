"""
Brown/Disease spots detection filter for image transformation pipeline.
Detects and visualizes diseased areas in leaf tissue.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from ..Transformation import TransformConfig
except ImportError:
    # Fallback for direct script execution
    import sys
    from pathlib import Path

    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from Transformation import TransformConfig


def apply_brown_filter(
    rgb: np.ndarray, mask: Optional[np.ndarray], cfg: TransformConfig
) -> Tuple[np.ndarray, float, int]:
    """
    Detect brown/diseased areas in leaf.

    Args:
        rgb: Input RGB image
        mask: Leaf mask
        cfg: Transform configuration

    Returns:
        Tuple of (visualization, brown_percentage, brown_count)
    """
    if mask is None:
        return rgb, 0.0, 0

    # Create leaf-only region
    if mask.ndim == 2:
        leaf_mask = mask > 0
    else:
        leaf_mask = mask[..., 0] > 0

    if cfg.use_lab_brown:
        # LAB-based detection
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        brown_regions = (a >= cfg.lab_a_min) & (b >= cfg.lab_b_min) & leaf_mask
    else:
        # HSV-based detection
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        lo, hi = cfg.brown_hue_range
        brown_regions = (
            (h >= lo)
            & (h <= hi)
            & (s >= cfg.brown_s_min)
            & (v <= cfg.brown_v_max)
            & leaf_mask
        )

    # Morphological operations to clean up noise
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.brown_morph_kernel, cfg.brown_morph_kernel),
    )
    brown_clean = cv2.morphologyEx(
        brown_regions.astype(np.uint8) * 255, cv2.MORPH_OPEN, k
    )
    brown_clean = cv2.morphologyEx(brown_clean, cv2.MORPH_CLOSE, k)

    # Filter by area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        brown_clean, connectivity=8
    )
    filtered_brown = np.zeros_like(brown_clean)
    brown_count = 0
    total_brown_area = 0

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= cfg.brown_min_area_px:
            filtered_brown[labels == i] = 255
            brown_count += 1
            total_brown_area += area

    # Calculate percentage of leaf affected
    leaf_area = np.sum(leaf_mask)
    brown_percentage = (total_brown_area / max(leaf_area, 1)) * 100

    # Create visualization
    vis = rgb.copy()
    vis[filtered_brown > 0] = [255, 100, 0]  # Orange overlay for brown spots

    logging.info(
        f"Brown spots detected: {brown_count} regions, "
        f"{brown_percentage:.1f}% of leaf area ({total_brown_area} pixels)"
    )

    return vis, brown_percentage, brown_count
