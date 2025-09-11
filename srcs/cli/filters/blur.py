"""
Blur/Saliency filter for image transformation pipeline.
Creates a saliency map showing important regions in white/gray
and less important in black.
"""

from __future__ import annotations

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


def apply_blur_filter(
    rgb: np.ndarray, cfg: TransformConfig, make_mask_func: callable
) -> np.ndarray:
    """
    Create a saliency map showing important regions in white/gray
    and less important in black.

    Args:
        rgb: Input RGB image
        cfg: Transform configuration
        make_mask_func: Function to create leaf mask

    Returns:
        RGB saliency map
    """
    # Get leaf mask
    mask, _ = make_mask_func(rgb)
    if mask is None:
        return rgb

    leaf_mask = (mask > 0) if mask.ndim == 2 else (mask[..., 0] > 0)

    # Create saliency map
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    saliency = np.zeros_like(gray, dtype=np.float32)

    # 1. Edge saliency (veins, leaf borders)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    saliency += edges_dilated.astype(np.float32) * 0.4

    # 2. Texture/gradient saliency
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(grad_x, grad_y)
    gradient_norm = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    saliency += gradient_norm.astype(np.float32) * 0.3

    # 3. Brown/disease region saliency
    if hasattr(cfg, "brown_hue_range"):
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

        # Clean and dilate brown regions for better visibility
        brown_clean = cv2.morphologyEx(
            brown_regions.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel
        )
        brown_dilated = cv2.dilate(brown_clean, kernel, iterations=2)
        saliency += brown_dilated.astype(np.float32) * 0.6  # Higher weight for disease

    # 4. Color variation saliency (areas with unusual colors)
    blurred_rgb = cv2.GaussianBlur(rgb, (15, 15), 0)
    color_diff = np.mean(
        np.abs(rgb.astype(np.float32) - blurred_rgb.astype(np.float32)), axis=2
    )
    color_diff_norm = cv2.normalize(color_diff, None, 0, 255, cv2.NORM_MINMAX)
    saliency += color_diff_norm * 0.2

    # Normalize saliency map
    saliency_norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # Apply Gaussian blur for smooth transitions
    saliency_blurred = cv2.GaussianBlur(saliency_norm, (5, 5), cfg.gaussian_sigma)

    # Mask to leaf area only
    result = np.zeros_like(gray)
    result[leaf_mask] = saliency_blurred[leaf_mask]

    # Convert to 3-channel for consistency with other outputs
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    return result_rgb
