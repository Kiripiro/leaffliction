"""
Histogram filter for image transformation pipeline.
Creates HSV histograms for color analysis.
"""

from __future__ import annotations

import cv2
import matplotlib
import matplotlib.pyplot as plt
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


def apply_histogram_filter(rgb: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    """
    Create HSV histogram visualization.

    Args:
        rgb: Input RGB image
        cfg: Transform configuration

    Returns:
        RGB image of the histogram plot
    """
    matplotlib.use("Agg")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    fig = plt.figure(figsize=(6, 4))
    bins = 50
    plt.hist(h.ravel(), bins=bins, color="r", alpha=0.5, label="H")
    plt.hist(s.ravel(), bins=bins, color="g", alpha=0.5, label="S")
    plt.hist(v.ravel(), bins=bins, color="b", alpha=0.5, label="V")
    plt.title("HSV Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    fig.tight_layout()

    # Render to RGB image array
    fig.canvas.draw()
    w, h2 = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h2, w, 4))
    rgb_img = rgba[..., :3].copy()
    plt.close(fig)

    return rgb_img
