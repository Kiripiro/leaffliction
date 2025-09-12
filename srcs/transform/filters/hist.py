from __future__ import annotations

from typing import Dict

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    from ..Transformation import TransformConfig
except ImportError:
    import sys
    from pathlib import Path

    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from Transformation import TransformConfig


def _analyze_color_regions(
    rgb: np.ndarray, hsv: np.ndarray, mask: np.ndarray = None
) -> Dict[str, float]:
    h, s, v = cv2.split(hsv)

    if mask is None:
        mask = np.ones(h.shape, dtype=bool)
    else:
        mask = mask > 0 if mask.ndim == 2 else mask[..., 0] > 0

    total_pixels = np.sum(mask)
    if total_pixels == 0:
        return {}

    color_analysis = {}

    healthy_green = mask & (h >= 35) & (h <= 85) & (s >= 40) & (v >= 30)
    color_analysis["Healthy Green"] = (np.sum(healthy_green) / total_pixels) * 100

    yellowish_green = mask & (h >= 20) & (h <= 40) & (s >= 25) & (v >= 30)
    color_analysis["Yellowish Green"] = (np.sum(yellowish_green) / total_pixels) * 100

    yellow = mask & (h >= 15) & (h <= 35) & (s >= 50) & (v >= 50)
    color_analysis["Yellow"] = (np.sum(yellow) / total_pixels) * 100

    brown_orange = mask & (((h >= 0) & (h <= 25)) | (h >= 160)) & (s >= 30) & (v >= 20)
    color_analysis["Brown/Orange"] = (np.sum(brown_orange) / total_pixels) * 100

    red_spots = (
        mask
        & (((h >= 160) & (h <= 180)) | ((h >= 0) & (h <= 10)))
        & (s >= 40)
        & (v >= 30)
    )
    color_analysis["Red"] = (np.sum(red_spots) / total_pixels) * 100

    dark_areas = mask & (v <= 50) & (s >= 20)
    color_analysis["Dark Areas"] = (np.sum(dark_areas) / total_pixels) * 100

    bright_areas = mask & (v >= 200) & (s <= 30)
    color_analysis["Bright Areas"] = (np.sum(bright_areas) / total_pixels) * 100

    purple_areas = mask & (h >= 120) & (h <= 160) & (s >= 20)
    color_analysis["Purple"] = (np.sum(purple_areas) / total_pixels) * 100

    return color_analysis


def _create_color_distribution_plot(color_analysis: Dict[str, float], ax) -> None:
    significant_colors = {k: v for k, v in color_analysis.items() if v >= 1.0}

    if not significant_colors:
        ax.text(
            0.5,
            0.5,
            "No significant\ncolors detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Color Distribution")
        return

    colors = list(significant_colors.keys())
    percentages = list(significant_colors.values())

    bar_colors = []
    for color_name in colors:
        if "Healthy Green" in color_name:
            bar_colors.append("#2E7D32")
        elif "Yellowish" in color_name:
            bar_colors.append("#7CB342")
        elif "Yellow" in color_name:
            bar_colors.append("#FBC02D")
        elif "Brown" in color_name or "Orange" in color_name:
            bar_colors.append("#FF6F00")
        elif "Red" in color_name:
            bar_colors.append("#D32F2F")
        elif "Dark" in color_name:
            bar_colors.append("#424242")
        elif "Bright" in color_name:
            bar_colors.append("#E0E0E0")
        elif "Purple" in color_name:
            bar_colors.append("#7B1FA2")
        else:
            bar_colors.append("#90A4AE")

    bars = ax.bar(
        range(len(colors)),
        percentages,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

    ax.set_xlabel("Color Types")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Detected Color Distribution")
    ax.set_xticks(range(len(colors)))
    ax.set_xticklabels(colors, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(percentages) * 1.15 if percentages else 10)


def _create_enhanced_hsv_plot(hsv: np.ndarray, mask: np.ndarray, ax) -> None:
    h, s, v = cv2.split(hsv)

    if mask is not None:
        mask_bool = mask > 0 if mask.ndim == 2 else mask[..., 0] > 0
        h_masked = h[mask_bool]
        s_masked = s[mask_bool]
        v_masked = v[mask_bool]
    else:
        h_masked = h.ravel()
        s_masked = s.ravel()
        v_masked = v.ravel()

    bins = 60

    ax.hist(h_masked, bins=bins, color="red", alpha=0.6, label="Hue (H)", density=True)
    ax.hist(
        s_masked,
        bins=bins,
        color="green",
        alpha=0.6,
        label="Saturation (S)",
        density=True,
    )
    ax.hist(
        v_masked, bins=bins, color="blue", alpha=0.6, label="Value (V)", density=True
    )

    ax.axvline(x=35, color="darkgreen", linestyle="--", alpha=0.7, label="Green start")
    ax.axvline(x=85, color="darkgreen", linestyle="--", alpha=0.7, label="Green end")
    ax.axvline(x=15, color="orange", linestyle=":", alpha=0.7, label="Yellow/Brown")

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Enhanced HSV Histogram")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def apply_histogram_filter(rgb: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    matplotlib.use("Agg")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv_copy = hsv.copy()
    h, s, v = cv2.split(hsv_copy)

    leaf_mask = (s > 10) & (v > 15) & (v < 245)
    color_analysis = _analyze_color_regions(rgb, hsv, leaf_mask)

    fig = plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(2, 2, 1)
    _create_color_distribution_plot(color_analysis, ax1)

    ax2 = plt.subplot(2, 2, 2)
    _create_enhanced_hsv_plot(hsv, leaf_mask, ax2)

    ax3 = plt.subplot(2, 2, 3)
    ax3.axis("off")

    summary_lines = ["COLOR ANALYSIS:", ""]
    total_analyzed = np.sum(leaf_mask)
    summary_lines.append(f"Analyzed pixels: {total_analyzed:,}")
    summary_lines.append("")

    sorted_colors = sorted(color_analysis.items(), key=lambda x: x[1], reverse=True)

    for color_name, percentage in sorted_colors[:6]:
        if percentage >= 0.5:
            summary_lines.append(f"• {color_name}: {percentage:.1f}%")

    summary_lines.append("")
    healthy_total = color_analysis.get("Healthy Green", 0) + color_analysis.get(
        "Yellowish Green", 0
    )
    disease_total = (
        color_analysis.get("Brown/Orange", 0)
        + color_analysis.get("Red", 0)
        + color_analysis.get("Yellow", 0)
    )

    if healthy_total > 50:
        health_status = "Mostly healthy foliage"
    elif disease_total > 30:
        health_status = "Significant disease signs"
    elif color_analysis.get("Yellow", 0) > 20:
        health_status = "Possible yellowing/stress"
    else:
        health_status = "Mixed or undetermined state"

    summary_lines.append(f"STATUS: {health_status}")

    summary_text = "\n".join(summary_lines)
    ax3.text(
        0.05,
        0.95,
        summary_text,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8},
    )

    ax4 = plt.subplot(2, 2, 4)

    hue_ranges = {
        "Green (35-85°)": np.sum(leaf_mask & (h >= 35) & (h <= 85)),
        "Yellow/Orange (15-35°)": np.sum(leaf_mask & (h >= 15) & (h <= 35)),
        "Red (0-15° & 160-180°)": np.sum(
            leaf_mask & (((h >= 0) & (h <= 15)) | (h >= 160))
        ),
        "Purple (120-160°)": np.sum(leaf_mask & (h >= 120) & (h <= 160)),
        "Others": np.sum(leaf_mask & (((h > 85) & (h < 120)) | ((h > 35) & (h < 15)))),
    }

    total_hue_pixels = sum(hue_ranges.values())
    if total_hue_pixels > 0:
        hue_percentages = {
            k: (v / total_hue_pixels) * 100 for k, v in hue_ranges.items() if v > 0
        }

        if hue_percentages:
            colors_pie = ["#4CAF50", "#FFC107", "#F44336", "#9C27B0", "#607D8B"]
            wedges, texts, autotexts = ax4.pie(
                hue_percentages.values(),
                labels=hue_percentages.keys(),
                colors=colors_pie[: len(hue_percentages)],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax4.set_title("Distribution by Hue")

            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_weight("bold")
                autotext.set_fontsize(8)
        else:
            ax4.text(
                0.5,
                0.5,
                "Insufficient data\nfor analysis",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Distribution by Hue")

    plt.tight_layout()

    fig.canvas.draw()
    w, h_fig = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
        (h_fig, w, 4)
    )
    rgb_img = rgba[..., :3].copy()
    plt.close(fig)

    return rgb_img
