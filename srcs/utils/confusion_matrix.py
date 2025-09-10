from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def _gather_predictions_and_labels(
    model: Any, data: Any
) -> Tuple[List[int], List[int]]:
    """Iterate over a dataset yielding (X, y) and return flat y_true and y_pred indices.

    - Supports sparse (1D) or one-hot (2D) y labels.
    - Uses model.predict with verbose="auto".
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    for bx, by in data:
        probs = model.predict(bx, verbose="auto")
        pred_idx = np.argmax(probs, axis=-1)
        if getattr(by, "ndim", 1) > 1:
            true_idx = np.argmax(by, axis=-1)
        else:
            true_idx = by
        y_true.extend([int(x) for x in np.asarray(true_idx).tolist()])
        y_pred.extend([int(x) for x in np.asarray(pred_idx).tolist()])
    return y_true, y_pred


def compute_confusion_counts(
    y_true: List[int], y_pred: List[int], num_classes: int
) -> List[List[int]]:
    """Compute raw confusion matrix counts as a nested list [true][pred]."""
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def save_confusion_json(cm: List[List[int]], labels: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"matrix": cm, "labels": labels}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_confusion_png(
    cm: List[List[int]], labels: List[str], out_path: Path, *, normalize: bool = True
) -> None:
    """Plot confusion matrix to PNG. Normalizes rows by default.

    Matplotlib is imported lazily to avoid hard dependency at import time.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        LOGGER.warning("matplotlib unavailable, skipping confusion matrix PNG: %s", e)
        return

    num_classes = len(labels)
    cm_np = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm_np.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_np, np.maximum(row_sums, 1.0))
    else:
        cm_plot = cm_np

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(cm_plot, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    title = "Confusion Matrix" + (" (normalized)" if normalize else "")
    ax.set_title(title)
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_plot[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}" if normalize else f"{int(val)}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def confusion_matrix(
    model: Any,
    data: Any,
    labels: List[str],
    out_dir: Path,
    *,
    normalize_png: bool = True,
) -> Tuple[Path, Path]:
    """Compute and export confusion matrix JSON + PNG.

    Returns (json_path, png_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        y_true, y_pred = _gather_predictions_and_labels(model, data)
        cm = compute_confusion_counts(y_true, y_pred, num_classes=len(labels))
        json_path = out_dir / "confusion_matrix.json"
        png_path = out_dir / "confusion_matrix.png"
        save_confusion_json(cm, labels, json_path)
        try:
            plot_confusion_png(cm, labels, png_path, normalize=normalize_png)
        except Exception as e:
            LOGGER.warning("Failed to plot confusion matrix: %s", e)
        return json_path, png_path
    except Exception as e:
        LOGGER.warning("Failed to compute confusion matrix: %s", e)
        return out_dir / "confusion_matrix.json", out_dir / "confusion_matrix.png"
