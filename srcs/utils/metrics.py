from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from srcs.utils.common import get_logger

LOGGER = get_logger(__name__)


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


def compute_classification_metrics(
    y_true: List[int], y_pred: List[int], labels: List[str]
) -> Dict[str, float]:
    """Compute essential classification metrics.

    Args:
        y_true: True class labels as integers
        y_pred: Predicted class labels as integers
        labels: List of class names for per-class metrics

    Returns:
        Dictionary containing accuracy, F1, precision, and recall metrics
    """
    num_classes = len(labels)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    if num_classes == 2:
        metrics["binary_f1"] = float(
            f1_score(y_true, y_pred, average="binary", zero_division=0)
        )
        metrics["binary_precision"] = float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        )
        metrics["binary_recall"] = float(
            recall_score(y_true, y_pred, average="binary", zero_division=0)
        )

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    for i, label in enumerate(labels):
        if i < len(per_class_f1):
            metrics[f"f1_{label}"] = float(per_class_f1[i])
            metrics[f"precision_{label}"] = float(per_class_precision[i])
            metrics[f"recall_{label}"] = float(per_class_recall[i])

    return metrics


def save_metrics_json(metrics: Dict[str, float], out_path: Path) -> None:
    """Save metrics to JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def log_metrics_summary(metrics: Dict[str, float], labels: List[str]) -> None:
    """Log a summary of key metrics."""
    LOGGER.info("Classification Metrics Summary:")
    LOGGER.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    LOGGER.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
    LOGGER.info(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

    if len(labels) == 2:
        LOGGER.info(f"  Binary F1: {metrics.get('binary_f1', 'N/A'):.4f}")
        binary_precision = metrics.get("binary_precision", "N/A")
        LOGGER.info(f"  Binary Precision: {binary_precision:.4f}")
        LOGGER.info(f"  Binary Recall: {metrics.get('binary_recall', 'N/A'):.4f}")

    LOGGER.info("Per-class F1 scores:")
    for label in labels:
        f1_key = f"f1_{label}"
        if f1_key in metrics:
            LOGGER.info(f"  {label}: {metrics[f1_key]:.4f}")


def compute_evaluation_metrics(
    model: Any,
    data: Any,
    labels: List[str],
    out_dir: Path,
) -> Dict[str, float]:
    """Compute and save classification metrics.

    Args:
        model: Trained Keras model
        data: Validation/test data generator
        labels: List of class names
        out_dir: Output directory for metrics JSON

    Returns:
        Dictionary of computed metrics
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        y_true, y_pred = _gather_predictions_and_labels(model, data)
        metrics = compute_classification_metrics(y_true, y_pred, labels)

        metrics_path = out_dir / "metrics.json"
        save_metrics_json(metrics, metrics_path)

        log_metrics_summary(metrics, labels)

        return metrics

    except Exception as e:
        LOGGER.error(f"Failed to compute metrics: {e}")
        return {}
