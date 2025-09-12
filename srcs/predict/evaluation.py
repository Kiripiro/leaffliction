from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from srcs.predict.predictor import Predictor
from srcs.utils.common import get_logger
from srcs.utils.metrics import compute_classification_metrics

logger = get_logger(__name__)


class PredictionEvaluator:
    """Evaluator for computing metrics on prediction results with ground truth."""

    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def evaluate_predictions(
        self,
        image_paths: List[Path],
        true_labels: List[str],
        output_dir: Path | None = None,
    ) -> Dict[str, float]:
        """Evaluate predictions against ground truth labels.

        Args:
            image_paths: List of image paths to predict
            true_labels: List of ground truth class names
            output_dir: Optional directory to save evaluation results

        Returns:
            Dictionary of computed metrics
        """
        if len(image_paths) != len(true_labels):
            raise ValueError("Number of images must match number of true labels")

        logger.info(f"Evaluating {len(image_paths)} predictions")

        predictions = self.predictor.predict_batch(image_paths)

        if len(predictions) != len(true_labels):
            logger.warning(
                f"Only {len(predictions)} out of {len(image_paths)} images "
                f"were successfully processed"
            )

        pred_labels = [p["top_prediction"] for p in predictions]

        label_to_idx = {
            label: idx for idx, label in enumerate(self.predictor.model_loader.labels)
        }

        y_true = []
        y_pred = []
        valid_predictions = []

        for i, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
            if true_label not in label_to_idx or pred_label not in label_to_idx:
                logger.warning(f"Skipping unknown label: {true_label} or {pred_label}")
                continue

            y_true.append(label_to_idx[true_label])
            y_pred.append(label_to_idx[pred_label])
            valid_predictions.append(predictions[i])

        if not y_true:
            logger.error("No valid predictions to evaluate")
            return {}

        metrics = compute_classification_metrics(
            y_true, y_pred, self.predictor.model_loader.labels
        )

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            eval_results = {
                "metrics": metrics,
                "evaluation_info": {
                    "total_images": len(image_paths),
                    "valid_predictions": len(valid_predictions),
                    "class_labels": self.predictor.model_loader.labels,
                },
                "detailed_results": [
                    {
                        "image_path": str(pred["image_path"]),
                        "true_label": true_labels[i],
                        "predicted_label": pred["top_prediction"],
                        "confidence": pred["confidence"],
                        "correct": true_labels[i] == pred["top_prediction"],
                    }
                    for i, pred in enumerate(valid_predictions)
                ],
            }

            results_path = output_dir / "evaluation_results.json"
            with results_path.open("w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2)

            logger.info(f"Evaluation results saved to: {results_path}")

        logger.info("Evaluation completed successfully")
        return metrics


def evaluate_from_manifest(
    predictor: Predictor,
    manifest_path: Path,
    split: str = "test",
    output_dir: Path | None = None,
) -> Dict[str, float]:
    """Evaluate predictions using a manifest file.

    Args:
        predictor: Initialized predictor instance
        manifest_path: Path to manifest JSON file
        split: Split to evaluate ("test", "val", etc.)
        output_dir: Optional directory to save results

    Returns:
        Dictionary of computed metrics
    """
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    if isinstance(manifest_data, dict) and "items" in manifest_data:
        items = manifest_data["items"]
    else:
        items = manifest_data

    test_items = [item for item in items if item.get("split") == split]

    if not test_items:
        logger.error(f"No items found for split '{split}' in manifest")
        return {}

    image_paths = [Path(item["src"]) for item in test_items]
    true_labels = [item.get("label", item["class"]) for item in test_items]

    evaluator = PredictionEvaluator(predictor)
    return evaluator.evaluate_predictions(image_paths, true_labels, output_dir)
