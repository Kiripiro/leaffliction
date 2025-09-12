from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Union

from srcs.utils.common import get_logger

logger = get_logger(__name__)


class DisplayUtils:

    @staticmethod
    def open_image_viewer(image_path: Union[str, Path]) -> None:
        """Open image with system default viewer."""
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Cannot display image, file not found: {image_path}")
            return

        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(image_path)], check=True)
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", str(image_path)], check=True)
            elif sys.platform == "win32":
                subprocess.run(["start", str(image_path)], shell=True, check=True)
            else:
                logger.warning(
                    f"Unsupported platform for image display: {sys.platform}"
                )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to open image viewer: {e}")
        except FileNotFoundError:
            logger.warning("No image viewer found")

    @staticmethod
    def create_confusion_matrix(
        results: List[dict], output_path: Union[str, Path]
    ) -> Path | None:
        """Create confusion matrix from batch prediction results."""
        from srcs.utils.confusion_matrix import (
            compute_confusion_counts,
            plot_confusion_png,
            save_confusion_json,
        )

        output_path = Path(output_path)

        if not results:
            logger.warning("No results to create confusion matrix")
            return None

        y_true = []
        y_pred = []
        all_labels = set()

        for result in results:
            image_path = Path(result["image_path"])
            true_label = image_path.parent.name
            pred_label = result["top_prediction"]

            y_true.append(true_label)
            y_pred.append(pred_label)
            all_labels.add(true_label)
            all_labels.add(pred_label)

        labels = sorted(all_labels)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}

        y_true_idx = [label_to_idx[label] for label in y_true]
        y_pred_idx = [label_to_idx[label] for label in y_pred]

        cm = compute_confusion_counts(y_true_idx, y_pred_idx, len(labels))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = output_path.with_suffix(".json")

        try:
            save_confusion_json(cm, labels, json_path)
            plot_confusion_png(cm, labels, output_path, normalize=True)
            logger.info(f"Confusion matrix saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to create confusion matrix: {e}")
            return None

    @staticmethod
    def create_batch_dashboard(
        results: List[dict], output_path: Union[str, Path], metrics: dict = None
    ) -> Path | None:
        """Create comprehensive dashboard for batch prediction analysis."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not available for dashboard creation")
            return None

        output_path = Path(output_path)

        if not results:
            logger.warning("No results for dashboard creation")
            return None

        if metrics:
            fig = plt.figure(figsize=(20, 15))
            ax1 = plt.subplot2grid((3, 2), (0, 0))
            ax2 = plt.subplot2grid((3, 2), (0, 1))
            ax3 = plt.subplot2grid((3, 2), (1, 0))
            ax4 = plt.subplot2grid((3, 2), (1, 1))
            ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        DisplayUtils._plot_prediction_distribution(results, ax1)
        DisplayUtils._plot_confidence_histogram(results, ax2)
        DisplayUtils._plot_probability_heatmap(results, ax3)
        DisplayUtils._plot_lowest_confidence(results, ax4)

        if metrics:
            DisplayUtils._plot_evaluation_metrics(metrics, ax5)

        plt.tight_layout(pad=3.0)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Batch dashboard saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save dashboard: {e}")
            return None

    @staticmethod
    def _plot_prediction_distribution(results, ax):
        predictions = [r["top_prediction"] for r in results]
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1

        classes = sorted(pred_counts.keys())
        counts = [pred_counts[c] for c in classes]
        bars = ax.bar(range(len(classes)), counts, color="skyblue")
        ax.set_title("Prediction Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Classes")
        ax.set_ylabel("Number of Images")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(
            [c.replace("__", "\n") for c in classes],
            rotation=45,
            ha="right",
            fontsize=10,
        )
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(count),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    @staticmethod
    def _plot_confidence_histogram(results, ax):
        import numpy as np

        confidences = [r["confidence"] for r in results]
        ax.hist(confidences, bins=30, color="lightcoral", alpha=0.7, edgecolor="black")
        ax.axvline(
            np.mean(confidences),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(confidences):.2%}",
        )
        ax.set_title("Confidence Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_probability_heatmap(results, ax):
        import matplotlib.pyplot as plt
        import numpy as np

        predictions = [r["top_prediction"] for r in results]
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1

        classes = sorted(pred_counts.keys())
        all_classes = set()
        for result in results:
            all_classes.update(result["all_probabilities"].keys())
        all_classes = sorted(all_classes)

        prob_matrix = np.zeros((len(classes), len(all_classes)))
        class_indices = {cls: i for i, cls in enumerate(classes)}
        all_class_indices = {cls: i for i, cls in enumerate(all_classes)}

        for pred_class in classes:
            class_results = [r for r in results if r["top_prediction"] == pred_class]
            if class_results:
                for all_class in all_classes:
                    avg_prob = np.mean(
                        [
                            r["all_probabilities"].get(all_class, 0)
                            for r in class_results
                        ]
                    )
                    prob_matrix[class_indices[pred_class]][
                        all_class_indices[all_class]
                    ] = avg_prob

        im = ax.imshow(prob_matrix, cmap="Blues", aspect="auto")
        ax.set_title(
            "Average Probabilities by Predicted Class", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("All Classes")
        ax.set_ylabel("Predicted Classes")
        ax.set_xticks(range(len(all_classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(
            [c.replace("__", "\n") for c in all_classes],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_yticklabels([c.replace("__", "\n") for c in classes], fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    @staticmethod
    def _plot_lowest_confidence(results, ax):
        lowest_conf = sorted(results, key=lambda x: x["confidence"])[:10]
        conf_values = [r["confidence"] for r in lowest_conf]
        names = [
            (
                Path(r["image_path"]).name[:15] + "..."
                if len(Path(r["image_path"]).name) > 15
                else Path(r["image_path"]).name
            )
            for r in lowest_conf
        ]

        bars = ax.barh(range(len(names)), conf_values, color="orange")
        ax.set_title("Lowest Confidence Predictions", fontsize=14, fontweight="bold")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Image Names")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0, 1)
        for bar, conf in zip(bars, conf_values):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{conf:.2%}",
                va="center",
                fontsize=8,
            )

    @staticmethod
    def _plot_evaluation_metrics(metrics, ax):
        main_metrics = ["accuracy", "macro_f1", "weighted_f1"]
        values = [metrics.get(metric, 0) for metric in main_metrics]
        labels = ["Accuracy", "Macro F1", "Weighted F1"]

        bars = ax.bar(labels, values, color=["#2E8B57", "#4169E1", "#FF6347"])
        ax.set_title("Evaluation Metrics", fontsize=16, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
            )
