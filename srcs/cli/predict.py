import argparse
import json
import sys
import time
from pathlib import Path

from srcs.predict.evaluation import evaluate_from_manifest
from srcs.predict.prediction_visualizer import PredictionVisualizer
from srcs.predict.predictor import Predictor
from srcs.utils.common import get_logger
from srcs.utils.image_utils import ImageLoader
from srcs.utils.visualization_utils import DisplayUtils

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict leaf disease from image(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image_path", help="Path to image or directory to predict")
    parser.add_argument(
        "-learnings",
        "--learnings-dir",
        default="artifacts/models",
        help="Directory containing model and metadata (default: artifacts/models)",
    )
    parser.add_argument(
        "-out",
        "--output-dir",
        default="artifacts/prediction_output",
        help="Directory to save prediction montages"
        " (default: artifacts/prediction_output)",
    )
    parser.add_argument(
        "-json",
        "--json-output",
        default="artifacts/prediction_output/batch_results.json",
        help="JSON output path"
        " (default: artifacts/prediction_output/batch_results.json)",
    )
    parser.add_argument(
        "-batch",
        "--batch-mode",
        action="store_true",
        help="Process directory of images and output JSON results",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate predictions against ground truth (requires --manifest)",
    )
    parser.add_argument(
        "--manifest",
        help="Path to manifest JSON file for evaluation",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Split to evaluate from manifest (default: val)",
    )
    return parser.parse_args()


def validate_inputs(args):
    image_path = Path(args.image_path)
    learnings_dir = Path(args.learnings_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Path not found: {image_path}")

    if args.batch_mode and not image_path.is_dir():
        raise ValueError(f"Batch mode requires a directory, got: {image_path}")

    if not args.batch_mode and not image_path.is_file():
        raise ValueError(f"Single mode requires an image file, got: {image_path}")

    if not learnings_dir.exists():
        raise FileNotFoundError(f"Learnings directory not found: {learnings_dir}")

    meta_file = learnings_dir / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_file}")

    if args.evaluate:
        if not args.batch_mode:
            raise ValueError("--evaluate requires --batch-mode")
        if not args.manifest:
            raise ValueError("--evaluate requires --manifest")
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    return image_path, learnings_dir


def print_prediction_result(result):
    logger.info(f"Image: {result['image_path']}")
    logger.info(f"Prediction: {result['top_prediction']} ({result['confidence']:.2%})")

    sorted_probs = sorted(
        result["all_probabilities"].items(), key=lambda x: x[1], reverse=True
    )
    logger.info("Top 3 predictions:")
    for i, (class_name, prob) in enumerate(sorted_probs[:3]):
        marker = "→" if i == 0 else " "
        logger.info(f"  {marker} {class_name}: {prob:.2%}")


def create_output_montage(result, output_dir):
    visualizer = PredictionVisualizer()
    output_dir = Path(output_dir)
    image_name = result["image_path"].stem
    output_file = output_dir / f"{image_name}_prediction.png"
    visualizer.create_montage(result, output_file)
    return output_file


def get_image_files(directory_path):
    return ImageLoader.get_image_files(directory_path)


def process_batch_predictions(
    predictor, image_directory, manifest_path=None, split=None
):
    """Process all images in directory and return batch results."""
    if manifest_path and split:
        image_files = get_images_from_manifest(manifest_path, split, image_directory)
    else:
        image_files = get_image_files(image_directory)

    if not image_files:
        logger.warning(f"No image files found in {image_directory}")
        return []

    logger.info(f"Found {len(image_files)} images to process")

    start_time = time.time()
    results = predictor.predict_batch(image_files)
    processing_time = time.time() - start_time

    return results, processing_time


def get_images_from_manifest(manifest_path, split, base_directory):
    """Get image paths from manifest for specific split."""
    import json

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    image_files = []
    base_dir = Path(base_directory)

    for item in manifest.get("items", []):
        if item.get("split") == split:
            image_path = base_dir / item["id"]
            if image_path.exists():
                image_files.append(image_path)
            else:
                logger.warning(f"Image not found: {image_path}")

    return image_files


def create_batch_summary(results, processing_time):
    """Create summary statistics for batch predictions."""
    if not results:
        return {"total_images": 0, "processing_time": f"{processing_time:.2f}s"}

    predictions = [r["top_prediction"] for r in results]
    prediction_counts = {}
    for pred in predictions:
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1

    avg_confidence = sum(r["confidence"] for r in results) / len(results)

    return {
        "total_images": len(results),
        "processing_time": f"{processing_time:.2f}s",
        "average_confidence": f"{avg_confidence:.2%}",
        "prediction_distribution": prediction_counts,
    }


def save_batch_results_json(results, processing_time, output_path):
    """Save batch results to JSON file."""
    output_path = Path(output_path)

    if not output_path.is_absolute() and not str(output_path).startswith("artifacts/"):
        output_path = Path("artifacts/prediction_output") / output_path.name

    json_results = []
    for result in results:
        json_result = {
            "image_path": str(result["image_path"]),
            "top_prediction": result["top_prediction"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"],
        }
        json_results.append(json_result)

    summary = create_batch_summary(results, processing_time)

    output_data = {"batch_results": json_results, "summary": summary}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return output_path


def _run_evaluation(args, predictor):
    """Run evaluation metrics computation."""
    logger.info("Computing evaluation metrics...")
    try:
        eval_metrics = evaluate_from_manifest(
            predictor,
            Path(args.manifest),
            split=args.split,
            output_dir=Path("artifacts/prediction_output/evaluation"),
        )

        if eval_metrics:
            logger.info("Evaluation completed successfully")
            logger.info(
                "Evaluation results saved to: "
                "artifacts/prediction_output/evaluation/"
            )
            return eval_metrics
        else:
            logger.error("Evaluation failed - no metrics computed")
            return None
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None


def _create_and_display_dashboard(results, eval_metrics):
    """Create and display the prediction dashboard."""
    dashboard_dir = Path("artifacts/prediction_output")
    dashboard_file = DisplayUtils.create_batch_dashboard(
        results, dashboard_dir / "batch_dashboard.png", eval_metrics
    )
    if dashboard_file:
        DisplayUtils.open_image_viewer(dashboard_file)


def log_batch_summary(results, processing_time):
    """Log summary of batch predictions."""
    if not results:
        logger.warning("No predictions made.")
        return

    summary = create_batch_summary(results, processing_time)

    logger.info("Batch Processing Summary:")
    logger.info(f"  Total images processed: {summary['total_images']}")
    logger.info(f"  Processing time: {summary['processing_time']}")
    logger.info(f"  Average confidence: {summary['average_confidence']}")
    logger.info("Prediction distribution:")
    for pred, count in summary["prediction_distribution"].items():
        logger.info(f"  {pred}: {count} images")


def main():
    try:
        args = parse_args()

        image_path, learnings_dir = validate_inputs(args)

        predictor = Predictor(learnings_dir)
        predictor.load()
        logger.info(f"Model loaded: {predictor.model_loader.num_classes} classes")

        if args.batch_mode:
            logger.info(f"Processing directory: {image_path}")
            manifest_path = args.manifest if args.evaluate else None
            split = args.split if args.evaluate else None
            results, processing_time = process_batch_predictions(
                predictor, image_path, manifest_path, split
            )

            if not results:
                logger.error("No images found or processed successfully.")
                sys.exit(1)

            log_batch_summary(results, processing_time)

            if args.json_output:
                output_file = save_batch_results_json(
                    results, processing_time, args.json_output
                )
                logger.info(f"Results saved to: {output_file}")

            eval_metrics = _run_evaluation(args, predictor) if args.evaluate else None
            _create_and_display_dashboard(results, eval_metrics)

            logger.info("Batch prediction completed successfully")

        else:
            logger.info(f"Processing image: {image_path}")
            result = predictor.predict_single(image_path)

            print_prediction_result(result)

            if args.output_dir:
                output_file = create_output_montage(result, args.output_dir)
                logger.info(f"Montage saved: {output_file}")
                DisplayUtils.open_image_viewer(output_file)

            logger.info("Prediction completed successfully")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Input error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
