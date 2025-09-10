import argparse
import sys
from pathlib import Path

from srcs.predict.prediction_visualizer import PredictionVisualizer
from srcs.predict.predictor import Predictor
from srcs.utils.common import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict leaf disease from image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image_path", help="Path to image to predict")
    parser.add_argument(
        "-learnings",
        "--learnings-dir",
        default="artifacts/models",
        help="Directory containing model and metadata (default: artifacts/models)",
    )
    parser.add_argument(
        "-out", "--output-dir", help="Directory to save prediction montages"
    )
    return parser.parse_args()


def validate_inputs(args):
    image_path = Path(args.image_path)
    learnings_dir = Path(args.learnings_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not learnings_dir.exists():
        raise FileNotFoundError(f"Learnings directory not found: {learnings_dir}")

    meta_file = learnings_dir / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_file}")

    return image_path, learnings_dir


def print_prediction_result(result):
    print(f"Image: {result['image_path']}")
    print(f"Prediction: {result['top_prediction']} ({result['confidence']:.2%})")

    sorted_probs = sorted(
        result["all_probabilities"].items(), key=lambda x: x[1], reverse=True
    )
    print("Top 3 predictions:")
    for i, (class_name, prob) in enumerate(sorted_probs[:3]):
        marker = "→" if i == 0 else " "
        print(f"  {marker} {class_name}: {prob:.2%}")


def create_output_montage(result, output_dir):
    visualizer = PredictionVisualizer()
    output_dir = Path(output_dir)
    image_name = result["image_path"].stem
    output_file = output_dir / f"{image_name}_prediction.png"
    visualizer.create_montage(result, output_file)
    return output_file


def main():
    try:
        args = parse_args()

        image_path, learnings_dir = validate_inputs(args)
        logger.info(f"Processing image: {image_path}")

        predictor = Predictor(learnings_dir)
        predictor.load()
        logger.info(f"Model loaded: {predictor.model_loader.num_classes} classes")

        result = predictor.predict_single(image_path)

        print_prediction_result(result)

        if args.output_dir:
            output_file = create_output_montage(result, args.output_dir)
            print(f"\nMontage saved: {output_file}")

        logger.info("Prediction completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
