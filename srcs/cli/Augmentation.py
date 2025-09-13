import argparse
import shutil
import sys
from pathlib import Path

from srcs.cli.Distribution import count_images, merge_csv, plot_per_plant
from srcs.preprocessing.dataset_balancer import DatasetBalancer
from srcs.preprocessing.image_augmenter import ImageAugmenter
from srcs.utils.common import get_logger, setup_logging

logger = get_logger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
DEFAULT_DATASET_OUTPUT = "artifacts/augmented_directory"
DEFAULT_SINGLE_OUTPUT = "artifacts/example"
DEFAULT_SEED = 42
TRANSFORMATIONS = ["flip", "rotate", "skew", "shear", "crop", "distortion"]


class AugmentationError(Exception):
    pass


class InputValidationError(AugmentationError):
    pass


class ProcessingError(AugmentationError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply augmentations to balance a dataset. "
            "Preferred usage: provide a dataset root (PLANT/CLASS/*.jpg)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dataset mode (recommended): provide the dataset root
  leaffliction-augment images/
  leaffliction-augment images/ --output my_augmented_dataset

  # Single image mode (creates example folder)
  leaffliction-augment image.jpg
  leaffliction-augment image.jpg --output my_examples
        """,
    )

    parser.add_argument(
        "input_path",
        help=(
            "Path to dataset root directory (preferred) OR single image file. "
            "Providing a JSON manifest is deprecated."
        ),
    )
    parser.add_argument(
        "-out",
        "--output",
        help="Output directory (default: artifacts/augmented_directory for "
        "datasets, artifacts/example for single images)",
    )
    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)",
    )

    return parser.parse_args()


def main():
    setup_logging()

    try:
        args = parse_args()
        input_path = Path(args.input_path)

        if not input_path.exists():
            raise InputValidationError(f"Input path not found: {input_path}")

        # Single image mode
        if (
            input_path.is_file()
            and input_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ):
            single_image_mode(args, input_path)
            return

        # Dataset mode by directory (preferred)
        if input_path.is_dir():
            dataset_mode_dir(args, input_path)
            return

        raise InputValidationError(
            "Unsupported input. Provide a dataset directory or an image file."
        )

    except InputValidationError as e:
        logger.error(f"Input validation error: {e}")
        sys.exit(1)
    except ProcessingError as e:
        logger.error(f"Processing error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


def single_image_mode(args, image_path):
    output_dir = Path(args.output) if args.output else Path(DEFAULT_SINGLE_OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing single image: {image_path}")
    logger.info(f"Output directory: {output_dir}")

    original_output = output_dir / f"original_{image_path.name}"
    shutil.copy2(image_path, original_output)
    logger.info(f"Original image copied: {original_output}")

    augmenter = ImageAugmenter(seed=args.seed)

    transformation_methods = {
        "flip": augmenter.flip,
        "rotate": augmenter.rotate,
        "skew": augmenter.skew,
        "shear": augmenter.shear,
        "crop": augmenter.crop,
        "distortion": augmenter.distortion,
    }

    for transform in TRANSFORMATIONS:
        output_path = output_dir / f"{transform}_{image_path.name}"
        transformation_method = transformation_methods[transform]
        success = transformation_method(str(image_path), str(output_path))

        if success:
            logger.info(f"{transform.capitalize()} applied: {output_path}")
        else:
            raise ProcessingError(f"Failed to apply {transform} transformation")

    logger.info("Single image augmentation completed successfully")


def dataset_mode_dir(args, source_dir: Path):
    """Process a dataset directly from a source directory (preferred)."""
    target_dir = Path(args.output) if args.output else Path(DEFAULT_DATASET_OUTPUT)
    if not source_dir.exists():
        raise InputValidationError(f"Source directory not found: {source_dir}")
    logger.info(f"Processing dataset directory: {source_dir}")
    logger.info(f"Target directory: {target_dir}")
    balancer = DatasetBalancer(
        source_dir=str(source_dir),
        target_dir=str(target_dir),
        seed=args.seed,
        workers=args.workers,
    )
    balancer.run()
    logger.info("Dataset augmentation completed successfully")
    try:
        analyze_distribution(target_dir)
    except Exception as e:
        logger.warning(f"Distribution analysis failed: {e}")


def analyze_distribution(target_dir: Path) -> None:
    """Analyze distribution of balanced dataset"""

    if not target_dir.exists():
        logger.warning("Target directory doesn't exist: %s", target_dir)
        return

    logger.info("Analyzing distribution of balanced dataset...")
    rows = count_images(target_dir, None)

    if not rows:
        logger.warning("No images found in target directory")
        return

    out_dir = Path("artifacts") / "distribution"
    csv_path = out_dir / "balanced_distribution.csv"

    merge_csv(rows, csv_path)
    logger.info("Distribution CSV written: %s", csv_path.resolve())

    plot_per_plant(rows, out_dir)
    logger.info("Distribution plots written: %s", out_dir.resolve())

    total = sum(n for _, _, n in rows)
    logger.info("Total balanced images: %d", total)


if __name__ == "__main__":
    main()
