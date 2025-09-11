import argparse
import sys
from pathlib import Path

from srcs.preprocessing.image_augmenter import ImageAugmenter
from srcs.utils.common import get_logger
from srcs.utils.image_utils import ImageLoader

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply image augmentations to a single image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available transformations:
  flip      - Random horizontal or vertical flip
  rotate    - Random rotation (-30° to +30°)
  skew      - Perspective skew transformation
  shear     - Affine shear transformation
  crop      - Random crop and resize
  distortion - Add noise and adjust contrast

Examples:
  python srcs/cli/Augmentation.py flip image.jpg
  python srcs/cli/Augmentation.py rotate --output custom_output.jpg image.jpg
  python srcs/cli/Augmentation.py crop --seed 42 image.jpg
        """,
    )

    parser.add_argument(
        "transformation",
        choices=["flip", "rotate", "skew", "shear", "crop", "distortion"],
        help="Type of transformation to apply",
    )
    parser.add_argument(
        "image_path",
        help="Path to input image",
    )
    parser.add_argument(
        "-out",
        "--output",
        help="Output path (default: prediction_output/augmented_<transform>_<name>)",
    )
    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        help="Random seed for reproducible results",
    )

    return parser.parse_args()


def validate_inputs(args):
    return ImageLoader.validate_image_path(args.image_path)


def generate_output_path(input_path, transformation, custom_output=None):
    """Generate output path for augmented image."""
    if custom_output:
        return Path(custom_output)

    input_path = Path(input_path)
    output_dir = Path("prediction_output")
    output_name = f"augmented_{transformation}_{input_path.name}"

    return output_dir / output_name


def main():
    try:
        args = parse_args()

        image_path = validate_inputs(args)
        output_path = generate_output_path(image_path, args.transformation, args.output)

        logger.info(f"Applying {args.transformation} to: {image_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        augmenter = ImageAugmenter(seed=args.seed)

        transformation_method = getattr(augmenter, args.transformation)
        success = transformation_method(str(image_path), str(output_path))

        if success:
            logger.info(f"Augmented image saved: {output_path}")
            logger.info(f"Successfully applied {args.transformation}")
        else:
            logger.error(f"Failed to apply {args.transformation}")
            sys.exit(1)

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Input error: {e}")
        sys.exit(1)
    except AttributeError:
        logger.error(f"Unknown transformation: {args.transformation}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Augmentation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
