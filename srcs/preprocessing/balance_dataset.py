import argparse
import sys
from pathlib import Path

from dataset_balancer import DatasetBalancer
from utils.common import setup_logging

sys.path.append(str(Path(__file__).parent.parent))


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Balance leaf disease dataset with augmentations"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)",
    )
    args = parser.parse_args()

    manifest_path = "../../datasets/manifest_split.json"
    source_dir = "../../images"
    target_dir = "../../augmented_directory"
    seed = 42

    balancer = DatasetBalancer(
        manifest_path=manifest_path,
        source_dir=source_dir,
        target_dir=target_dir,
        seed=seed,
        workers=args.workers,
    )

    balancer.run()


if __name__ == "__main__":
    main()
