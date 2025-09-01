import argparse
from pathlib import Path

try:
    from srcs.preprocessing.dataset_balancer import DatasetBalancer
    from srcs.utils.common import setup_logging
except ModuleNotFoundError:
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from srcs.preprocessing.dataset_balancer import DatasetBalancer
    from srcs.utils.common import setup_logging


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
