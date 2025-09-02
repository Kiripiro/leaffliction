import argparse
from pathlib import Path

try:
    from srcs.cli.Distribution import count_images, merge_csv, plot_per_plant
    from srcs.preprocessing.dataset_balancer import DatasetBalancer
    from srcs.utils.common import get_logger, setup_logging
except ModuleNotFoundError:
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from srcs.cli.Distribution import count_images, merge_csv, plot_per_plant
    from srcs.preprocessing.dataset_balancer import DatasetBalancer
    from srcs.utils.common import get_logger, setup_logging


def main():

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Balance leaf disease dataset with augmentations"
    )
    parser.add_argument(
        "manifest_path",
        help="Path to the manifest.json file",
    )
    parser.add_argument(
        "source_dir",
        help="Path to the source images directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    manifest_path = args.manifest_path
    source_dir = args.source_dir
    target_dir = project_root / "augmented_directory"
    seed = 42

    balancer = DatasetBalancer(
        manifest_path=manifest_path,
        source_dir=source_dir,
        target_dir=target_dir,
        seed=seed,
        workers=args.workers,
    )

    balancer.run()

    analyze_distribution(target_dir)


def analyze_distribution(target_dir: str) -> None:
    """Analyze distribution of balanced dataset"""
    logger = get_logger(__name__)

    target_path = Path(target_dir)
    if not target_path.exists():
        logger.warning("Target directory doesn't exist: %s", target_dir)
        return

    logger.info("Analyzing distribution of balanced dataset...")
    rows = count_images(target_path, None)

    if not rows:
        logger.warning("No images found in target directory")
        return

    out_dir = target_path.parent / "artifacts" / "distribution"
    csv_path = out_dir / "balanced_distribution.csv"

    merge_csv(rows, csv_path)
    logger.info("Distribution CSV written: %s", csv_path.resolve())

    plot_per_plant(rows, out_dir)
    logger.info("Distribution plots written: %s", out_dir.resolve())

    total = sum(n for _, _, n in rows)
    logger.info("Total balanced images: %d", total)


if __name__ == "__main__":
    main()
