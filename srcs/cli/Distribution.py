from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

IMAGE_EXTS = {".jpg"}

try:
    from srcs.utils.common import setup_logging
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from srcs.utils.common import setup_logging


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def iter_type_images(root: Path) -> Iterable[Tuple[str, Path]]:
    """Expected structure: root/TYPE/*.jpg -> (type, image_path)."""
    for type_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in type_dir.iterdir():
            if is_image(img):
                yield type_dir.name, img


def count_types(root: Path) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = defaultdict(int)
    for typ, _ in iter_type_images(root):
        counts[typ] += 1
    return sorted(((t, n) for t, n in counts.items()), key=lambda x: x[0])


def write_counts_csv(rows: List[Tuple[str, int]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "count"])
        w.writerows(rows)


def plot_types(rows: List[Tuple[str, int]], outdir: Path) -> None:
    """Distribution plots (bar & pie) for the single-level structure."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("matplotlib unavailable, skipping plots (%s)", exc)
        return

    outdir.mkdir(parents=True, exist_ok=True)
    labels = [t for t, _ in rows]
    values = [n for _, n in rows]

    fig1 = plt.figure()
    plt.title("Distribution (bar)")
    plt.bar(labels, values)
    plt.xlabel("Type")
    plt.ylabel("Number of images")
    plt.xticks(rotation=45, ha="right")
    fig1.tight_layout()
    fig1.savefig(str(outdir / "distribution_bar.png"), dpi=150)
    plt.close(fig1)

    fig2 = plt.figure()
    plt.title("Distribution (pie)")
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    fig2.tight_layout()
    fig2.savefig(str(outdir / "distribution_pie.png"), dpi=150)
    plt.close(fig2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze the distribution of a dataset organized as root/TYPE/*.jpg. "
            "Outputs a CSV and optional plots (bar, pie)."
        )
    )
    p.add_argument("--src", type=Path, required=True, help="Dataset root directory")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/plots"),
        help="Output directory for CSV/PNG",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if not args.src.exists():
        logging.error("Source directory does not exist: %s", args.src)
        return

    rows = count_types(args.src)
    if not rows:
        logging.warning(
            "No images found (supported extensions: %s)",
            ", ".join(sorted(IMAGE_EXTS)),
        )
        return
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "distribution.csv"
    write_counts_csv(rows, csv_path)
    logging.info("CSV written: %s", csv_path.resolve())
    if not args.no_plots:
        plot_types(rows, args.out)
        logging.info("Plots written to: %s", args.out.resolve())
    total = sum(n for _, n in rows)
    logging.info("Total images counted: %d", total)


if __name__ == "__main__":
    main()
