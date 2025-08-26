from __future__ import annotations

import argparse
import csv
import logging
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


def iter_images(root: Path, plants: Iterable[str] | None) -> Iterable[Tuple[str, str]]:
    """Yield (plant, class) for each image under root/PLANT/CLASS/*.jpg.

    plants: optional subset of plant directory names to include.
    """
    plant_filter = set(plants) if plants else None
    for plant_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if plant_filter and plant_dir.name not in plant_filter:
            continue
        for class_dir in sorted(d for d in plant_dir.iterdir() if d.is_dir()):
            for img in class_dir.iterdir():
                if is_image(img):
                    yield plant_dir.name, class_dir.name


def count_images(
    root: Path, plants: Iterable[str] | None
) -> List[Tuple[str, str, int]]:
    counts: Dict[Tuple[str, str], int] = {}
    for plant, cls in iter_images(root, plants):
        counts[(plant, cls)] = counts.get((plant, cls), 0) + 1
    return sorted(
        ((p, c, n) for (p, c), n in counts.items()), key=lambda x: (x[0], x[1])
    )


def merge_csv(rows: List[Tuple[str, str, int]], csv_path: Path) -> None:
    """Create or update distribution.csv (columns: plant,class,count)."""
    existing: Dict[Tuple[str, str], int] = {}
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                if r.fieldnames and [h.lower() for h in r.fieldnames] == [
                    "plant",
                    "class",
                    "count",
                ]:
                    for row in r:
                        try:
                            try:
                                existing[(row["plant"], row["class"])] = int(
                                    row["count"]
                                )
                            except KeyError as e:
                                logging.warning("Missing key in CSV row: %s", e)
                        except Exception:
                            continue
                else:
                    logging.warning("Replacing incompatible CSV header: %s", csv_path)
        except Exception as exc:
            logging.warning("Unable to read existing CSV (%s), recreating", exc)
    for plant, cls, cnt in rows:
        existing[(plant, cls)] = cnt
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plant", "class", "count"])
        for plant, cls in sorted(existing):
            w.writerow([plant, cls, existing[(plant, cls)]])


def plot_per_plant(rows: List[Tuple[str, str, int]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("matplotlib unavailable, skipping plots (%s)", exc)
        return

    per_plant: Dict[str, List[Tuple[str, int]]] = {}
    for plant, cls, n in rows:
        per_plant.setdefault(plant, []).append((cls, n))

    out_dir.mkdir(parents=True, exist_ok=True)
    for plant, items in per_plant.items():
        labels = [c for c, _ in items]
        values = [n for _, n in items]

        fig1 = plt.figure()
        plt.title(f"Distribution — {plant} (bar)")
        plt.bar(labels, values)
        plt.xlabel("Class")
        plt.ylabel("Images")
        plt.xticks(rotation=45, ha="right")
        fig1.tight_layout()
        fig1.savefig(str(out_dir / f"{plant}_bar.png"), dpi=150)
        plt.close(fig1)

        fig2 = plt.figure()
        plt.title(f"Distribution — {plant} (pie)")
        plt.pie(values, labels=labels, autopct="%1.1f%%")
        fig2.tight_layout()
        fig2.savefig(str(out_dir / f"{plant}_pie.png"), dpi=150)
        plt.close(fig2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze dataset distribution (multi-level root/PLANT/CLASS/*.jpg). "
            "Usage: Distribution.py [ROOT] [--plants P1 P2]"
        )
    )
    p.add_argument(
        "root", nargs="?", default=None, help="Dataset root (default ./images or CWD)"
    )
    p.add_argument(
        "--plants", nargs="+", default=None, help="Subset of plant names to include"
    )
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return p.parse_args()


def resolve_root(arg_root: str | None) -> Path:
    if arg_root:
        return Path(arg_root)
    default = Path("images")
    return default if default.exists() else Path.cwd()


def main() -> None:
    args = parse_args()
    setup_logging()
    root = resolve_root(args.root)
    if not root.exists():
        logging.error("Root directory does not exist: %s", root)
        return

    all_plants = {p.name for p in root.iterdir() if p.is_dir()}
    plants_filter = None
    if args.plants:
        requested = set(args.plants)
        missing = sorted(requested - all_plants)
        if missing:
            for m in missing:
                logging.warning("Plant directory not found: %s", m)
            logging.error(
                "Aborting due to unknown plant(s). Available: %s",
                ", ".join(sorted(all_plants)),
            )
            return
        plants_filter = requested

    logging.info(
        "Analyzing (root=%s, plants=%s)",
        root.resolve(),
        ",".join(sorted(plants_filter)) if plants_filter else "*ALL*",
    )

    rows = count_images(root, plants_filter)
    if not rows:
        logging.warning(
            "No images found (supported extensions: %s)", ", ".join(sorted(IMAGE_EXTS))
        )
        return

    out_dir = Path("artifacts/plots")
    csv_path = out_dir / "distribution.csv"
    merge_csv(rows, csv_path)
    logging.info("CSV written/updated: %s", csv_path.resolve())

    if not args.no_plots:
        plot_per_plant(rows, out_dir)
        logging.info("Plots written to: %s", out_dir.resolve())

    total = sum(n for _, _, n in rows)
    logging.info("Total images counted: %d", total)


if __name__ == "__main__":
    main()
