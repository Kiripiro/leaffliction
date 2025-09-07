from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from srcs.utils.common import setup_logging

LOGGER = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg"}


@dataclass(frozen=True)
class ImgItem:
    """Represents a single source image and its stable relative identifier."""

    plant: str
    cls: str  # class
    label: str  # e.g. Apple__rust
    src: Path  # absolute source path
    rel_id: str  # stable relative id: plant/class/filename


def is_image(path: Path) -> bool:
    """Return True if path is a file with an allowed image extension."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def scan_dataset(root: Path) -> List[ImgItem]:
    """Scan dataset hierarchy root/PLANT/CLASS and collect image items."""
    items: List[ImgItem] = []
    if not root.exists():
        return items
    for plant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        plant = plant_dir.name
        for class_dir in sorted(c for c in plant_dir.iterdir() if c.is_dir()):
            cls = class_dir.name
            label = f"{plant}__{cls}"
            for img in sorted(class_dir.iterdir()):
                if not is_image(img):
                    continue
                rel = Path(plant) / cls / img.name
                items.append(
                    ImgItem(
                        plant=plant,
                        cls=cls,
                        label=label,
                        src=img.resolve(),
                        rel_id=str(rel).replace("\\", "/"),
                    )
                )
    return items


def allocate_validation_counts(
    by_label_counts: Mapping[str, int], min_total: int
) -> Dict[str, int]:
    """Compute per-label validation allocation.

    Rules:
    - Keep ≥1 training image per label (so capacity = max(n-1, 0)).
    - Distribute validation images as evenly as possible (round‑robin) until
      reaching min_total or exhausting capacity.
    - If total capacity < min_total allocate full capacity and warn.
    """
    if min_total < 0:
        raise ValueError("min_total must be >= 0")

    labels = sorted(by_label_counts)
    capacity: Dict[str, int] = {
        lab: (by_label_counts[lab] - 1 if by_label_counts[lab] > 1 else 0)
        for lab in labels
    }
    eligible = [lab for lab in labels if capacity[lab] > 0]
    total_capacity = sum(capacity[lab] for lab in eligible)

    if not eligible or total_capacity <= 0:
        if not eligible:
            LOGGER.warning("No classes with capacity for validation (all singleton?).")
        return dict.fromkeys(labels, 0)

    alloc: Dict[str, int] = dict.fromkeys(labels, 0)
    target = min_total

    if total_capacity < target:
        for lab in eligible:
            alloc[lab] = capacity[lab]
        LOGGER.warning(
            "Total capacity (%d) less than requested min_val (%d); using all capacity.",
            total_capacity,
            target,
        )
        return alloc

    remaining = target
    active = eligible.copy()
    while remaining > 0 and active:
        for lab in list(active):
            if remaining == 0:
                break
            if alloc[lab] < capacity[lab]:
                alloc[lab] += 1
                remaining -= 1
            if alloc[lab] >= capacity[lab]:
                active.remove(lab)

    if remaining > 0:
        LOGGER.warning(
            "Could not exactly reach requested min_val=%d; allocated %d.",
            target,
            target - remaining,
        )
    return alloc


def build_split_map(
    items_by_label: Mapping[str, List[ImgItem]],
    alloc_val: Mapping[str, int],
    seed: int,
) -> Dict[str, str]:
    """Return mapping rel_id -> 'train' | 'val' deterministically."""
    rng = random.Random(seed)
    split_map: Dict[str, str] = {}
    for lab, items in items_by_label.items():
        files = list(items)
        rng.shuffle(files)
        k_val = min(alloc_val.get(lab, 0), len(files))
        val_ids = {f.rel_id for f in files[:k_val]}
        for f in files:
            split_map[f.rel_id] = "val" if f.rel_id in val_ids else "train"
    return split_map


def write_manifest(
    out_path: Path,
    items: Iterable[ImgItem],
    split_map: Mapping[str, str],
    src_root: Path,
    seed: int,
    min_val: int,
) -> None:
    """Write manifest_split.json with meta + per-item entries."""
    now = datetime.now(tz=timezone.utc).isoformat()
    manifest = {
        "meta": {
            "created_at": now,
            "seed": seed,
            "strategy": "minimal-even >= min_val",
            "min_val": min_val,
            "src_root": str(src_root.resolve()),
        },
        "items": [
            {
                "plant": it.plant,
                "class": it.cls,
                "label": it.label,
                "split": split_map[it.rel_id],
                "src": it.src.as_posix(),
                "id": it.rel_id,
            }
            for it in items
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    LOGGER.info("Manifest written: %s", out_path.resolve())


def write_summary(
    out_path: Path,
    items_by_label: Mapping[str, List[ImgItem]],
    split_map: Mapping[str, str],
) -> None:
    """Write per-label summary CSV with train/val counts and totals."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_train = 0
    n_val = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "n_train", "n_val", "total"])
        for lab in sorted(items_by_label):
            items = items_by_label[lab]
            val_count = sum(1 for it in items if split_map[it.rel_id] == "val")
            train_count = len(items) - val_count
            writer.writerow([lab, train_count, val_count, len(items)])
            n_train += train_count
            n_val += val_count
        writer.writerow(["_TOTAL_", n_train, n_val, n_train + n_val])
        LOGGER.info(
            "Summary CSV written: %s (train=%d, val=%d)",
            out_path.resolve(),
            n_train,
            n_val,
        )


def log_allocation(alloc: Mapping[str, int], counts: Mapping[str, int]) -> None:
    """Log allocation details per label for transparency (debug aid)."""
    lines = ["Validation allocation per label (val/total):"]
    for lab in sorted(counts):
        lines.append(f"  {lab}: {alloc.get(lab, 0)}/{counts[lab]}")
    LOGGER.info("\n".join(lines))


def validate_source_structure(root: Path) -> None:
    """Emit warnings for potential structural issues (empty dirs, no images)."""
    if not root.exists():
        LOGGER.error("Source directory does not exist: %s", root)
        sys.exit(1)
    if not any(p.is_dir() for p in root.iterdir()):
        LOGGER.error("No subdirectories found under source root: %s", root)
        sys.exit(1)
    empty_class_dirs: List[Path] = []
    total_dirs = 0
    for plant_dir in (p for p in root.iterdir() if p.is_dir()):
        for class_dir in (c for c in plant_dir.iterdir() if c.is_dir()):
            total_dirs += 1
            if not any(is_image(f) for f in class_dir.iterdir() if f.is_file()):
                empty_class_dirs.append(class_dir)
    if total_dirs == 0:
        LOGGER.error("No class directories found inside plants under: %s", root)
        sys.exit(1)
    if empty_class_dirs:
        LOGGER.warning(
            "Empty class directories (ignored): %s",
            ", ".join(d.as_posix() for d in empty_class_dirs[:15])
            + (" ..." if len(empty_class_dirs) > 15 else ""),
        )


def reset_split_outputs(out_root: Path) -> None:
    """Remove previous split artifacts (train/, val/, manifest & summary) if present.

    Safety: only removes the specific known artifacts inside out_root. Other files/
    directories at the same level are left untouched.
    """
    targets = [
        out_root / "manifest_split.json",
        out_root / "split_summary.csv",
    ]
    removed: List[Path] = []
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
            removed.append(t)
        elif t.is_file():
            t.unlink()
            removed.append(t)
    if removed:
        LOGGER.info(
            "Reset: removed %d previous artifacts (%s)",
            len(removed),
            ", ".join(sorted(p.name for p in removed)),
        )
    else:
        LOGGER.info("Reset: nothing to remove (no prior artifacts found).")


def parse_args() -> argparse.Namespace:
    """Configure and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Minimal balanced split: smallest validation set meeting --min-val "
            "(even across classes, keeps ≥1 train). Writes manifest + summary only."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("images"),
        help="Original images root (read-only).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets"),
        help="Output root for manifest / summary / views.",
    )
    parser.add_argument(
        "--min-val",
        type=int,
        default=100,
        help="Minimum total number of validation images across all classes.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Deterministic random seed for selection."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Remove existing split outputs (train/, val/, manifest, summary) "
            "before running."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    LOGGER.debug("Arguments: %s", vars(args))
    try:
        validate_source_structure(args.src)
        if args.reset:
            reset_split_outputs(args.out)
        items = scan_dataset(args.src)
        if not items:
            LOGGER.error(
                "No images discovered after scan (extensions: %s)",
                ", ".join(sorted(IMAGE_EXTS)),
            )
            sys.exit(1)
        items_by_label: Dict[str, List[ImgItem]] = {}
        for it in items:
            items_by_label.setdefault(it.label, []).append(it)
        counts = {lab: len(lst) for lab, lst in items_by_label.items()}
        alloc_val = allocate_validation_counts(counts, args.min_val)
        log_allocation(alloc_val, counts)
        split_map = build_split_map(items_by_label, alloc_val, args.seed)
        if len(split_map) != len(items):
            LOGGER.error(
                "Split map size mismatch (%d vs %d)", len(split_map), len(items)
            )
            sys.exit(1)
        manifest_path = args.out / "manifest_split.json"
        write_manifest(
            manifest_path,
            items,
            split_map,
            src_root=args.src,
            seed=args.seed,
            min_val=args.min_val,
        )
        summary_path = args.out / "split_summary.csv"
        write_summary(summary_path, items_by_label, split_map)
        LOGGER.info("Split completed.")
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        LOGGER.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
