import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from srcs.utils.common import get_logger

logger = get_logger(__name__)


class DistributionAnalyzer:
    """Analyze class distribution from either a manifest file or a dataset root.

    If input_path is a directory, it scans root/PLANT/CLASS/* (jpg/jpeg/png/bmp/tiff).
    If input_path is a file, it expects a manifest JSON with "items" entries.
    """

    IMG_EXTS = {".jpg"}

    def __init__(self, input_path):
        self.input_path = Path(input_path)
        self.counts: Dict[str, Dict[str, int]] = {}
        self.original_manifest = None

    def _is_image(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.IMG_EXTS

    def _analyze_dir(self, root: Path) -> Dict[str, Dict[str, int]]:
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        for plant_dir in (d for d in root.iterdir() if d.is_dir()):
            plant = plant_dir.name
            for class_dir in (c for c in plant_dir.iterdir() if c.is_dir()):
                cls = class_dir.name
                n = 0
                for f in class_dir.iterdir():
                    if self._is_image(f):
                        n += 1
                if n > 0:
                    counts[plant][cls] += n
        return counts

    def _analyze_manifest(self, path: Path) -> Dict[str, Dict[str, int]]:
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.original_manifest = manifest
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for item in manifest.get("items", []):
            plant = item.get("plant")
            class_name = item.get("class")
            if plant and class_name:
                counts[plant][class_name] += 1
        return counts

    def analyze(self):
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input not found: {self.input_path}")
        if self.input_path.is_dir():
            self.counts = dict(self._analyze_dir(self.input_path))
        else:
            self.counts = dict(self._analyze_manifest(self.input_path))
        return self.counts

    def display_distribution(self):
        logger.info("Analyzing dataset distribution...")
        for plant, classes in sorted(self.counts.items()):
            logger.info(f"\n[{plant}]")
            for class_name, count in sorted(classes.items()):
                logger.info(f"  {class_name}: {count} images")


class AugmentationPlanner:
    def __init__(self, counts):
        self.counts = counts
        self.plan = {}

    def calculate_plan(self):
        logger.info("Calculating augmentation plan...")

        deficits = {}
        for _plant, classes in self.counts.items():
            plant_max = max(classes.values())
            for class_name, count in classes.items():
                deficit = plant_max - count
                if deficit > 0:
                    deficits[class_name] = deficit

        if not deficits:
            logger.info("Dataset already balanced - no augmentations needed")
            return {}

        transformations = ["flip", "rotate", "skew", "shear", "crop", "distortion"]
        plan = {}

        for class_name, deficit in deficits.items():
            plan[class_name] = {}
            base_per_transform = deficit // 6
            remainder = deficit % 6

            for i, transform in enumerate(transformations):
                count = base_per_transform + (1 if i < remainder else 0)
                if count > 0:
                    plan[class_name][transform] = count

        self.plan = plan
        self._display_plan(deficits)
        return plan

    def _display_plan(self, deficits):
        logger.info("Execution plan:")
        for class_name, deficit in sorted(deficits.items()):
            logger.info(f"  Class: {class_name} - {deficit} images needed")
            if class_name in self.plan:
                for transform_name, count in sorted(self.plan[class_name].items()):
                    logger.info(f"    - {transform_name}: {count} images")


class ManifestGenerator:
    def __init__(self, original_manifest, source_dir, target_dir, workers):
        self.original_manifest = original_manifest
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.workers = workers

    def generate_augmented_manifest(self):
        logger.info("Generating augmented manifest...")

        items = []

        for plant_dir in self.target_dir.iterdir():
            if not plant_dir.is_dir():
                continue
            plant_name = plant_dir.name
            for class_dir in plant_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                for img in class_dir.iterdir():
                    if not img.is_file():
                        continue
                    rel = img.relative_to(self.target_dir)
                    items.append(
                        {
                            "plant": plant_name,
                            "class": class_name,
                            "label": f"{plant_name}__{class_name}",
                            "split": "train",
                            "src": str(img),
                            "id": str(rel),
                            "augmented": "_aug_" in img.stem,
                        }
                    )

        created_at = None
        original_seed = None
        if self.original_manifest and isinstance(self.original_manifest, dict):
            meta = self.original_manifest.get("meta", {})
            created_at = meta.get("created_at")
            original_seed = meta.get("seed")

        augmented_manifest = {
            "meta": {
                "created_at": created_at,
                "augmented_at": datetime.now(timezone.utc).isoformat(),
                "original_seed": original_seed,
                "augmentation_seed": 42,
                "workers": self.workers,
                "src_root": str(self.target_dir),
                "total_images": len(items),
                "original_images": len([i for i in items if not i.get("augmented")]),
                "augmented_images": len([i for i in items if i.get("augmented")]),
            },
            "items": items,
        }

        return augmented_manifest

    def save_manifest(self, manifest, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Augmented manifest saved: {output_path}")
        logger.info(f"  Total images: {manifest['meta']['total_images']}")
        logger.info(f"  Original: {manifest['meta']['original_images']}")
        logger.info(f"  Augmented: {manifest['meta']['augmented_images']}")
