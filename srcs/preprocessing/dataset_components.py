import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from srcs.utils.common import get_logger

logger = get_logger(__name__)


class DistributionAnalyzer:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        self.counts = {}
        self.original_manifest = None

    def analyze(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.original_manifest = manifest
        counts = defaultdict(lambda: defaultdict(int))

        for item in manifest.get("items", []):
            plant = item.get("plant")
            class_name = item.get("class")
            if plant and class_name:
                counts[plant][class_name] += 1

        self.counts = dict(counts)
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

        augmented_items = []

        if self.original_manifest and "items" in self.original_manifest:
            for item in self.original_manifest["items"]:
                updated_item = item.copy()
                old_path = Path(item["src"])

                try:
                    abs_source_dir = self.source_dir.resolve()
                    relative_part = old_path.relative_to(abs_source_dir)
                except ValueError:
                    relative_part = Path(item["id"])

                new_path = self.target_dir / relative_part
                updated_item["src"] = str(new_path)
                updated_item["id"] = str(relative_part)
                augmented_items.append(updated_item)

        for plant_dir in self.target_dir.iterdir():
            if not plant_dir.is_dir():
                continue

            plant_name = plant_dir.name
            for class_dir in plant_dir.iterdir():
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                aug_images = list(class_dir.glob("*_aug_*"))

                for aug_img in aug_images:
                    augmented_item = {
                        "plant": plant_name,
                        "class": class_name,
                        "label": f"{plant_name}__{class_name}",
                        "split": "train",
                        "src": str(aug_img),
                        "id": str(aug_img.relative_to(self.target_dir)),
                        "augmented": True,
                    }
                    augmented_items.append(augmented_item)

        augmented_manifest = {
            "meta": {
                "created_at": (
                    self.original_manifest["meta"]["created_at"]
                    if self.original_manifest
                    else None
                ),
                "augmented_at": datetime.now(timezone.utc).isoformat(),
                "original_seed": (
                    self.original_manifest["meta"].get("seed")
                    if self.original_manifest
                    else None
                ),
                "augmentation_seed": 42,
                "workers": self.workers,
                "src_root": str(self.target_dir),
                "total_images": len(augmented_items),
                "original_images": len(
                    [i for i in augmented_items if not i.get("augmented", False)]
                ),
                "augmented_images": len(
                    [i for i in augmented_items if i.get("augmented", False)]
                ),
            },
            "items": augmented_items,
        }

        return augmented_manifest

    def save_manifest(self, manifest, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Augmented manifest saved: {output_path}")
        logger.info(f"  Total images: {manifest['meta']['total_images']}")
        logger.info(f"  Original: {manifest['meta']['original_images']}")
        logger.info(f"  Augmented: {manifest['meta']['augmented_images']}")
