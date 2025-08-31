import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from transformations import ImageTransformer


class DatasetBalancer:
    def __init__(
        self,
        manifest_path,
        source_dir="images",
        target_dir="augmented_directory",
        seed=42,
    ):
        self.manifest_path = Path(manifest_path)
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.transformer = ImageTransformer(seed=seed)
        self.counts = {}
        self.plan = {}

    def analyze_distribution(self):
        print("Analyzing dataset distribution...")

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        counts = defaultdict(lambda: defaultdict(int))

        for item in manifest.get("items", []):
            plant = item.get("plant")
            class_name = item.get("class")
            if plant and class_name:
                counts[plant][class_name] += 1

        self.counts = dict(counts)
        self._display_distribution()

    def _display_distribution(self):
        for plant, classes in sorted(self.counts.items()):
            print(f"\n[{plant}]")
            for class_name, count in sorted(classes.items()):
                print(f"  {class_name}: {count} images")

    def calculate_plan(self):
        print("\nCalculating augmentation plan...")

        deficits = {}

        for _plant, classes in self.counts.items():
            plant_max = max(classes.values())

            for class_name, count in classes.items():
                deficit = plant_max - count
                if deficit > 0:
                    deficits[class_name] = deficit

        if not deficits:
            print("Dataset already balanced - no augmentations needed")
            return

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

    def _display_plan(self, deficits):
        print("\nExecution plan:")
        for class_name, deficit in sorted(deficits.items()):
            print(f"  Class: {class_name} - {deficit} images needed")
            if class_name in self.plan:
                for transform_name, count in sorted(self.plan[class_name].items()):
                    print(f"    - {transform_name}: {count} images")

    def _prepare_target_directory(self):
        print(f"\nPreparing target directory: {self.target_dir}")

        if self.target_dir.exists():
            shutil.rmtree(self.target_dir)

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        shutil.copytree(self.source_dir, self.target_dir)
        print(f"Copied all original images to {self.target_dir}")

    def _get_images_by_class(self):
        images_by_class = defaultdict(list)

        for plant_dir in self.target_dir.iterdir():
            if plant_dir.is_dir():
                for class_dir in plant_dir.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        images = list(class_dir.glob("*.JPG")) + list(
                            class_dir.glob("*.jpg")
                        )
                        images_by_class[class_name] = images

        return images_by_class

    def execute_balancing(self):
        if not self.plan:
            print("No augmentation plan - skipping execution")
            return

        self._prepare_target_directory()
        images_by_class = self._get_images_by_class()

        total_augmentations = sum(
            sum(transforms.values()) for transforms in self.plan.values()
        )

        print(
            f"\nStarting augmentation process: {total_augmentations} images to generate"
        )
        completed = 0

        for class_name, transforms in self.plan.items():
            print(f"\n[{class_name}]")

            if class_name not in images_by_class:
                print(f"WARNING: No images found for class '{class_name}'")
                continue

            source_images = images_by_class[class_name]
            class_dir = source_images[0].parent

            for transform_name, count in transforms.items():
                print(f"  {transform_name}: generating {count} images")

                for i in range(count):
                    source_img = random.choice(source_images)

                    suffix = f"_aug_{transform_name}_{i + 1}"
                    new_name = source_img.stem + suffix + source_img.suffix
                    output_path = class_dir / new_name

                    transform_method = getattr(self.transformer, transform_name)
                    if not transform_method(source_img, output_path):
                        print(f"    Failed to generate {output_path}")
                        continue

                    completed += 1
                    if completed % 100 == 0:
                        progress = (completed / total_augmentations) * 100
                        print(
                            f"    [{completed}/{total_augmentations}] "
                            f"{progress:.1f}% complete"
                        )

        print(f"\nAugmentation complete: {completed} new images generated successfully")

    def run(self):
        print("=== Dataset Balancing System ===")

        try:
            self.analyze_distribution()
            self.calculate_plan()
            self.execute_balancing()

            print("\n=== Balancing Complete ===")

        except Exception as e:
            print(f"ERROR: Dataset balancing failed - {e}")
            raise
