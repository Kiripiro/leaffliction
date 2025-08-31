import json
import multiprocessing as mp
import random
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from transformations import ImageTransformer


class DatasetBalancer:
    def __init__(
        self,
        manifest_path,
        source_dir="images",
        target_dir="augmented_directory",
        seed=42,
        workers=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.transformer = ImageTransformer(seed=seed)
        self.workers = self._validate_workers(workers)
        self.counts = {}
        self.plan = {}
        self.original_manifest = None

    def _validate_workers(self, workers):
        max_workers = mp.cpu_count()

        if workers is None:
            workers = max(1, max_workers // 2)
        else:
            workers = max(1, int(workers))

            if workers > max_workers:
                print(
                    f"WARNING: Requested {workers} workers, "
                    f"but only {max_workers} CPUs available"
                )
                print(f"Using {max_workers} workers instead")
                workers = max_workers

        print(f"Using {workers} worker processes (max available: {max_workers})")
        return workers

    def analyze_distribution(self):
        print("Analyzing dataset distribution...")

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

        tasks = []
        for class_name, transforms in self.plan.items():
            if class_name not in images_by_class:
                print(f"WARNING: No images found for class '{class_name}'")
                continue

            source_images = images_by_class[class_name]
            class_dir = source_images[0].parent

            for transform_name, count in transforms.items():
                for i in range(count):
                    source_img = random.choice(source_images)
                    suffix = f"_aug_{transform_name}_{i + 1}"
                    new_name = source_img.stem + suffix + source_img.suffix
                    output_path = class_dir / new_name

                    tasks.append(
                        {
                            "source_img": str(source_img),
                            "output_path": str(output_path),
                            "transform_name": transform_name,
                            "class_name": class_name,
                        }
                    )

        total_tasks = len(tasks)
        print(f"\nStarting parallel augmentation: {total_tasks} images to generate")

        completed = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            future_to_task = {
                executor.submit(_process_single_transformation, task): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    success = future.result()
                    if success:
                        completed += 1
                    else:
                        failed += 1
                        print(f"    Failed: {task['output_path']}")

                    if (completed + failed) % 500 == 0:
                        progress = ((completed + failed) / total_tasks) * 100
                        print(
                            f"    Progress: {completed + failed}/{total_tasks} "
                            f"({progress:.1f}%) - {completed} success, {failed} failed"
                        )

                except Exception as e:
                    failed += 1
                    print(f"    Error processing {task['output_path']}: {e}")

        print(f"\nAugmentation complete: {completed} images generated, {failed} failed")

        self._generate_augmented_manifest()

    def _generate_augmented_manifest(self):
        print("\nGenerating augmented manifest...")

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
                "augmented_at": "2025-08-31T00:00:00+00:00",
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

        manifest_path = self.manifest_path.parent / "manifest_augmented.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(augmented_manifest, f, indent=2, ensure_ascii=False)

        print(f"Augmented manifest saved: {manifest_path}")
        print(f"  Total images: {augmented_manifest['meta']['total_images']}")
        print(f"  Original: {augmented_manifest['meta']['original_images']}")
        print(f"  Augmented: {augmented_manifest['meta']['augmented_images']}")

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


def _process_single_transformation(task):
    try:
        transformer = ImageTransformer(seed=42)
        transform_method = getattr(transformer, task["transform_name"])
        return transform_method(task["source_img"], task["output_path"])
    except Exception:
        return False
