import multiprocessing as mp
import random
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dataset_components import (
    AugmentationPlanner,
    DistributionAnalyzer,
    ManifestGenerator,
)
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

        self.analyzer = DistributionAnalyzer(manifest_path)
        self.planner = None
        self.manifest_generator = None
        self.plan = {}

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
        counts = self.analyzer.analyze()
        self.analyzer.display_distribution()
        return counts

    def calculate_plan(self):
        counts = self.analyzer.counts
        self.planner = AugmentationPlanner(counts)
        self.plan = self.planner.calculate_plan()
        return self.plan

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
        self.manifest_generator = ManifestGenerator(
            self.analyzer.original_manifest,
            self.source_dir,
            self.target_dir,
            self.workers,
        )
        manifest = self.manifest_generator.generate_augmented_manifest()
        manifest_path = self.manifest_path.parent / "manifest_augmented.json"
        self.manifest_generator.save_manifest(manifest, manifest_path)

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
