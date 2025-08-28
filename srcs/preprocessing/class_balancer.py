import json
import shutil
from collections import defaultdict
from pathlib import Path


def count_images_by_category(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    counts = defaultdict(lambda: defaultdict(int))

    for item in manifest.get("items", []):
        plant = item.get("plant")
        class_name = item.get("class")
        if plant and class_name:
            counts[plant][class_name] += 1

    return dict(counts)


def calculate_deficits(counts):
    all_counts = []
    for plant_classes in counts.values():
        all_counts.extend(plant_classes.values())

    target_count = max(all_counts)

    deficits = {}
    for _plant, classes in counts.items():
        for class_name, count in classes.items():
            deficit = target_count - count
            if deficit > 0:
                deficits[class_name] = deficit

    return deficits


def plan_augmentations(deficits):
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

    return plan


def create_augmented_directory():
    source_dir = Path("images")
    target_dir = Path("augmented_directory")

    if target_dir.exists():
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir)
    print(f"Created {target_dir} with all original images")


def main():
    manifest_path = Path("datasets/manifest_split.json")
    if not manifest_path.exists():
        print(f"Error: {manifest_path} does not exist")
        return

    counts = count_images_by_category(manifest_path)

    for plant, classes in sorted(counts.items()):
        print(f"\nPlant: {plant}")
        for class_name, count in sorted(classes.items()):
            print(f"{class_name}: {count}")

    deficits = calculate_deficits(counts)
    if deficits:
        print("\nDeficits:")
        for class_name, deficit in sorted(deficits.items()):
            print(f"{class_name}: {deficit}")

    augmentations_plan = plan_augmentations(deficits)
    if augmentations_plan:
        print("\nAugmentations Plan:")
        for class_name, transforms in sorted(augmentations_plan.items()):
            print(f"{class_name}:")
            for transform, count in sorted(transforms.items()):
                print(f"  {transform}: {count}")

    create_augmented_directory()


if __name__ == "__main__":
    main()
