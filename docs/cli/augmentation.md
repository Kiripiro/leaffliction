# Image Augmentation System

## Overview

The Leaffliction project implements a comprehensive image augmentation system designed to balance datasets for plant disease classification. The system provides both single image transformation capabilities and large-scale dataset balancing through various augmentation techniques.

## Architecture

The augmentation system consists of several interconnected components:

1. **ImageAugmenter** (`srcs/preprocessing/image_augmenter.py`) - Core transformation engine
2. **DatasetBalancer** (`srcs/preprocessing/dataset_balancer.py`) - High-level orchestration for dataset operations
3. **ImageLoader** (`srcs/utils/image_utils.py`) - Image I/O utilities
4. **CLI Interface** (`srcs/cli/Augmentation.py`) - Command-line interface

## Available Transformations

The system supports six types of image transformations:

### 1. Flip
- **Function**: `flip(image_path, output_path)`
- **Operation**: Random horizontal or vertical flip
- **Parameters**: 50% chance for each direction
- **Use case**: Increases dataset variety by mirroring images

### 2. Rotate
- **Function**: `rotate(image_path, output_path)`
- **Operation**: Random rotation within angle range
- **Parameters**: Angle between -30° and +30°, white background fill
- **Use case**: Simulates different viewing angles

### 3. Skew
- **Function**: `skew(image_path, output_path)`
- **Operation**: Perspective transformation with skewing
- **Parameters**: Skew factor between 0.05 and 0.15
- **Use case**: Mimics perspective distortion from different camera positions

### 4. Shear
- **Function**: `shear(image_path, output_path)`
- **Operation**: Affine transformation with shearing
- **Parameters**: Shear factor between -0.2 and 0.2, random horizontal/vertical direction
- **Use case**: Creates slanted appearance effects

### 5. Crop
- **Function**: `crop(image_path, output_path)`
- **Operation**: Random crop with resize back to original dimensions
- **Parameters**: Crop ratio between 80% and 95% of original size
- **Use case**: Focuses on different parts of the image while maintaining size

### 6. Distortion
- **Function**: `distortion(image_path, output_path)`
- **Operation**: Gaussian noise addition with contrast adjustment
- **Parameters**: Noise level of 5, contrast cutoff between 0-2%
- **Use case**: Simulates camera sensor noise and lighting variations

## Execution Methods

### Method 1: Single Image Augmentation

Transform a single image with all available augmentations:

```bash
# Basic usage
leaffliction-augment path/to/image.jpg

# With custom output directory
leaffliction-augment path/to/image.jpg --output my_examples

# With custom seed for reproducibility
leaffliction-augment path/to/image.jpg --seed 12345
```

**Output**: Creates 7 files (1 original + 6 transformations) in the output directory.

### Method 2: Dataset Balancing

Balance an entire dataset using manifest-driven augmentation:

```bash
# Basic dataset balancing
leaffliction-augment datasets/manifest_split.json images/

# With custom output directory
leaffliction-augment datasets/manifest_split.json images/ --output balanced_dataset

# With parallel processing
leaffliction-augment datasets/manifest_split.json images/ --workers 8

# With custom seed
leaffliction-augment datasets/manifest_split.json images/ --seed 42
```

**Process**:
1. Analyzes class distribution from manifest
2. Calculates augmentation plan to balance classes
3. Copies original dataset to target directory
4. Applies augmentations in parallel using multiple workers
5. Generates new manifest for augmented dataset
6. Creates distribution analysis reports

### Method 3: Programmatic Usage

Direct integration in Python code:

```python
from srcs.preprocessing.image_augmenter import ImageAugmenter
from srcs.preprocessing.dataset_balancer import DatasetBalancer

# Single transformations
augmenter = ImageAugmenter(seed=42)
augmenter.flip("input.jpg", "output_flip.jpg")
augmenter.rotate("input.jpg", "output_rotate.jpg")

# Dataset balancing
balancer = DatasetBalancer(
    manifest_path="manifest.json",
    source_dir="images/",
    target_dir="balanced/",
    seed=42,
    workers=4
)
balancer.run()
```

### Method 4: Component-Level Access

Access individual system components:

```python
# Distribution analysis only
from srcs.preprocessing.dataset_components import DistributionAnalyzer
analyzer = DistributionAnalyzer("manifest.json")
counts = analyzer.analyze()
analyzer.display_distribution()

# Augmentation planning only
from srcs.preprocessing.dataset_components import AugmentationPlanner
planner = AugmentationPlanner(counts)
plan = planner.calculate_plan()

# Manifest generation only
from srcs.preprocessing.dataset_components import ManifestGenerator
generator = ManifestGenerator(original_manifest, source_dir, target_dir)
new_manifest = generator.generate_augmented_manifest()
```

## Configuration Options

### Random Seed Control
All augmentation operations support seeded randomization for reproducible results:
- Set via `--seed` parameter in CLI
- Pass to `ImageAugmenter(seed=value)` constructor
- Each parallel worker gets unique derived seed

### Parallel Processing
Dataset balancing supports configurable parallelization:
- `--workers N` to set worker count
- Automatic CPU detection with optimal defaults
- Progress reporting every 500 completed images

### Output Structure
The system maintains directory structure:
```
output_directory/
├── Plant_A/
│   ├── healthy/
│   │   ├── original_image1.jpg
│   │   ├── flip_aug_1.jpg
│   │   └── rotate_aug_1.jpg
│   └── diseased/
└── Plant_B/
    └── ...
```

## Technical Implementation

### Image Processing Pipeline
1. **Loading**: PIL-based image loading with RGB conversion
2. **Transformation**: Specific algorithm application with parameter randomization
3. **Saving**: JPEG output with 95% quality preservation
4. **Error Handling**: Comprehensive exception handling with detailed logging

### Performance Optimization
- **Multiprocessing**: Parallel transformation execution
- **Memory Management**: Per-worker process isolation
- **Progress Tracking**: Real-time completion monitoring
- **Worker Auto-scaling**: CPU-based worker count optimization

### Quality Assurance
- **Deterministic Results**: Seed-based reproducibility
- **Error Recovery**: Graceful failure handling with detailed logging
- **Validation**: Input path and format verification
- **Resource Management**: Automatic cleanup and directory preparation

## Output Analysis

After dataset balancing, the system automatically generates:
- `balanced_distribution.csv`: Class count statistics
- Distribution plots in `artifacts/distribution/`
- Updated manifest file (`manifest_augmented.json`)
- Detailed execution logs with success/failure metrics

This comprehensive augmentation system ensures robust dataset preparation for machine learning training while maintaining data quality and providing extensive configurability for different use cases.
