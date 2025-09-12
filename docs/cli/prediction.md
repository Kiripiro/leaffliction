# Prediction System

## Overview

The Leaffliction prediction system provides comprehensive plant disease classification capabilities using trained deep learning models. The system supports both single image classification and batch processing with evaluation metrics, visualization tools, and comprehensive output analysis.

## Architecture

The prediction system consists of several interconnected components:

1. **Predictor** (`srcs/predict/predictor.py`) - Core prediction orchestration
2. **ModelLoader** (`srcs/predict/model_loader.py`) - Model and metadata management
3. **ImageProcessor** (`srcs/predict/image_processor.py`) - Image preprocessing pipeline
4. **PredictionEvaluator** (`srcs/predict/evaluation.py`) - Performance evaluation against ground truth
5. **PredictionVisualizer** (`srcs/predict/prediction_visualizer.py`) - Visualization and montage generation
6. **CLI Interface** (`srcs/cli/predict.py`) - Command-line interface

## Core Components

### ModelLoader
- **Purpose**: Load trained Keras models and associated metadata
- **Key Features**:
  - Automatic metadata loading from `meta.json`
  - Class label mapping
  - Model configuration extraction (image size, number of classes)
  - Keras model instantiation

### ImageProcessor
- **Purpose**: Standardize input images for model inference
- **Pipeline**:
  1. Load image using PIL with RGB conversion
  2. Resize to model's expected dimensions (default: 224x224)
  3. Normalize pixel values to [0,1] range
  4. Add batch dimension for model input
- **Output**: Original array and processed array for visualization

### Predictor
- **Purpose**: Coordinate inference operations
- **Capabilities**:
  - Single image prediction with confidence scores
  - Batch processing with optimized numpy operations
  - Full probability distribution for all classes
  - Error handling and validation

## Execution Methods

### Method 1: Single Image Prediction

Classify a single image and display results:

```bash
# Basic prediction
leaffliction-predict path/to/image.jpg

# With custom model directory
leaffliction-predict path/to/image.jpg --learnings-dir custom_models/

# With output visualization
leaffliction-predict path/to/image.jpg --output-dir results/
```

**Output**:
- Prediction result with confidence score
- Top 3 class probabilities in console
- Visualization montage showing original and processed images
- Automatic image viewer launch on supported platforms

### Method 2: Batch Processing

Process multiple images efficiently:

```bash
# Process entire directory
leaffliction-predict images_directory/ --batch-mode

# With JSON output
leaffliction-predict images_directory/ --batch-mode --json-output results.json

# With custom output directory
leaffliction-predict images_directory/ --batch-mode --output-dir batch_results/

# With parallel processing optimization
leaffliction-predict images_directory/ --batch-mode --learnings-dir models/
```

**Process**:
1. Automatic image file discovery (supports .jpg, .jpeg, .png, .bmp, .tiff)
2. Batch preprocessing for optimized inference
3. Parallel prediction using numpy vectorization
4. Summary statistics generation
5. JSON export with detailed results
6. Optional visualization dashboard creation

### Method 3: Evaluation Mode

Evaluate model performance against ground truth:

```bash
# Evaluate with manifest file
leaffliction-predict test_images/ --batch-mode --evaluate --manifest manifest.json

# Evaluate specific split
leaffliction-predict test_images/ --batch-mode --evaluate --manifest manifest.json --split val

# Full evaluation with all outputs
leaffliction-predict test_images/ --batch-mode --evaluate --manifest manifest.json \
  --json-output evaluation_results.json --output-dir evaluation_output/
```

**Evaluation Features**:
- Accuracy, precision, recall, F1-score computation
- Per-class performance metrics
- Confusion matrix generation
- Detailed prediction analysis with correctness flags
- Comprehensive evaluation reports in JSON format

### Method 4: Programmatic Usage

Direct integration in Python applications:

```python
from srcs.predict.predictor import Predictor
from pathlib import Path

# Initialize predictor
predictor = Predictor("artifacts/models")
predictor.load()

# Single prediction
result = predictor.predict_single("image.jpg")
print(f"Prediction: {result['top_prediction']} ({result['confidence']:.2%})")

# Batch prediction
image_paths = [Path("img1.jpg"), Path("img2.jpg")]
results = predictor.predict_batch(image_paths)

# Access detailed results
for result in results:
    print(f"Image: {result['image_path']}")
    print(f"Class: {result['top_prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"All probabilities: {result['all_probabilities']}")
```

### Method 5: Advanced Evaluation

Component-level evaluation access:

```python
from srcs.predict.evaluation import PredictionEvaluator, evaluate_from_manifest
from srcs.predict.predictor import Predictor

# Custom evaluation
predictor = Predictor("models/")
predictor.load()

evaluator = PredictionEvaluator(predictor)
metrics = evaluator.evaluate_predictions(
    image_paths=["img1.jpg", "img2.jpg"],
    true_labels=["healthy", "diseased"],
    output_dir="eval_results/"
)

# Manifest-based evaluation
metrics = evaluate_from_manifest(
    predictor=predictor,
    manifest_path=Path("test_manifest.json"),
    split="test",
    output_dir=Path("evaluation_output/")
)
```

## Output Formats

### Single Image Results
```json
{
  "image_path": "image.jpg",
  "top_prediction": "leaf_spot",
  "confidence": 0.87,
  "all_probabilities": {
    "healthy": 0.05,
    "leaf_spot": 0.87,
    "rust": 0.08
  }
}
```

### Batch Processing Results
```json
{
  "batch_results": [
    {
      "image_path": "img1.jpg",
      "top_prediction": "healthy",
      "confidence": 0.92,
      "all_probabilities": {...}
    }
  ],
  "summary": {
    "total_images": 100,
    "processing_time": "15.3s",
    "average_confidence": "85.2%",
    "prediction_distribution": {
      "healthy": 45,
      "leaf_spot": 32,
      "rust": 23
    }
  }
}
```

### Evaluation Metrics
```json
{
  "metrics": {
    "accuracy": 0.89,
    "precision_macro": 0.87,
    "recall_macro": 0.85,
    "f1_macro": 0.86,
    "per_class_metrics": {
      "healthy": {"precision": 0.91, "recall": 0.88, "f1": 0.89},
      "diseased": {"precision": 0.83, "recall": 0.82, "f1": 0.83}
    }
  },
  "evaluation_info": {
    "total_images": 200,
    "valid_predictions": 198,
    "class_labels": ["healthy", "leaf_spot", "rust"]
  }
}
```

## Visualization Features

### Prediction Montages
- **Content**: Side-by-side original and processed images
- **Annotations**: Prediction class and confidence score
- **Format**: PNG with optimized dimensions
- **Labels**: "Original" and "Processed" image indicators

### Batch Dashboards
- **Summary Statistics**: Processing metrics and distribution charts
- **Confusion Matrix**: Interactive heatmap for evaluation results
- **Performance Graphs**: Accuracy trends and confidence distributions
- **Export Options**: PNG and JSON formats

## Technical Implementation

### Image Processing Pipeline
1. **Loading**: PIL-based RGB image loading with format validation
2. **Preprocessing**: Resize to model dimensions with LANCZOS resampling
3. **Normalization**: Pixel value scaling to [0,1] range
4. **Batching**: Numpy array stacking for efficient inference

### Model Integration
- **Framework**: Keras/TensorFlow model loading
- **Metadata**: JSON-based configuration with class mappings
- **Input Validation**: Shape and format verification
- **Output Processing**: Softmax probability interpretation

### Performance Optimization
- **Batch Processing**: Vectorized operations for multiple images
- **Memory Management**: Efficient array operations and cleanup
- **Error Recovery**: Graceful handling of invalid images
- **Progress Tracking**: Real-time processing status updates

### Quality Assurance
- **Input Validation**: Path existence and format verification
- **Error Handling**: Comprehensive exception management with logging
- **Result Validation**: Probability distribution verification
- **Resource Management**: Automatic cleanup and memory optimization

## Configuration Options

### Model Directory Structure
```
artifacts/models/
├── meta.json          # Model metadata and configuration
├── model.keras        # Trained Keras model file
└── training_log.json  # Optional training history
```

### Required Metadata Format
```json
{
  "model_file": "artifacts/models/model.keras",
  "labels": ["healthy", "leaf_spot", "rust", "blight"],
  "data": {
    "img_size": 224,
    "num_classes": 4
  },
  "training_info": {
    "dataset": "plant_disease_v2",
    "epochs": 50,
    "accuracy": 0.89
  }
}
```

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## Error Handling

The system provides comprehensive error handling for:
- **Missing Files**: Model, metadata, or image files
- **Invalid Formats**: Unsupported image formats or corrupted files
- **Model Errors**: Loading failures or incompatible architectures
- **Processing Errors**: Memory issues or computation failures
- **Evaluation Errors**: Manifest format issues or missing ground truth

All errors are logged with detailed context and provide actionable guidance for resolution.

This prediction system ensures robust and accurate plant disease classification with extensive configurability and comprehensive output analysis for various use cases ranging from single image classification to large-scale dataset evaluation.
