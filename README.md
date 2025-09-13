# leaffliction

Image classification by disease recognition on leaves

# Installation

## 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# (deactivate with: deactivate)
```

## 2. Install dependencies

Dev / tooling dependencies:

```bash
pip install -r dev-requirements.txt
```

Runtime depedencies:

```bash
pip install -r requirements.txt
```
Make the package :

```bash
pip install -e .
```

## 3. Configure pre-commit hooks

```bash
pre-commit install --install-hooks
pre-commit install --hook-type pre-push
```

Test hooks manually on all files:

```bash
pre-commit run --all-files
```

This automatically checks if the code adheres to the specified style and quality guidelines.
It can be run manually at any time.
Otherwise, it will run automatically before each commit.

## 4. Project layout and dataset structure

```
images/                 # Dataset root (root/PLANT/CLASS/*.jpg)
srcs/cli/Distribution.py  # Dataset distribution analysis
srcs/cli/split.py         # Dataset split manifest
srcs/cli/train.py         # Training script
datasets/               # Output: manifests + metadata
  manifest_split.json      # Train/val split manifest
artifacts/plots/          # Output: CSV + plots
  distribution.csv         # Cumulative distribution CSV
  <PLANT>_bar.png          # Per-plant bar chart
  <PLANT>_pie.png          # Per-plant pie chart
artifacts/models/         # Output: trained models + metadata
  leaf_cnn.keras           # Final saved model
  labels.json              # Label→index mapping
  history.json             # Training curves
  confusion_matrix.json    # Validation evaluation
  confusion_matrix.png     # Validation evaluation (visual)
  meta.json                # Run/data/model/training/system metadata
```

## 5. End-to-end workflow (augmented dataset with 80/20 split)

Goal: use the balanced augmented dataset `artifacts/augmented_directory` for training, with 20% validation and 80% training.

Checklist:
- Distribution: show original dataset imbalance.
- Augmentation: balance classes and build `augmented_directory` with a new manifest.
- Split: generate a manifest with 20% validation from the augmented dataset.
- Train: default trainer consumes the augmented manifest and respects splits.
- Predict: evaluate in single or batch mode using the trained model.

Steps:
1) Distribution on original images
   - Inspect imbalance and generate plots/CSV
   - Example:
     ```bash
     python srcs/cli/Distribution.py images
     ```

2) Augmentation to balance classes
   Entrée: `images/`
   Sorties: `artifacts/augmented_directory/` et `artifacts/datasets/manifest_augmented.json`
   Exemple:
     ```bash
     python srcs/cli/Augmentation.py images --output artifacts/augmented_directory
     ```

3) Split the augmented dataset with a 20% validation ratio
   - Writes a new manifest from the augmented directory. Use `--val-ratio 0.2` and target `manifest_augmented.json`.
   - Example:
     ```bash
     python srcs/cli/split.py --src artifacts/augmented_directory \
       --out artifacts/datasets \
       --val-ratio 0.2 \
       --out-manifest artifacts/datasets/manifest_augmented.json
     ```

4) Train using the augmented manifest (default)
   - The trainer defaults to `artifacts/datasets/manifest_augmented.json` and falls back to `manifest_split.json`.
   - Example:
     ```bash
     python -m srcs.cli.train --epochs 20 --batch-size 32 --img-size 224
     ```

5) Predict and evaluate
   - Single image or batch; for evaluation, provide the manifest and split (e.g., `val`).
   - Examples:
     ```bash
     # Single image
     python srcs/cli/predict.py test_images/Unit_test1/Apple_healthy1.JPG

     # Batch with evaluation on the augmented validation split
     python srcs/cli/predict.py artifacts/augmented_directory --batch-mode \
       --evaluate --manifest artifacts/datasets/manifest_augmented.json --split val
     ```

## 6. Documentation

-   Distribution analysis: `docs/cli/distribution.md`
-   Train/validation split: `docs/cli/split.md`
-   Training guide: `docs/training.md`
-   Model architecture: `docs/architecture.md`

## 7. Logging

Simple colored logging at INFO level (suppresses loud third-party DEBUG like font discovery). No user options required.

## 8. Code style & quality

Tools enforced via pre-commit:

-   black (formatting)
-   isort (imports)
-   flake8 (+ bugbear, comprehensions, pep8-naming)

Run manually:

```bash
black . && isort . && flake8
```
