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
