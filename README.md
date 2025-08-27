# leaffliction

Image classification by disease recognition on leaves

# Installation

## 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# (deactivate with: deactivate)
```

## 4. Install dependencies

Dev / tooling dependencies:

```bash
pip install -r dev-requirements.txt
```

Runtime depedencies:

```bash
pip install -r requirements.txt
```

## 5. Configure pre-commit hooks

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

## 6. Project layout and dataset structure

```
images/                 # Dataset root (root/PLANT/CLASS/*.jpg)
srcs/cli/Distribution.py  # Dataset distribution analysis script
# ... in progress
artifacts/plots/        # Output: CSV + plots
```

## 7. Distribution analysis script

The script scans a multi-level dataset organized as:

```
<root>/PLANT/CLASS/image (n).jpg
```

It produces:

-   `artifacts/plots/distribution.csv` (cumulative; updated counts per run)
-   Per-plant bar + pie charts in `artifacts/plots/`

### 7.1 Basic usage

Explicit dataset root:

```bash
python srcs/cli/Distribution.py /path/to/images
```

Filter specific plants (e.g. only Apple and Grape):

```bash
python srcs/cli/Distribution.py --plants Apple Grape
```

Filter specific plants (e.g. only Apple):

```bash
python srcs/cli/Distribution.py --plants Apple
```

No filter used, all plants will be included:

```bash
python srcs/cli/Distribution.py
```

Skip plot generation (CSV only):

```bash
python srcs/cli/Distribution.py --no-plots
```

### 7.2 Output files

| Path                               | Description                                           |
| ---------------------------------- | ----------------------------------------------------- |
| `artifacts/plots/distribution.csv` | Aggregated counts (plant,class,count) updated per run |
| `artifacts/plots/<PLANT>_bar.png`  | Bar chart distribution for the plant                  |
| `artifacts/plots/<PLANT>_pie.png`  | Pie chart distribution for the plant                  |

Re-running merges (overwrites) counts for rows (plant,class) with current scan results, preserving other plants/classes.

### 7.3 Supported extensions

Currently only `.jpg` (case-insensitive). Extend by editing `IMAGE_EXTS` in `srcs/cli/Distribution.py`.

## 8. Logging

Simple colored logging at INFO level (suppresses loud third-party DEBUG like font discovery). No user options required.

## 9. Code style & quality

Tools enforced via pre-commit:

-   black (formatting)
-   isort (imports)
-   flake8 (+ bugbear, comprehensions, pep8-naming)

Run manually:

```bash
black . && isort . && flake8
```
