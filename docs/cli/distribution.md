# Distribution CLI

Analyze dataset distribution (root/PLANT/CLASS/\*.jpg). Produces cumulative CSV and per-plant bar + pie charts.

## Usage

Basic (auto-detect `images/` or CWD):

```bash
python srcs/cli/Distribution.py
```

Explicit root:

```bash
python srcs/cli/Distribution.py /path/to/images
```

Filter plants:

```bash
python srcs/cli/Distribution.py --plants Apple Grape
```

CSV only (skip plots):

```bash
python srcs/cli/Distribution.py --no-plots
```

## Output

| Path                               | Description                                           |
| ---------------------------------- | ----------------------------------------------------- |
| `artifacts/plots/distribution.csv` | Aggregated counts (plant,class,count) updated per run |
| `artifacts/plots/<PLANT>_bar.png`  | Bar chart per plant                                   |
| `artifacts/plots/<PLANT>_pie.png`  | Pie chart per plant                                   |

Re-running merges counts for existing (plant,class) rows.

## Supported extensions

Only `.jpg` for now (case-insensitive). Update `IMAGE_EXTS` in `srcs/cli/Distribution.py` to extend.

## Notes

-   Empty plants/classes just produce fewer rows.
-   Logging at INFO by default (configured via `setup_logging`).
