# Split CLI

Minimal train/validation metadata split (no file copying). Writes only:

```
datasets/
  manifest_split.json
  split_summary.csv
```

## Goal & Strategy

Allocate the smallest validation set whose size ≥ `--min-val` (default 100) with:

1. Even distribution across labels (difference ≤ 1 when capacity allows)
2. At least 1 training image kept per label (labels with 1 image go fully to train)
3. If capacity < requested, use all capacity and warn

Manifest meta strategy: `"minimal-even >= min_val"`.

## Arguments

| Arg         | Default    | Description                                 |
| ----------- | ---------- | ------------------------------------------- |
| `--src`     | `images`   | Dataset root (`PLANT/CLASS/*.jpg`)          |
| `--out`     | `datasets` | Output directory for metadata files         |
| `--min-val` | `100`      | Requested minimum total validation images   |
| `--seed`    | `42`       | Shuffle seed (deterministic)                |
| `--reset`   | (flag)     | Remove previous metadata outputs before run |

## Usage

Default roots:

```bash
python srcs/cli/split.py
```

Custom min validation:

```bash
python srcs/cli/split.py --min-val 150
```

Custom paths & seed:

```bash
python srcs/cli/split.py --src /data/leaves --out /data/metadata --seed 7
```

Reset before re-splitting:

```bash
python srcs/cli/split.py --reset
```

## Manifest Example (abridged)

```jsonc
{
    "meta": {
        "created_at": "2025-08-27T11:22:33.123456+00:00",
        "seed": 42,
        "strategy": "minimal-even >= min_val",
        "min_val": 100,
        "src_root": "/abs/path/images"
    },
    "items": [
        {
            "plant": "Apple",
            "class": "Apple_rust",
            "label": "Apple__Apple_rust",
            "split": "val",
            "src": "/abs/path/images/Apple/Apple_rust/image (1).JPG",
            "id": "Apple/Apple_rust/image (1).JPG"
        }
    ]
}
```

## Summary CSV

`label,n_train,n_val,total` plus `_TOTAL_` aggregate row.

## Edge Cases & Warnings

-   Empty class dirs ignored with warning
-   Single-image label cannot contribute to validation
-   Insufficient capacity -> allocate all possible, warn
-   Deterministic with same seed & dataset

## Typical Workflow

1. Run distribution analysis (`Distribution.py`)
2. Run split to generate metadata
3. Training code loads `manifest_split.json` & filters by `split`
