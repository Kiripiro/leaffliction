# Training

Reference guide for the manifest-driven Keras training CLI.

## Data inputs

-   The trainer consumes a manifest JSON with items containing absolute `src`, textual `label`, and `split` in {train, val}.
-   Defaults to `artifacts/datasets/manifest_augmented.json`; if not found, it falls back to `artifacts/datasets/manifest_split.json`.

## Requirements

-   Python 3.10+
-   Install runtime deps: `pip install -r requirements.txt`

## Quick start

```bash
python -m srcs.cli.train --epochs 5 --batch-size 8 --small
```

Key options:

-   `--manifest <path>`: override the default manifest path
-   `--scale tiny|small|base` or shorthands `--tiny|--small|--base`
-   `--separable`: use depthwise‑separable convolutions (lite)
-   `--no-normalization`: disable adaptive input normalization
-   `--no-mixed-precision`: disable mixed_float16 policy
-   `--fast`: lighter regularization, higher LR, in-memory cache

## Manifest format (abridged)

Example entry from `artifacts/datasets/manifest_split.json` or `artifacts/datasets/manifest_augmented.json`:

```jsonc
{
    "meta": { "src_root": "/abs/path/images", "seed": 42, ... },
    "items": [
        {
            "plant": "Apple",
            "class": "Apple_rust",
            "label": "Apple__Apple_rust",
            "split": "train",
            "src": "/abs/path/images/Apple/Apple_rust/image (1).JPG",
            "id": "Apple/Apple_rust/image (1).JPG"
        }
    ]
}
```

## What the trainer does

-   Builds a compact CNN (see `docs/architecture.md`) with optional adapted input `Normalization`.
-   Optimizer: AdamW by default (Adam in `--fast`).
-   LR: cosine decay by default; constant when disabled via config.
-   Loss: CategoricalCrossentropy with label smoothing when smoothing > 0.
-   Tracks EMA of weights; after training, evaluates base vs EMA on validation and saves the single best model.

## Outputs (artifacts/models)

-   `leaf_cnn.keras` – final saved model
-   `labels.json` – label→index mapping built from train items
-   `history.json` – training curves
-   `confusion_matrix.json` and `.png` – validation evaluation
-   `meta.json` – run/data/model/training/system metadata

## Notes and tips

-   Mixed precision is enabled by default; disable with `--no-mixed-precision` if you hit numerical issues.
-   Increase batch size if memory allows; decrease on OOM.
-   Keep the default seed for reproducibility, or set `--seed <int>`.

## Related

-   Architecture details: `docs/architecture.md`
-   Data splitting: `docs/cli/split.md`
-   Distribution analysis: `docs/cli/distribution.md`
