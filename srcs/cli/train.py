from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
from keras import mixed_precision

from srcs.dataio.manifest import build_label_mapping, load_manifest, select_items
from srcs.dataio.sequence import ManifestSequence, setup_sequence_logging
from srcs.dataio.transforms import (
    create_inference_transform,
    create_training_transform,
)
from srcs.model.cnn import adapt_normalization, build_leafcnn
from srcs.train.utils import (
    build_callbacks,
    build_loss,
    build_optimizer,
    save_best_variant,
)
from srcs.utils.common import setup_logging
from srcs.utils.system_info import get_optimal_worker_count

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

LOGGER = logging.getLogger(__name__)

REGULARIZED_CFG = {
    "optimizer": "adamw",
    "lr": 0.002,
    "weight_decay": 0.00005,
    "label_smoothing": 0.03,
    "cosine_decay": True,
    "ema_decay": 0.999,
    "clipnorm": 0.5,
    "cache": False,
}

FAST_OVERRIDE = {
    "optimizer": "adam",
    "lr": 3e-3,
    "weight_decay": 0.0,
    "label_smoothing": 0.0,
    "cosine_decay": True,
    "ema_decay": 0.0,
    "clipnorm": 0.0,
    "cache": True,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CNN (standalone Keras) using manifest_split.json"
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/datasets/manifest_augmented.json"),
        help=(
            "Path to manifest_augmented.json "
            "(falls back to manifest_split.json if not found)"
        ),
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-normalization",
        action="store_true",
        help="Disable adaptive normalization layer",
    )
    p.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed_float16 policy",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast prototyping mode (lighter regularization, more aggressive LR)",
    )
    p.add_argument(
        "--scale",
        choices=["tiny", "small", "base"],
        default="base",
        help="Model scale preset: tiny, small, or base (default)",
    )
    # Convenience shorthands (mutually exclusive) mapping to --scale
    mx = p.add_mutually_exclusive_group()
    mx.add_argument("--tiny", action="store_true", help="Shorthand for --scale tiny")
    mx.add_argument("--small", action="store_true", help="Shorthand for --scale small")
    mx.add_argument("--base", action="store_true", help="Shorthand for --scale base")
    p.add_argument(
        "--separable",
        action="store_true",
        help="Use depthwise-separable convolutions (lite)",
    )
    # Optional on-the-fly transform (pipeline)
    p.add_argument(
        "--use-pipeline-transform",
        action="store_true",
        help="Enable PlantCV pipeline transform during loading (train only by default)",
    )
    p.add_argument(
        "--transform-iters",
        type=int,
        default=1,
        help="Number of times to iterate through transform steps",
    )
    p.add_argument(
        "--transform-types",
        type=str,
        default="",
        help="Comma-separated transform types to apply (default = pipeline defaults)",
    )
    p.add_argument(
        "--transform-on-val",
        action="store_true",
        help="Also apply pipeline transform on validation",
    )
    p.add_argument(
        "--transform-config",
        type=Path,
        default=None,
        help="YAML config file for the PlantCV pipeline (uses defaults if omitted)",
    )
    # Sequence logging controls
    p.add_argument(
        "--seq-log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="Logging level for data sequence",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="Root logging level",
    )
    p.add_argument(
        "--seq-log-file",
        type=Path,
        default=Path("artifacts/sequence.log"),
        help="File path for sequence logs",
    )
    p.add_argument(
        "--no-seq-console",
        action="store_true",
        help="Disable console logs from sequence (file only)",
    )
    args = p.parse_args()
    if getattr(args, "tiny", False):
        args.scale = "tiny"
    elif getattr(args, "small", False):
        args.scale = "small"
    elif getattr(args, "base", False):
        args.scale = "base"
    return args


def validate_manifest(args) -> Path:
    """Validate manifest file existence and handle fallback logic.

    Args:
        args: Parsed command line arguments

    Returns:
        Path: Valid manifest file path

    Raises:
        FileNotFoundError: If no valid manifest is found
    """
    if not args.manifest.exists():
        if args.manifest.name == "manifest_augmented.json":
            fallback = args.manifest.with_name("manifest_split.json")
            if fallback.exists():
                LOGGER.warning(
                    "Augmented manifest not found, falling back to: %s",
                    fallback,
                )
                return fallback
            else:
                LOGGER.error("Manifest not found: %s", args.manifest)
                raise FileNotFoundError(f"Manifest not found: {args.manifest}")
        else:
            LOGGER.error("Manifest not found: %s", args.manifest)
            raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    return args.manifest


def prepare_data(manifest_path: Path) -> Tuple[List, List, Dict]:
    """Load and prepare training and validation data.

    Args:
        manifest_path: Path to the manifest file

    Returns:
        Tuple containing train_items, val_items, and label2idx mapping

    Raises:
        ValueError: If insufficient training or validation data
    """
    items = load_manifest(manifest_path)
    train_items = select_items(items, "train")
    val_items = select_items(items, "val")

    if not train_items or not val_items:
        LOGGER.error(
            "Insufficient data (train=%d, val=%d)", len(train_items), len(val_items)
        )
        raise ValueError("Insufficient training or validation data")

    label2idx = build_label_mapping(train_items)
    LOGGER.info("Classes: %d", len(label2idx))

    return train_items, val_items, label2idx


def setup_mixed_precision(no_mixed_precision: bool) -> None:
    """Setup mixed precision training if available and requested.

    Args:
        no_mixed_precision: Flag to disable mixed precision
    """
    if not no_mixed_precision:
        try:
            mixed_precision.set_global_policy("mixed_float16")
            LOGGER.info("Mixed precision enabled (mixed_float16)")
        except (ValueError, AttributeError) as exc:
            LOGGER.warning("Cannot enable mixed precision: %s", exc)


def get_training_config(fast_mode: bool) -> Dict:
    """Get training configuration based on mode.

    Args:
        fast_mode: Whether to use fast training mode

    Returns:
        Training configuration dictionary
    """
    cfg = REGULARIZED_CFG.copy()
    if fast_mode:
        cfg.update(FAST_OVERRIDE)
        LOGGER.info("Mode: FAST (override applied) -> %s", cfg)
    else:
        LOGGER.info("Mode: REGULARIZED -> %s", cfg)

    return cfg


def create_data_sequences(
    train_items: List,
    val_items: List,
    label2idx: Dict,
    args,
    cfg: Dict,
    num_classes: int,
    logger: logging.Logger,
    train_transform=None,
    val_transform=None,
) -> Tuple[Any, Any]:
    """Create training and validation data sequences.

    Args:
        train_items: Training data items
        val_items: Validation data items
        label2idx: Label to index mapping
        args: Command line arguments
        cfg: Training configuration
        num_classes: Number of classes

    Returns:
        Tuple of train_seq and val_seq
    """
    seq_workers = get_optimal_worker_count()
    LOGGER.info("Data loader workers (sequence): %d", seq_workers)

    train_seq = ManifestSequence(
        train_items,
        label2idx,
        args.img_size,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
        limit=None,
        num_classes=num_classes,
        one_hot=cfg["label_smoothing"] > 0.0,
        cache=cfg["cache"],
        workers=seq_workers,
        logger=logger,
        transform=train_transform,
    )
    val_seq = ManifestSequence(
        val_items,
        label2idx,
        args.img_size,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        limit=None,
        num_classes=num_classes,
        one_hot=cfg["label_smoothing"] > 0.0,
        cache=True,
        workers=seq_workers,
        logger=logger,
        transform=val_transform,
    )

    return train_seq, val_seq


def get_model_parameters(scale: str) -> Tuple[List[int], float, float]:
    """Get model parameters based on scale.

    Args:
        scale: Model scale (tiny, small, or base)

    Returns:
        Tuple of widths, drop_block, and drop_top parameters
    """
    if scale == "tiny":
        return [16, 32, 64], 0.10, 0.30
    elif scale == "small":
        return [32, 64, 128], 0.15, 0.35
    else:  # base
        return [32, 64, 128, 256], 0.15, 0.40


def build_and_compile_model(args, cfg: Dict, num_classes: int, train_seq: Any) -> Any:
    """Build and compile the model.

    Args:
        args: Command line arguments
        cfg: Training configuration
        num_classes: Number of classes
        train_seq: Training data sequence

    Returns:
        Compiled Keras model
    """
    widths, drop_block, drop_top = get_model_parameters(args.scale)

    if args.separable:
        LOGGER.info("Using depthwise-separable convolutions (lite mode)")

    model, norm_layer = build_leafcnn(
        num_classes=num_classes,
        img_size=args.img_size,
        use_norm=not args.no_normalization,
        widths=widths,
        drop_block=drop_block,
        drop_top=drop_top,
        l2_reg=cfg["weight_decay"],
        separable=args.separable,
    )
    adapt_normalization(norm_layer, train_seq)

    steps_per_epoch = len(train_seq)
    if cfg["cosine_decay"]:
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg["lr"],
            decay_steps=steps_per_epoch * args.epochs,
        )
        base_lr = lr_schedule
    else:
        base_lr = cfg["lr"]

    opt = build_optimizer(cfg, base_lr)
    loss = build_loss(cfg)

    opt_any: Any = opt
    loss_any: Any = loss
    model.compile(optimizer=opt_any, loss=loss_any, metrics=["accuracy"])

    return model


def create_training_metadata(
    args, cfg: Dict, num_classes: int, train_items: List, val_items: List
) -> Dict:
    """Create metadata dictionary for training run.

    Args:
        args: Command line arguments
        cfg: Training configuration
        num_classes: Number of classes
        train_items: Training data items
        val_items: Validation data items

    Returns:
        Metadata dictionary
    """
    widths, drop_block, drop_top = get_model_parameters(args.scale)
    seq_workers = get_optimal_worker_count()

    return {
        "run": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        "data": {
            "manifest": str(args.manifest.resolve()),
            "img_size": args.img_size,
            "num_classes": num_classes,
            "train_items": len(train_items),
            "val_items": len(val_items),
        },
        "model": {
            "name": "leaf_cnn",
            "scale": args.scale,
            "separable": bool(args.separable),
            "use_normalization": not args.no_normalization,
            "widths": widths,
            "drop_block": drop_block,
            "drop_top": drop_top,
            "l2": cfg["weight_decay"],
        },
        "training": {
            "optimizer": cfg["optimizer"],
            "base_lr": cfg["lr"],
            "cosine_decay": bool(cfg["cosine_decay"]),
            "label_smoothing": cfg["label_smoothing"],
            "ema_decay": cfg["ema_decay"],
            "clipnorm": cfg["clipnorm"],
            "mixed_precision": not args.no_mixed_precision,
        },
        "system": {
            "sequence_workers": seq_workers,
            "backend": os.environ.get("KERAS_BACKEND", "tensorflow"),
        },
    }


def train_and_save_model(
    model: Any,
    train_seq: Any,
    val_seq: Any,
    args,
    cfg: Dict,
    label2idx: Dict,
    meta: Dict,
) -> None:
    """Train the model and save the best variant.

    Args:
        model: Compiled Keras model
        train_seq: Training data sequence
        val_seq: Validation data sequence
        args: Command line arguments
        cfg: Training configuration
        label2idx: Label to index mapping
        meta: Training metadata
    """
    callbacks, ema_cb = build_callbacks(cfg)

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    save_best_variant(
        model,
        val_seq,
        ema_cb,
        out_dir=Path("artifacts/models"),
        label2idx=label2idx,
        history=history,
        meta=meta,
    )


def main() -> None:
    """Main training function with refactored logic."""
    args = parse_args()
    setup_logging(args.log_level)
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        manifest_path = validate_manifest(args)
        train_items, val_items, label2idx = prepare_data(manifest_path)
        num_classes = len(label2idx)

        setup_mixed_precision(args.no_mixed_precision)
        cfg = get_training_config(args.fast)
        # Configure Sequence logger (stderr + file) at requested level
        level = getattr(logging, str(args.seq_log_level).upper(), logging.INFO)
        seq_logger = setup_sequence_logging(
            str(args.seq_log_file), level=level, also_console=not args.no_seq_console
        )
        # Optional pipeline transform (uses transform/config.yaml by default)
        train_transform = None
        val_transform = None
        if args.use_pipeline_transform:
            types = (
                [t.strip() for t in args.transform_types.split(",") if t.strip()]
                if args.transform_types
                else None
            )
            cfg_path = str(args.transform_config or Path("transform/config.yaml"))
            train_transform = create_training_transform(
                config_path=cfg_path,
                transform_types=tuple(types) if types else None,
                apply_augmentation=True,
            )
            if args.transform_on_val:
                val_transform = create_inference_transform(
                    config_path=cfg_path,
                    transform_types=tuple(types) if types else None,
                )

        train_seq, val_seq = create_data_sequences(
            train_items,
            val_items,
            label2idx,
            args,
            cfg,
            num_classes,
            seq_logger,
            train_transform=train_transform,
            val_transform=val_transform,
        )
        model = build_and_compile_model(args, cfg, num_classes, train_seq)
        meta = create_training_metadata(args, cfg, num_classes, train_items, val_items)
        train_and_save_model(model, train_seq, val_seq, args, cfg, label2idx, meta)

    except (FileNotFoundError, ValueError) as e:
        LOGGER.error("Training failed: %s", e)
        return


if __name__ == "__main__":
    main()
