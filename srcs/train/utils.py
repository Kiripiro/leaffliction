from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import keras
import numpy as np

from srcs.utils.confusion_matrix import confusion_matrix

LOGGER = logging.getLogger(__name__)


def build_optimizer(cfg: Dict, base_lr) -> keras.Optimizer:
    opt_kwargs = {}
    if cfg.get("clipnorm", 0.0) > 0:
        opt_kwargs["clipnorm"] = cfg["clipnorm"]
    if cfg.get("optimizer") == "adamw":
        return keras.optimizers.AdamW(
            learning_rate=base_lr,
            weight_decay=cfg.get("weight_decay", 0.0),
            **opt_kwargs,
        )
    return keras.optimizers.Adam(learning_rate=base_lr, **opt_kwargs)


def build_loss(cfg: Dict):
    if cfg.get("label_smoothing", 0.0) > 0:
        return keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg["label_smoothing"]
        )
    return "sparse_categorical_crossentropy"


class EMACallback(keras.callbacks.Callback):
    def __init__(self, decay: float) -> None:
        super().__init__()
        self.decay = float(decay)
        self.ema_weights: List[np.ndarray] | None = None

    def on_train_batch_end(self, batch, logs=None):
        if self.decay <= 0.0:
            return
        model = getattr(self, "model", None)
        if model is None:
            return
        w = model.get_weights()
        if self.ema_weights is None:
            self.ema_weights = [x.copy() for x in w]
        else:
            for i, x in enumerate(w):
                self.ema_weights[i] = (
                    self.decay * self.ema_weights[i] + (1.0 - self.decay) * x
                )


def build_callbacks(
    cfg: Dict,
) -> Tuple[List[keras.callbacks.Callback], EMACallback | None]:
    callbacks: List[keras.callbacks.Callback] = [
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.3, verbose=1),
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
    ]
    ema_cb: EMACallback | None = None
    decay = float(cfg.get("ema_decay", 0.0) or 0.0)
    if decay > 0.0:
        ema_cb = EMACallback(decay)
        callbacks.append(ema_cb)
    return callbacks, ema_cb


def save_best_variant(
    model: keras.Model,
    val_data,
    ema_cb: EMACallback | None,
    out_dir: Path,
    label2idx: Dict[str, int],
    history: keras.callbacks.History,
    meta: Dict[str, Any] | None = None,
) -> None:
    best_acc = model.evaluate(val_data, verbose="auto")[1]
    saved_variant = "base"
    if ema_cb and ema_cb.ema_weights is not None:
        original_weights = model.get_weights()
        model.set_weights(ema_cb.ema_weights)
        ema_acc = model.evaluate(val_data, verbose="auto")[1]
        if float(ema_acc) <= float(best_acc):
            model.set_weights(original_weights)
        else:
            saved_variant = "ema"

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "leaf_cnn.keras"
    model.save(model_path)
    LOGGER.info("Model saved: %s", model_path.resolve())

    with (out_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump({"label2idx": label2idx}, f, indent=2)
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(
            {k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2
        )

    labels_sorted = sorted(label2idx, key=lambda k: label2idx[k])

    try:
        meta_out: Dict[str, Any] = {
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "model_file": str(model_path),
            "labels_file": str((out_dir / "labels.json")),
            "history_file": str((out_dir / "history.json")),
            "confusion_matrix_file": str((out_dir / "confusion_matrix.json")),
            "keras_version": getattr(keras, "__version__", "unknown"),
            "tensorflow_version": getattr(
                __import__("tensorflow"), "__version__", "unknown"
            ),
            "saved_variant": saved_variant,
            "labels": labels_sorted,
        }
        if meta:
            meta_out.update(meta)
        with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)
    except Exception as e:
        LOGGER.warning("Failed to write meta.json: %s", e)

    confusion_matrix(model, val_data, labels_sorted, out_dir)
