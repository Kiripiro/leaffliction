from __future__ import annotations

from typing import Any, List, Tuple

import keras
from keras import layers, regularizers


def _se_block(x, se_ratio: int = 8) -> Any:
    """Squeeze-and-Excitation block."""
    in_channels = x.shape[-1]
    if in_channels is None:
        return x
    se = layers.GlobalAveragePooling2D(keepdims=True)(x)
    se = layers.Conv2D(int(in_channels // se_ratio), 1, activation="relu")(se)
    se = layers.Conv2D(int(in_channels), 1, activation="sigmoid")(se)
    return layers.Multiply()([x, se])


def _conv_block(x, filters: int, separable: bool, l2_reg: float) -> Any:
    reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    if separable:
        x = layers.SeparableConv2D(
            filters, 3, padding="same", use_bias=False, kernel_regularizer=reg
        )(x)
    else:
        x = layers.Conv2D(
            filters, 3, padding="same", use_bias=False, kernel_regularizer=reg
        )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def _res_block(x, filters: int, separable: bool, l2_reg: float, use_se: bool) -> Any:
    """Residual block with optional SE and (separable) convs."""
    shortcut = x
    y = _conv_block(x, filters, separable, l2_reg)
    y = _conv_block(y, filters, separable, l2_reg)
    if use_se:
        y = _se_block(y)
    # Match channels for residual add
    if shortcut.shape[-1] != y.shape[-1]:
        proj = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        proj = layers.BatchNormalization()(proj)
        shortcut = proj
    y = layers.Add()([shortcut, y])
    y = layers.Activation("relu")(y)
    return y


def build_leafcnn(
    *,
    num_classes: int,
    img_size: int = 224,
    use_norm: bool = True,
    widths: List[int] | None = None,
    drop_block: float = 0.15,
    drop_top: float = 0.40,
    l2_reg: float = 0.0,
    separable: bool = False,
    augment: bool = True,
    use_se: bool = True,
) -> Tuple[keras.Model, layers.Layer | None]:
    """Build a compact CNN for leaf classification.

    Returns (model, norm_layer) where norm_layer may be None if use_norm=False.
    """
    widths = widths or [32, 64, 128]
    inputs = layers.Input((img_size, img_size, 3))

    norm_layer = None
    x = inputs
    if augment:
        aug = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomContrast(0.1),
            ],
            name="augment",
        )
        x = aug(x)
    if use_norm:
        norm_layer = layers.Normalization(axis=-1, name="input_norm")
        x = norm_layer(x)

    # Stem
    x = _conv_block(x, widths[0], separable, l2_reg)

    # Stacked stages with residual blocks
    for f in widths:
        x = _res_block(x, f, separable, l2_reg, use_se)
        if drop_block and drop_block > 0:
            x = layers.SpatialDropout2D(rate=drop_block)(x)
        x = layers.MaxPool2D(pool_size=2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    if drop_top and drop_top > 0:
        x = layers.Dropout(drop_top)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="leaf_cnn")
    return model, norm_layer


def adapt_normalization(norm_layer: layers.Layer | None, train_seq) -> None:
    """Adapt the Normalization layer on a subset of training images.

    If no normalization layer is provided, do nothing.
    """
    if norm_layer is None or not hasattr(norm_layer, "adapt"):
        return

    # Collect up to ~2048 samples from first few batches
    import numpy as np

    samples = []
    max_samples = 2048
    collected = 0
    for i in range(min(len(train_seq), 64)):
        batch = train_seq[i]
        X = batch[0] if isinstance(batch, (list, tuple)) else batch
        samples.append(X)
        collected += len(X)
        if collected >= max_samples:
            break
    if not samples:
        return
    data = np.concatenate(samples, axis=0)
    norm_layer.adapt(data)
