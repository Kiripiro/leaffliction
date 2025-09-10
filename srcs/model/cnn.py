from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import keras
import numpy as np
from keras import layers, regularizers

LOGGER = logging.getLogger(__name__)


def build_leafcnn(
    num_classes: int,
    img_size: int,
    use_norm: bool,
    widths: List[int],
    drop_block: float,
    drop_top: float,
    l2_reg: float,
    separable: bool = False,
) -> Tuple[keras.Model, Optional[layers.Normalization]]:
    """Flexible CNN builder with scaling and separable option.

    Returns (model, normalization_layer_or_None).
    """
    kr = regularizers.l2(l2_reg) if l2_reg > 0 else None
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = inputs
    norm_layer: Optional[layers.Normalization] = None
    if use_norm:
        norm_layer = layers.Normalization(name="adapt_norm")
        x = norm_layer(x)

    Conv = layers.SeparableConv2D if separable else layers.Conv2D
    for bi, f in enumerate(widths, start=1):
        name = f"b{bi}"
        for ci in range(2):
            if separable:
                x = Conv(
                    f,
                    3,
                    padding="same",
                    use_bias=False,
                    depthwise_initializer="he_normal",
                    pointwise_initializer="he_normal",
                    depthwise_regularizer=kr,
                    pointwise_regularizer=kr,
                    name=f"{name}_sepconv{ci + 1}",
                )(x)
            else:
                x = Conv(
                    f,
                    3,
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_normal",
                    kernel_regularizer=kr,
                    name=f"{name}_conv{ci + 1}",
                )(x)
            x = layers.BatchNormalization(name=f"{name}_bn{ci + 1}")(x)
            x = layers.ReLU(name=f"{name}_relu{ci + 1}")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"{name}_pool")(x)
        x = layers.Dropout(drop_block, name=f"{name}_drop")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(drop_top, name="top_drop")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=kr,
        name="probs",
    )(x)
    model = keras.Model(inputs, outputs, name="leaf_cnn")
    return model, norm_layer


def build_model(
    num_classes: int, img_size: int, use_norm: bool, weight_decay: float
) -> Tuple[keras.Model, Optional[layers.Normalization]]:
    """Backward-compatible builder: base scale.

    Equivalent to widths=[32,64,128,256], drop_block=0.15,
    drop_top=0.4, l2_reg=weight_decay, separable=False.
    """
    return build_leafcnn(
        num_classes=num_classes,
        img_size=img_size,
        use_norm=use_norm,
        widths=[32, 64, 128, 256],
        drop_block=0.15,
        drop_top=0.4,
        l2_reg=weight_decay,
        separable=False,
    )


def adapt_normalization(norm_layer: Optional[layers.Normalization], train_seq) -> None:
    nl = norm_layer
    if nl is None:
        return
    imgs = []
    seen = 0
    for i in range(len(train_seq)):
        batch_x, _ = train_seq[i]
        imgs.append(batch_x)
        seen += len(batch_x)
        if seen >= 2048:
            break
    arr = np.concatenate(imgs, axis=0)
    nl.adapt(arr)
    LOGGER.info("Normalization layer adapted on %d samples", arr.shape[0])
