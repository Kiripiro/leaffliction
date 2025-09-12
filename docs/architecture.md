# CNN Architecture (leaf_cnn)

This document describes the `leaf_cnn` network (defined in `srcs/model/cnn.py`) and the reasoning behind key choices.

## Overview

- Input: `(img_size, img_size, 3)` (default 224×224×3)
- Optional data-adapted normalization (`layers.Normalization`)
- 3–4 convolutional blocks (scale-dependent) with increasing channels
- Head: `GlobalAveragePooling2D → Dropout → Dense(num_classes, softmax)`

## Diagram

```mermaid
flowchart LR
    I["Input: (H,W,3)"] --> N{"Normalization?"}
    N -->|yes| A["Adapted Normalization"]
    N -->|no| A

    A --> B1["Block 1<br/>Conv3x3→BN→ReLU ×2<br/>MaxPool2x2 + Dropout"]
    B1 --> B2["Block 2<br/>..."]
    B2 --> B3["Block 3<br/>..."]
    B3 --> B4["Block 4 (base only)<br/>..."]
    B4 --> GAP["GlobalAveragePooling2D"]
    GAP --> D1["Dropout"]
    D1 --> FC["Dense num_classes + Softmax"]

    classDef node fill:#f7f7f7,stroke:#888,stroke-width:1px;
    class I,N,A,B1,B2,B3,B4,GAP,D1,FC node;
```

## Rationale

- Two 3×3 convolutions per block: larger effective receptive field with moderate cost.
- BatchNorm after conv: stable optimization, works well with mixed precision.
- GlobalAveragePooling: fewer parameters than Flatten+Dense; reduces overfitting.
- Dropout in blocks and head: light regularization.
- Optional depthwise-separable mode for lighter compute.

## Presets

- tiny: widths=[16,32,64], drop_block=0.10, drop_top=0.30
- small: widths=[32,64,128], drop_block=0.15, drop_top=0.35
- base: widths=[32,64,128,256], drop_block=0.15, drop_top=0.40

## Interactions

- Mixed precision is enabled by default; normalization helps numerical stability.
- EMA weights are evaluated post-training; only the best variant (base or EMA) is saved.

## Code

- Model and normalization: `srcs/model/cnn.py`
- Training and callbacks: `srcs/cli/train.py`, `srcs/train/utils.py`
- Data loading: `srcs/dataio/sequence.py`
