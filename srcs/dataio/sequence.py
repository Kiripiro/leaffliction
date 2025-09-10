from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from keras.utils import Sequence, img_to_array, load_img

from srcs.dataio.manifest import ManifestItem


class ManifestSequence(Sequence):
    """Keras Sequence for loading images from manifest entries.

    Supports optional in-memory caching and threaded I/O.
    """

    def __init__(
        self,
        items: List[ManifestItem],
        label2idx: Optional[Dict[str, int]],
        img_size: int,
        batch_size: int,
        shuffle: bool,
        seed: int,
        limit: int | None = None,
        num_classes: int | None = None,
        one_hot: bool = False,
        cache: bool = False,
        workers: int = 1,
        transform: Optional[
            Callable[[Path, ManifestItem, int], Tuple[np.ndarray, np.ndarray]]
        ] = None,
        **kwargs,
    ) -> None:
        if limit is not None:
            items = items[:limit]
        self.items = items
        self.label2idx = label2idx
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.indexes = list(range(len(items)))
        # Labels are optional (prediction mode)
        self.one_hot = one_hot and (label2idx is not None)
        if self.one_hot and num_classes is None:
            raise ValueError("num_classes must be provided when one_hot=True")
        self.num_classes = int(num_classes or 0)
        self.cache = cache
        # Avoid double-parallelism by default: keep internal workers at 1
        self.workers = max(1, int(workers))
        self.transform = transform
        self._cache_imgs: List[np.ndarray] = []
        self._cache_labels: List[np.ndarray] = []
        self._cache_origs: List[np.ndarray] = []
        if self.shuffle:
            self.rng.shuffle(self.indexes)
        super().__init__(**kwargs)
        if self.cache:
            self._build_cache()

    def __len__(self) -> int:
        return math.ceil(len(self.items) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def _load_one(self, i: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load and process a single item.

        Returns (orig_uint8, x_float32, y) where y may be None when label2idx is None.
        """
        it = self.items[i]
        # Read base image as uint8 (resized copy used when no external transform)
        if self.transform is not None:
            # Delegate to provided transform (handles reading + deterministic ops)
            orig_uint8, x_float32 = self.transform(Path(it.src), it, self.img_size)
        else:
            # Fallback: resize + normalize via Keras utils
            resized = img_to_array(
                load_img(
                    it.src,
                    target_size=(self.img_size, self.img_size),
                    color_mode="rgb",
                )
            )
            x_float32 = (resized / 255.0).astype("float32")
            orig_uint8 = np.clip(resized, 0, 255).astype("uint8")

        y: Optional[np.ndarray] = None
        if self.label2idx is not None:
            lab_idx = self.label2idx[it.label]
            if self.one_hot:
                la = np.zeros(self.num_classes, dtype="float32")
                la[lab_idx] = 1.0
                y = la
            else:
                y = np.asarray(lab_idx, dtype="int32")
        return orig_uint8, x_float32, y

    def _build_cache(self) -> None:
        imgs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        origs: List[np.ndarray] = []
        for i in range(len(self.items)):
            orig, arr, la = self._load_one(i)
            imgs.append(arr)
            origs.append(orig)
            if la is not None:
                labels.append(la)
        self._cache_imgs = imgs
        self._cache_origs = origs
        self._cache_labels = labels

    def __getitem__(self, idx: int):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.items))
        batch_idx = self.indexes[start:end]
        if self.cache and self._cache_imgs:
            imgs = [self._cache_imgs[i] for i in batch_idx]
            if self.label2idx is None:
                return np.asarray(imgs, dtype="float32")
            labels = [self._cache_labels[i] for i in batch_idx]
            return np.asarray(imgs, dtype="float32"), np.asarray(labels)

        imgs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        if self.workers > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                for _orig, arr, lab in ex.map(self._load_one, batch_idx):
                    imgs.append(arr)
                    if lab is not None:
                        labels.append(lab)
        else:
            for i in batch_idx:
                _orig, arr, lab = self._load_one(i)
                imgs.append(arr)
                if lab is not None:
                    labels.append(lab)
        X = np.asarray(imgs, dtype="float32")
        if self.label2idx is None:
            return X
        y = np.asarray(labels)
        return X, y

    def iter_with_info(
        self, batch_size: Optional[int] = None
    ) -> Iterable[
        Tuple[np.ndarray, Optional[np.ndarray], List[np.ndarray], List[ManifestItem]]
    ]:
        """Yield batches with originals and item metadata for visualization.

        Yields tuples: (X_float32, y_or_None, originals_uint8_list, items_list)
        """
        bs = batch_size or self.batch_size
        total = len(self.items)
        for start in range(0, total, bs):
            end = min(start + bs, total)
            idxs = self.indexes[start:end]
            Xs: List[np.ndarray] = []
            Ys: List[np.ndarray] = []
            Origs: List[np.ndarray] = []
            its: List[ManifestItem] = []
            for i in idxs:
                orig, arr, y = self._load_one(i)
                Xs.append(arr)
                Origs.append(orig)
                its.append(self.items[i])
                if y is not None:
                    Ys.append(y)
            Xb = np.asarray(Xs, dtype="float32")
            yb: Optional[np.ndarray] = None
            if self.label2idx is not None:
                yb = np.asarray(Ys)
            yield Xb, yb, Origs, its
