from __future__ import annotations

import logging
import math
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from keras.utils import Sequence, img_to_array, load_img

from srcs.dataio.manifest import ManifestItem


def setup_sequence_logging(
    log_file: Optional[str] = None, level: int = logging.INFO, also_console: bool = True
) -> logging.Logger:
    """
    Configure logging pour ManifestSequence avec fichier et/ou console.

    Args:
        log_file: Chemin du fichier de log (None = pas de fichier)
        level: Niveau de log (logging.DEBUG, INFO, etc.)
        also_console: Ajouter aussi la sortie console

    Returns:
        Logger configuré
    """
    logger = logging.getLogger("ManifestSequence")
    logger.setLevel(level)
    # Avoid duplicate logs via root logger
    logger.propagate = False

    # Supprimer les handlers existants
    logger.handlers.clear()

    # Format détaillé
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler fichier
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Handler console (optionnel)
    if also_console:
        # Send console logs to stderr so they don't get hidden by
        # Keras progbar on stdout
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class ManifestSequence(Sequence):
    """Keras Sequence for loading images from manifest entries.

    Supports optional in-memory caching and threaded I/O.

    New features:
    - Apply multiple transformations per image in-memory via `transform` function
    - Prevent duplicate application of the same filter per image (session-scoped)
    - Shared in-memory cache across calls for optimal performance
    - Optional logging using project logger
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
        # New options
        transformation: bool = True,
        transform_types: Optional[Tuple[str, ...]] = None,
        enforce_unique_filters: bool = True,
        logger: Optional[logging.Logger] = None,
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
        # Global switch to enable/disable transform pipeline
        self.transformation = bool(transformation)
        # New transformation control
        self.transform_types = transform_types
        self.enforce_unique_filters = enforce_unique_filters
        self.logger = logger or logging.getLogger(__name__)
        # Shared cache for transforms and decoded RGB/originals
        # Keyed as in Transformation.transform_single_image_for_training
        self._extern_cache: Dict[Any, Any] = {}
        # Track filters already applied per image src (canonical names)
        self._applied_filters: Dict[str, set[str]] = {}
        self._lock = threading.RLock()
        # Old cache of full tensors (Sequence-level cache)
        self._cache_imgs: List[np.ndarray] = []
        self._cache_labels: List[np.ndarray] = []
        self._cache_origs: List[np.ndarray] = []
        if self.shuffle:
            self.rng.shuffle(self.indexes)
        super().__init__(**kwargs)
        if self.cache:
            self._build_cache()

        # Log initial setup
        self.logger.info(
            "ManifestSequence initialized: %d items, %d batches, img_size=%d, "
            "batch_size=%d, workers=%d, transforms=%s, cache=%s",
            len(self.items),
            len(self),
            self.img_size,
            self.batch_size,
            self.workers,
            self.transform_types if self.transform_types else "default",
            self.cache,
        )

    def __len__(self) -> int:
        return math.ceil(len(self.items) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indexes)
            self.logger.debug("Epoch ended: shuffled %d indexes", len(self.indexes))

    def _canonical(self, name: str) -> str:
        try:
            from srcs.cli.Transformation import CANONICAL_TYPES
        except Exception:
            CANONICAL_TYPES = {}
        key = str(name).strip().lower()
        return CANONICAL_TYPES.get(key, name)

    def _compute_transforms_to_apply(self, src: Path) -> Optional[Tuple[str, ...]]:
        if not self.transform_types:
            return None
        if not self.enforce_unique_filters:
            return tuple(self.transform_types)
        # Enforce per-image uniqueness across calls
        with self._lock:
            already = self._applied_filters.get(str(src), set())
            to_apply: List[str] = []
            for t in self.transform_types:
                canon = self._canonical(t)
                if canon not in already:
                    to_apply.append(canon if canon else t)
            return tuple(to_apply)

    def _mark_applied(self, src: Path, applied: Iterable[str]) -> None:
        with self._lock:
            s = self._applied_filters.setdefault(str(src), set())
            for t in applied:
                s.add(self._canonical(t))

    def _load_one(self, i: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load and process a single item.

        Returns (orig_uint8, x_float32, y) where y may be None when label2idx is None.
        Applies multi-filter transform with cache and uniqueness enforcement.
        """
        import time

        item_start_time = time.time()
        it = self.items[i]

        if self.transformation and self.transform is not None:
            # Determine which transforms to apply for this call (for logging only)
            transforms_for_call = self._compute_transforms_to_apply(Path(it.src))
            if transforms_for_call is None:
                self.logger.debug(
                    "[%s] Using transform's default configuration (no override)",
                    it.id,
                )
            elif len(transforms_for_call) == 0:
                self.logger.debug(
                    "[%s] No new transforms to apply for %s; proceeding with transform",
                    it.id,
                    it.src,
                )
            else:
                self.logger.debug(
                    "[%s] Applying transforms %s to %s (delegated to transform)",
                    it.id,
                    ",".join(transforms_for_call),
                    it.src,
                )
            transform_start = time.time()
            orig_uint8, x_float32 = self.transform(
                Path(it.src),
                it,
                self.img_size,
            )
            transform_time = time.time() - transform_start
            self.logger.debug(
                "[%s] Transform completed in %.3fs", it.id, transform_time
            )
        else:
            self.logger.debug("[%s] Loading without transforms", it.id)
            load_start = time.time()
            resized = img_to_array(
                load_img(
                    it.src,
                    target_size=(self.img_size, self.img_size),
                    color_mode="rgb",
                )
            )
            x_float32 = (resized / 255.0).astype("float32")
            orig_uint8 = np.clip(resized, 0, 255).astype("uint8")
            load_time = time.time() - load_start
            self.logger.debug("[%s] Basic load completed in %.3fs", it.id, load_time)

        y: Optional[np.ndarray] = None
        if self.label2idx is not None:
            lab_idx = self.label2idx[it.label]
            if self.one_hot:
                la = np.zeros(self.num_classes, dtype="float32")
                la[lab_idx] = 1.0
                y = la
            else:
                y = np.asarray(lab_idx, dtype="int32")

        total_time = time.time() - item_start_time
        self.logger.debug(
            "[%s] Item loaded in %.3fs (label=%s)",
            it.id,
            total_time,
            it.label if self.label2idx else "none",
        )
        return orig_uint8, x_float32, y

    def _build_cache(self) -> None:
        self.logger.info("Building cache for %d items...", len(self.items))
        import time

        start_time = time.time()

        imgs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        origs: List[np.ndarray] = []
        for i in range(len(self.items)):
            if i % 100 == 0 and i > 0:
                self.logger.info(
                    "Cache progress: %d/%d items processed", i, len(self.items)
                )
            orig, arr, la = self._load_one(i)
            imgs.append(arr)
            origs.append(orig)
            if la is not None:
                labels.append(la)
        self._cache_imgs = imgs
        self._cache_origs = origs
        self._cache_labels = labels

        elapsed = time.time() - start_time
        self.logger.info(
            "Cache built successfully in %.2fs: %d images, %d labels, %d originals",
            elapsed,
            len(imgs),
            len(labels),
            len(origs),
        )

    def __getitem__(self, idx: int):
        import time

        batch_start_time = time.time()

        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.items))
        batch_idx = self.indexes[start:end]
        actual_batch_size = len(batch_idx)

        self.logger.debug(
            "Loading batch %d: items [%d:%d] -> %d images",
            idx,
            start,
            end,
            actual_batch_size,
        )

        if self.cache and self._cache_imgs:
            self.logger.debug("Using cached data for batch %d", idx)
            imgs = [self._cache_imgs[i] for i in batch_idx]
            if self.label2idx is None:
                batch_time = time.time() - batch_start_time
                self.logger.debug(
                    "Batch %d loaded from cache in %.3fs (%d items)",
                    idx,
                    batch_time,
                    actual_batch_size,
                )
                return np.asarray(imgs, dtype="float32")
            labels = [self._cache_labels[i] for i in batch_idx]
            batch_time = time.time() - batch_start_time
            self.logger.debug(
                "Batch %d loaded from cache in %.3fs (%d items)",
                idx,
                batch_time,
                actual_batch_size,
            )
            return np.asarray(imgs, dtype="float32"), np.asarray(labels)

        # Live loading
        imgs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        transforms_applied = 0

        if self.workers > 1:
            self.logger.debug("Loading batch %d with %d workers", idx, self.workers)
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                results = list(ex.map(self._load_one, batch_idx))
                for _orig, arr, lab in results:
                    imgs.append(arr)
                    if lab is not None:
                        labels.append(lab)
                    if self.transform is not None:
                        transforms_applied += 1
        else:
            for i in batch_idx:
                _orig, arr, lab = self._load_one(i)
                imgs.append(arr)
                if lab is not None:
                    labels.append(lab)
                if self.transform is not None:
                    transforms_applied += 1

        X = np.asarray(imgs, dtype="float32")
        batch_time = time.time() - batch_start_time

        if self.label2idx is None:
            self.logger.debug(
                "Batch %d loaded in %.3fs (%d items, %d transforms applied)",
                idx,
                batch_time,
                actual_batch_size,
                transforms_applied,
            )
            return X

        y = np.asarray(labels)
        self.logger.debug(
            "Batch %d loaded in %.3fs (%d items, %d transforms applied)",
            idx,
            batch_time,
            actual_batch_size,
            transforms_applied,
        )
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
