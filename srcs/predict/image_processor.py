from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from srcs.utils.common import get_logger

logger = get_logger(__name__)


class ImageProcessor:

    def __init__(self, img_size: int = 224):
        self.img_size = img_size

    def process_image(self, image_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        image_path = Path(image_path)
        logger.debug(f"Processing image: {image_path}")

        img = self._load_image(image_path)
        original_array = np.array(img)

        transformed_img = self._apply_transforms(img)
        processed_array = np.array(transformed_img)
        processed_array = self._normalize_for_model(processed_array)
        processed_array = np.expand_dims(processed_array, axis=0)

        return original_array, processed_array

    def _load_image(self, image_path: Path) -> Image.Image:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            raise RuntimeError(f"Error loading image: {e}")

    def _apply_transforms(self, img: Image.Image) -> Image.Image:
        logger.debug("Applying transformations to image")

        transformed = img.resize(
            (self.img_size, self.img_size), Image.Resampling.LANCZOS
        )

        logger.debug(f"Resized to {self.img_size}x{self.img_size}")
        return transformed

    def _normalize_for_model(self, img_array: np.ndarray) -> np.ndarray:
        normalized = img_array / 255.0

        logger.debug("Applied normalization (simulated)")
        return normalized
