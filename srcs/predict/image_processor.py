from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from srcs.utils.common import get_logger
from srcs.utils.image_utils import ImageLoader, ImageTransforms

logger = get_logger(__name__)


class ImageProcessor:

    def __init__(self, img_size: int = 224):
        self.img_size = img_size

    def process_image(self, image_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug(f"Processing image: {image_path}")

        img = ImageLoader.load_pil_image(image_path, ensure_rgb=True)
        original_array = np.array(img)

        transformed_img = ImageTransforms.resize_image(
            img, (self.img_size, self.img_size)
        )
        processed_array = np.array(transformed_img)
        processed_array = ImageTransforms.normalize_array(processed_array)
        processed_array = np.expand_dims(processed_array, axis=0)

        return original_array, processed_array
