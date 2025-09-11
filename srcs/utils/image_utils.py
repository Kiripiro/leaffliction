from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL import Image

from srcs.utils.common import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class ImageLoader:

    @staticmethod
    def load_pil_image(
        image_path: Union[str, Path], ensure_rgb: bool = True
    ) -> Image.Image:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        try:
            img = Image.open(image_path)

            if ensure_rgb and img.mode != "RGB":
                img = img.convert("RGB")

            return img

        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

    @staticmethod
    def load_as_array(
        image_path: Union[str, Path], ensure_rgb: bool = True
    ) -> np.ndarray:
        img = ImageLoader.load_pil_image(image_path, ensure_rgb)
        return np.array(img)

    @staticmethod
    def save_pil_image(
        img: Image.Image, output_path: Union[str, Path], quality: int = 95
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img.save(output_path, quality=quality)
            logger.debug(f"Saved image: {output_path}")
        except Exception as e:
            raise RuntimeError(f"Error saving image {output_path}: {e}")

    @staticmethod
    def array_to_pil(array: np.ndarray) -> Image.Image:
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)

        return Image.fromarray(array)

    @staticmethod
    def get_image_files(directory: Union[str, Path]) -> List[Path]:
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        image_files = []
        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
            image_files.extend(directory.glob(f"**/*{ext}"))
            image_files.extend(directory.glob(f"**/*{ext.upper()}"))

        return sorted(image_files)

    @staticmethod
    def validate_image_path(image_path: Union[str, Path]) -> Path:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Path not found: {image_path}")

        if not image_path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        return image_path


class ImageTransforms:

    @staticmethod
    def resize_image(
        img: Image.Image,
        size: Tuple[int, int],
        method: Image.Resampling = Image.Resampling.LANCZOS,
    ) -> Image.Image:
        return img.resize(size, method)

    @staticmethod
    def normalize_array(
        array: np.ndarray, target_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        min_val, max_val = target_range

        if array.dtype == np.uint8:
            normalized = array.astype(np.float32) / 255.0
        else:
            normalized = array.astype(np.float32)

        if (min_val, max_val) != (0.0, 1.0):
            normalized = normalized * (max_val - min_val) + min_val

        return normalized
