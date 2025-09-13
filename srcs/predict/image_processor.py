from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from srcs.utils.common import get_logger
from srcs.utils.image_utils import ImageLoader, ImageTransforms

logger = get_logger(__name__)


class ImageProcessor:

    def __init__(self, img_size: int = 224):
        self.img_size = img_size

    def process_image(
        self, image_path: str | Path, enable_subprocess: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.debug(f"Processing image: {image_path}")

        image_path = Path(image_path)
        img = ImageLoader.load_pil_image(image_path, ensure_rgb=True)
        original_array = np.array(img)

        transformed_array = self._get_transformed_array(
            image_path, original_array, enable_subprocess
        )

        transformed_img = ImageTransforms.resize_image(
            ImageLoader.array_to_pil(transformed_array),
            (self.img_size, self.img_size),
        )

        processed_array = np.array(transformed_img)
        processed_array = ImageTransforms.normalize_array(processed_array)
        processed_array = np.expand_dims(processed_array, axis=0)

        return original_array, processed_array, transformed_array

    def _get_transformed_array(
        self,
        image_path: Path,
        original_array: np.ndarray,
        enable_subprocess: bool = True,
    ) -> np.ndarray:
        if not enable_subprocess:
            return original_array

        mask_image_path = self._find_mask_image(image_path)
        logger.debug(f"Looking for mask image at: {mask_image_path}")

        if mask_image_path and mask_image_path.exists():
            return self._load_existing_mask(mask_image_path)

        return self._apply_transformation(original_array, enable_subprocess)

    def _load_existing_mask(self, mask_image_path: Path) -> np.ndarray:
        logger.info(f"Using existing transformed mask image: {mask_image_path}")
        mask_img = ImageLoader.load_pil_image(mask_image_path, ensure_rgb=True)
        return np.array(mask_img)

    def _apply_transformation(
        self, original_array: np.ndarray, enable_subprocess: bool = True
    ) -> np.ndarray:
        if not enable_subprocess:
            return original_array

        logger.info("No mask image found, applying transformation")
        try:
            return self._run_transformation_subprocess(original_array)
        except Exception as e:
            logger.warning(f"Failed to apply transformation: {e}")
            return original_array

    def _run_transformation_subprocess(self, original_array: np.ndarray) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_input:
            from PIL import Image

            Image.fromarray(original_array).save(temp_input.name)
            logger.info(f"Running transformation on: {temp_input.name}")

            result = subprocess.run(
                [
                    "python3",
                    "srcs/cli/Transformation.py",
                    temp_input.name,
                    "--types",
                    "Mask",
                    "--preview",
                ],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            transformed_array = self._process_transformation_result(
                result, original_array
            )
            Path(temp_input.name).unlink()
            return transformed_array

    def _process_transformation_result(
        self, result, original_array: np.ndarray
    ) -> np.ndarray:
        logger.info(f"Transformation result code: {result.returncode}")
        if result.stdout:
            logger.info(f"Transformation stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Transformation stderr: {result.stderr}")

        if result.returncode != 0:
            logger.warning("Transformation failed, using original")
            return original_array

        mask_output = self._extract_mask_path(result.stdout)
        if mask_output and mask_output.exists():
            return self._load_and_cleanup_mask(mask_output)

        logger.warning("Mask output not found in transformation output")
        return original_array

    def _extract_mask_path(self, stdout: str) -> Optional[Path]:
        for line in stdout.split("\n"):
            if "__T_Mask.jpg" in line and "- " in line:
                mask_path = line.split("- ")[-1].strip()
                return Path(mask_path)
        return None

    def _load_and_cleanup_mask(self, mask_output: Path) -> np.ndarray:
        from PIL import Image

        logger.info(f"Mask output found at: {mask_output}")
        transformed_array = np.array(Image.open(mask_output))
        try:
            mask_output.unlink()
            if mask_output.parent.exists():
                shutil.rmtree(mask_output.parent)
        except Exception:
            pass
        return transformed_array

    def _find_mask_image(self, image_path: Path) -> Path:
        stem = image_path.stem
        match = re.search(r"image \((\d+)\)", stem)
        if match:
            image_number = match.group(1)
            mask_path = (
                Path("artifacts")
                / "transformations"
                / image_number
                / f"{stem}__T_Mask.jpg"
            )
        else:
            mask_path = image_path.parent / f"{stem}__T_Mask.jpg"
        return mask_path

    def generate_mask_for_visualization(self, image_path: str | Path) -> np.ndarray:
        """Generate a mask/transformed array for display only.

        This does NOT affect the model input. Fallbacks to original if transform fails.
        """
        image_path = Path(image_path)
        img = ImageLoader.load_pil_image(image_path, ensure_rgb=True)
        original_array = np.array(img)
        try:
            return self._get_transformed_array(
                image_path, original_array, enable_subprocess=True
            )
        except Exception:
            return original_array
