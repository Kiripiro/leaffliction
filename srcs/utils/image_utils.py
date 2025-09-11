from __future__ import annotations

import subprocess
import sys
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


class DisplayUtils:

    @staticmethod
    def open_image_viewer(image_path: Union[str, Path]) -> None:
        """Open image with system default viewer."""
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Cannot display image, file not found: {image_path}")
            return

        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(image_path)], check=True)
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", str(image_path)], check=True)
            elif sys.platform == "win32":
                subprocess.run(["start", str(image_path)], shell=True, check=True)
            else:
                logger.warning(
                    f"Unsupported platform for image display: {sys.platform}"
                )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to open image viewer: {e}")
        except FileNotFoundError:
            logger.warning("No image viewer found")

    @staticmethod
    def create_confusion_matrix(
        results: List[dict], output_path: Union[str, Path]
    ) -> Path:
        """Create confusion matrix from batch prediction results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
        except ImportError:
            logger.error(
                "matplotlib, seaborn, or sklearn not available for confusion matrix"
            )
            return None

        output_path = Path(output_path)

        if not results:
            logger.warning("No results to create confusion matrix")
            return None

        y_true = []
        y_pred = []
        all_labels = set()

        for result in results:
            image_path = Path(result["image_path"])

            true_label = image_path.parent.name
            pred_label = result["top_prediction"]

            y_true.append(true_label)
            y_pred.append(pred_label)
            all_labels.add(true_label)
            all_labels.add(pred_label)

        labels = sorted(all_labels)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"shrink": 0.8},
        )

        plt.title("Confusion Matrix - Batch Prediction Results")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved to: {output_path}")
        return output_path
