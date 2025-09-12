import random

import numpy as np
from PIL import Image, ImageOps

from srcs.utils.common import get_logger
from srcs.utils.image_utils import ImageLoader

logger = get_logger(__name__)


class ImageAugmenter:
    NOISE_LEVEL = 5

    def __init__(self, seed=None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def flip(self, image_path, output_path):
        try:
            img = ImageLoader.load_pil_image(image_path)
            if random.choice([True, False]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            ImageLoader.save_pil_image(img, output_path)
            return True
        except Exception as e:
            logger.error(f"Failed to process {image_path} - {e}")
            return False

    def rotate(self, image_path, output_path):
        try:
            img = ImageLoader.load_pil_image(image_path)
            angle = random.uniform(-30, 30)
            img = img.rotate(angle, expand=True, fillcolor="white")
            ImageLoader.save_pil_image(img, output_path)
            return True
        except Exception as e:
            logger.error(f"Failed to process {image_path} - {e}")
            return False

    def skew(self, image_path, output_path):
        try:
            img = ImageLoader.load_pil_image(image_path)
            width, height = img.size
            skew_factor = random.uniform(0.05, 0.15)

            coeffs = [
                1 + skew_factor,
                0,
                -skew_factor * width,
                0,
                1 + skew_factor,
                -skew_factor * height,
                0,
                0,
            ]

            img = img.transform(
                (width, height),
                Image.PERSPECTIVE,
                coeffs,
                Image.BICUBIC,
            )
            ImageLoader.save_pil_image(img, output_path)
            return True
        except Exception as e:
            logger.error(f"Failed to process {image_path} - {e}")
            return False

    def shear(self, image_path, output_path):
        try:
            img = ImageLoader.load_pil_image(image_path)
            width, height = img.size
            shear_factor = random.uniform(-0.2, 0.2)

            if random.choice([True, False]):
                coeffs = [1, shear_factor, 0, 0, 1, 0]
            else:
                coeffs = [1, 0, 0, shear_factor, 1, 0]

            img = img.transform(
                (width, height),
                Image.AFFINE,
                coeffs,
                Image.BICUBIC,
            )
            ImageLoader.save_pil_image(img, output_path)
            return True
        except Exception as e:
            logger.error(f"Failed to process {image_path} - {e}")
            return False

    def crop(self, image_path, output_path):
        try:
            img = ImageLoader.load_pil_image(image_path)
            width, height = img.size

            crop_ratio = random.uniform(0.8, 0.95)
            new_width = int(width * crop_ratio)
            new_height = int(height * crop_ratio)

            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)

            img = img.crop((left, top, left + new_width, top + new_height))
            img = img.resize((width, height), Image.LANCZOS)
            ImageLoader.save_pil_image(img, output_path)
            return True
        except Exception as e:
            logger.error(f"Failed to process {image_path} - {e}")
            return False

    def distortion(self, image_path, output_path):
        try:
            img = ImageLoader.load_pil_image(image_path)
            img_array = np.array(img)

            noise = np.random.normal(0, self.NOISE_LEVEL, img_array.shape).astype(
                np.uint8
            )
            img_array = np.clip(img_array + noise, 0, 255)

            img = ImageLoader.array_to_pil(img_array)
            img = ImageOps.autocontrast(img, cutoff=random.uniform(0, 2))

            ImageLoader.save_pil_image(img, output_path)
            return True
        except Exception as e:
            logger.error(f"Failed to process {image_path} - {e}")
            return False
