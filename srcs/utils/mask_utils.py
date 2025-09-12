#!/usr/bin/env python3
"""
Mask utilities following PlantCV apply_mask logic.
"""

import cv2
import numpy as np


def apply_mask(
    img: np.ndarray, mask: np.ndarray, mask_color: str = "white"
) -> np.ndarray:
    """Apply white or black image mask to image, following PlantCV logic.

    Apply binary mask to an image using bitwise operations. This function follows
    the same logic as PlantCV's apply_mask function.

    Args:
        img: RGB image data (numpy array)
        mask: Binary mask image data (numpy array)
        mask_color: 'white' or 'black' - color to apply where mask is 0

    Returns:
        masked_img: masked image data (numpy array)

    Raises:
        ValueError: If mask_color is not 'white' or 'black'

    Example:
        >>> import numpy as np
        >>> from srcs.utils.mask_utils import apply_mask
        >>>
        >>> # Create sample image and mask
        >>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
        >>>
        >>> # Apply white mask (background becomes white where mask is 0)
        >>> masked_white = apply_mask(img, mask, 'white')
        >>>
        >>> # Apply black mask (background becomes black where mask is 0)
        >>> masked_black = apply_mask(img, mask, 'black')
    """

    # Validate mask_color parameter
    if mask_color.upper() == "WHITE":
        color_val = 255
    elif mask_color.upper() == "BLACK":
        color_val = 0
    else:
        raise ValueError(f'Mask Color {mask_color} is not "white" or "black"!')

    # Validate input arrays
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy array")

    # Ensure mask is 2D
    if mask.ndim == 3:
        if mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask = mask[:, :, 0]
    elif mask.ndim != 2:
        raise ValueError("mask must be 2D or 3D array")

    # Ensure mask is binary (0 or 255)
    mask = (mask > 127).astype(np.uint8) * 255

    # Copy the input image to avoid modifying the original
    array_data = img.copy()

    # Apply the mask: set pixels to color_val where mask is 0
    if len(array_data.shape) == 3:  # Color image
        # For RGB images, set all channels to color_val where mask is 0
        array_data[mask == 0] = color_val
    elif len(array_data.shape) == 2:  # Grayscale image
        # For grayscale images, set pixels to color_val where mask is 0
        array_data[mask == 0] = color_val
    else:
        raise ValueError("img must be 2D (grayscale) or 3D (color) array")

    return array_data


def create_binary_mask(img: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Create a binary mask from grayscale image.

    Args:
        img: Input grayscale image
        threshold: Threshold value for binarization (0-255)

    Returns:
        Binary mask (0 or 255)
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_mask


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Invert a binary mask.

    Args:
        mask: Binary mask (0s and 255s)

    Returns:
        Inverted binary mask
    """
    return 255 - mask


def combine_masks(
    mask1: np.ndarray, mask2: np.ndarray, operation: str = "and"
) -> np.ndarray:
    """Combine two binary masks using logical operations.

    Args:
        mask1: First binary mask
        mask2: Second binary mask
        operation: 'and', 'or', 'xor', or 'subtract'

    Returns:
        Combined binary mask
    """
    if operation.lower() == "and":
        return cv2.bitwise_and(mask1, mask2)
    elif operation.lower() == "or":
        return cv2.bitwise_or(mask1, mask2)
    elif operation.lower() == "xor":
        return cv2.bitwise_xor(mask1, mask2)
    elif operation.lower() == "subtract":
        return cv2.subtract(mask1, mask2)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def mask_to_contours(mask: np.ndarray) -> list:
    """Extract contours from binary mask.

    Args:
        mask: Binary mask

    Returns:
        List of contours
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def apply_morphological_operations(
    mask: np.ndarray,
    operation: str = "opening",
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    """Apply morphological operations to clean up mask.

    Args:
        mask: Binary mask
        operation: 'opening', 'closing', 'erosion', 'dilation'
        kernel_size: Size of morphological kernel
        iterations: Number of iterations

    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if operation.lower() == "opening":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation.lower() == "closing":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation.lower() == "erosion":
        return cv2.erode(mask, kernel, iterations=iterations)
    elif operation.lower() == "dilation":
        return cv2.dilate(mask, kernel, iterations=iterations)
    else:
        raise ValueError(f"Unknown morphological operation: {operation}")
