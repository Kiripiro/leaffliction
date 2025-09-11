from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np


def _default_config():
    # Build a default TransformConfig compatible with Transformation.TransformPipeline
    from srcs.cli.Transformation import TransformConfig

    return TransformConfig(
        gaussian_sigma=1.6,
        hsv_channel_for_mask="s",
        fill_size=7,
        morph_kernel=5,
        landmarks_count=32,
        roi_size=(224, 224),
        mask_strategy="auto",
        bg_bias=None,
        grabcut_refine=False,
        green_hue_range=(35, 85),
        min_object_area_ratio=0.02,
        max_object_area_ratio=0.95,
        mask_upscale_factor=1.0,
        mask_upscale_long_side=0,
        shadow_suppression=True,
        shadow_s_max=25,
        shadow_v_method="percentile",
        shadow_v_percentile=10,
        brown_hue_range=(5, 35),
        brown_s_min=40,
        brown_v_max=120,
        brown_min_area_px=100,
        brown_morph_kernel=3,
        use_lab_brown=False,
        lab_b_min=135,
        lab_a_min=140,
    )


def _pil_read_rgb(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        return np.array(im)


class PipelineRunner:
    """Helper to run all pipeline steps once per image (optionally iterated)."""

    def __init__(self, config: Any, steps: tuple[str, ...], iters: int) -> None:
        from srcs.cli.Transformation import TransformPipeline

        self.config = config
        self.steps = steps
        self.iters = iters
        self.pipe = TransformPipeline(config)

    @staticmethod
    def _to_image(x: Any) -> np.ndarray:
        arr = x[0] if isinstance(x, tuple) else x
        return np.asarray(arr)

    def _first_pass(self, base: np.ndarray) -> tuple[np.ndarray, Any, Any]:
        out = base
        mask_img = None
        contour = None
        if (
            "Mask" in self.steps
            or "ROI" in self.steps
            or "Analyze" in self.steps
            or "Landmarks" in self.steps
            or "Brown" in self.steps
        ):
            mask_img, contour = self.pipe.make_mask(out)
            if "Mask" in self.steps and mask_img is not None:
                masked = out.copy()
                masked[mask_img == 0] = 0
                out = masked
        return out, mask_img, contour

    def _iterate(self, out: np.ndarray, mask_img: Any, contour: Any) -> np.ndarray:
        if "Blur" in self.steps:
            out = self._to_image(self.pipe.blur(out))
        if "ROI" in self.steps and contour is not None:
            prev_hw = out.shape[:2]
            out = self._to_image(self.pipe.roi(out, contour))
            # If ROI changes size, recompute mask/contour to keep shapes aligned
            if out.shape[:2] != prev_hw:
                mask_img, contour = self.pipe.make_mask(out)
        if "Analyze" in self.steps and mask_img is not None and contour is not None:
            out = self._to_image(self.pipe.analyze(out, mask_img, contour))
        if "Landmarks" in self.steps and contour is not None:
            out = self._to_image(self.pipe.pseudolandmarks(out, contour))
        if "Brown" in self.steps and mask_img is not None:
            out = self._to_image(self.pipe.detect_brown_spots(out, mask_img))
        if "Hist" in self.steps:
            out = self._to_image(self.pipe.histogram_hsv(out))
        return out

    def apply_all(self, base: np.ndarray) -> np.ndarray:
        out, mask_img, contour = self._first_pass(base)
        for _ in range(self.iters):
            out = self._iterate(out, mask_img, contour)
        return out


def build_pipeline_transform(
    types: Optional[Sequence[str]] = None,
    iterations: int = 1,
    cfg: Optional[object] = None,
) -> Callable[[Path, object, int], Tuple[np.ndarray, np.ndarray]]:
    """Return a transform callable compatible with ManifestSequence.

    The callable loads the image, runs the Transformation pipeline steps, then
    returns (orig_uint8_resized, transformed_float32_resized_normed).

    Args:
    types: Transform step names (e.g., Blur/Mask/ROI). If None, use DEFAULT_TYPES.
        iterations: Number of times to iterate through the steps (>=1).
        cfg: Optional TransformConfig; defaults will be used if None.
    """

    from srcs.cli.Transformation import DEFAULT_TYPES

    steps = tuple(types) if types else DEFAULT_TYPES
    iters = max(1, int(iterations))
    config: Any = cfg or _default_config()

    def _pil_resize(arr: np.ndarray, size: int) -> np.ndarray:
        from PIL import Image

        img = Image.fromarray(arr)
        # Pillow >= 10 uses Image.Resampling.*
        try:  # pragma: no cover - runtime compatibility
            resample = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
        except Exception:  # Pillow < 10
            resample = Image.BILINEAR  # type: ignore[attr-defined]
        return np.asarray(img.resize((size, size), resample=resample))

    def _transform(
        path: Path, _item: Any, img_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        rgb = _pil_read_rgb(path)
        runner = PipelineRunner(config, steps, iters)
        transformed = runner.apply_all(rgb)
        # Resize both original and transformed to the requested size (PIL)
        orig_resized = _pil_resize(rgb, img_size)
        out_resized = _pil_resize(transformed, img_size)
        x_float32 = (out_resized / 255.0).astype("float32")
        orig_uint8 = np.clip(orig_resized, 0, 255).astype("uint8")
        return orig_uint8, x_float32

    return _transform
