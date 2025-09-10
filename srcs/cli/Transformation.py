#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Try to import logging helper early (imports must stay at top for E402)
try:
    from srcs.utils.common import setup_logging
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from srcs.utils.common import setup_logging  # type: ignore

# Limit thread oversubscription before importing numpy/opencv
# Heavy libs are imported lazily later to keep imports at top (E402)
for _k in (
    "OPENCV_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_TYPES = (
    "Blur",
    "Mask",
    "ROI",
    "Analyze",
    "Landmarks",
    "Hist",
    "Brown",
)
# Case-insensitive alias map to canonical transform names
CANONICAL_TYPES: Dict[str, str] = {
    "blur": "Blur",
    "mask": "Mask",
    "roi": "ROI",
    "analyze": "Analyze",
    "analyse": "Analyze",
    "landmarks": "Landmarks",
    "pseudolandmarks": "Landmarks",
    "pseudo-landmarks": "Landmarks",
    "hist": "Hist",
    "histogram": "Hist",
    "brown": "Brown",
    "disease": "Brown",
    "spots": "Brown",
}


@dataclass(frozen=True)
class TransformConfig:
    gaussian_sigma: float = 1.2
    hsv_channel_for_mask: str = "s"  # 'h' | 's' | 'v'
    fill_size: int = 50
    morph_kernel: int = 5  # odd
    landmarks_count: int = 64
    roi_size: Tuple[int, int] = (224, 224)  # (h, w)
    # New: robust mask options
    mask_strategy: str = (
        "auto"  # 'auto' | 'hsv_s' | 'hsv_v_dark' | 'hsv_h' | 'lab' | 'kmeans'
    )
    bg_bias: Optional[str] = None  # None|'light_bg'|'dark_bg'
    grabcut_refine: bool = True
    green_hue_range: Tuple[int, int] = (
        30,
        90,
    )  # inclusive H range for green-ish leaves
    min_object_area_ratio: float = 0.01
    max_object_area_ratio: float = 0.95
    # Optional pre-upscale for mask detection
    mask_upscale_factor: float = 1.0
    mask_upscale_long_side: int = 0  # if >0 and image long side < this, upscale to this
    # Shadow handling options
    shadow_suppression: bool = True
    shadow_s_max: int = (
        60  # pixels with saturation <= this and low V are considered shadow
    )
    shadow_v_method: str = "otsu"  # 'otsu' | 'percentile'
    shadow_v_percentile: int = 30  # used when method=percentile
    # Brown/Disease Detection
    brown_hue_range: Tuple[int, int] = (5, 25)
    brown_s_min: int = 40
    brown_v_max: int = 180
    brown_min_area_px: int = 100
    brown_morph_kernel: int = 3
    use_lab_brown: bool = False
    lab_b_min: int = 135
    lab_a_min: int = 125


@dataclass(frozen=True)
class ProcessArgs:
    img_path: Path
    out_dir: Path
    types: Tuple[str, ...]
    cfg: TransformConfig
    skip_existing: bool = False
    overwrite: bool = False


def load_config(path: Optional[Path]) -> TransformConfig:
    if not path:
        return TransformConfig()
    if not path.exists():
        logging.warning("Config not found, using defaults: %s", path)
        return TransformConfig()
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return TransformConfig(
            gaussian_sigma=float(
                data.get("gaussian_sigma", TransformConfig.gaussian_sigma)
            ),
            hsv_channel_for_mask=str(
                data.get("hsv_channel_for_mask", TransformConfig.hsv_channel_for_mask)
            ),
            fill_size=int(data.get("fill_size", TransformConfig.fill_size)),
            morph_kernel=int(data.get("morph_kernel", TransformConfig.morph_kernel)),
            landmarks_count=int(
                data.get("landmarks_count", TransformConfig.landmarks_count)
            ),
            roi_size=tuple(
                data.get("roi_size", TransformConfig.roi_size)
            ),  # type: ignore
            mask_strategy=str(data.get("mask_strategy", TransformConfig.mask_strategy)),
            bg_bias=data.get("bg_bias", TransformConfig.bg_bias),
            grabcut_refine=bool(
                data.get("grabcut_refine", TransformConfig.grabcut_refine)
            ),
            green_hue_range=tuple(
                data.get("green_hue_range", TransformConfig.green_hue_range)
            ),  # type: ignore
            min_object_area_ratio=float(
                data.get("min_object_area_ratio", TransformConfig.min_object_area_ratio)
            ),
            max_object_area_ratio=float(
                data.get("max_object_area_ratio", TransformConfig.max_object_area_ratio)
            ),
            mask_upscale_factor=float(
                data.get("mask_upscale_factor", TransformConfig.mask_upscale_factor)
            ),
            mask_upscale_long_side=int(
                data.get(
                    "mask_upscale_long_side",
                    TransformConfig.mask_upscale_long_side,
                )
            ),
            shadow_suppression=bool(
                data.get("shadow_suppression", TransformConfig.shadow_suppression)
            ),
            shadow_s_max=int(data.get("shadow_s_max", TransformConfig.shadow_s_max)),
            shadow_v_method=str(
                data.get("shadow_v_method", TransformConfig.shadow_v_method)
            ),
            shadow_v_percentile=int(
                data.get("shadow_v_percentile", TransformConfig.shadow_v_percentile)
            ),
            brown_hue_range=tuple(
                data.get("brown_hue_range", TransformConfig.brown_hue_range)
            ),
            brown_s_min=int(data.get("brown_s_min", TransformConfig.brown_s_min)),
            brown_v_max=int(data.get("brown_v_max", TransformConfig.brown_v_max)),
            brown_min_area_px=int(
                data.get("brown_min_area_px", TransformConfig.brown_min_area_px)
            ),
            brown_morph_kernel=int(
                data.get("brown_morph_kernel", TransformConfig.brown_morph_kernel)
            ),
            use_lab_brown=bool(
                data.get("use_lab_brown", TransformConfig.use_lab_brown)
            ),
            lab_b_min=int(data.get("lab_b_min", TransformConfig.lab_b_min)),
            lab_a_min=int(data.get("lab_a_min", TransformConfig.lab_a_min)),
        )
    except Exception as exc:
        logging.warning("Failed to read config (%s), using defaults", exc)
        return TransformConfig()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def imwrite_bgr(path: Path, rgb_img) -> None:
    import cv2
    import numpy as np

    if rgb_img is None:
        return
    arr = np.asarray(rgb_img)
    if arr.ndim == 2:
        bgr = arr
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), bgr)


def pil_read_rgb(path: Path):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        return np.array(im)


def draw_text(img, text: str, org=(10, 24)):
    import cv2

    if img is None:
        return None
    out = img.copy()
    cv2.putText(
        out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA
    )
    return out


class TransformPipeline:
    def __init__(self, cfg: TransformConfig) -> None:
        self.cfg = cfg
        # PlantCV global params
        from plantcv import plantcv as pcv  # type: ignore

        pcv.params.debug = None

    # --- core steps ---
    def blur(self, rgb):
        """Create a saliency map showing important regions in white/gray
        and less important in black"""
        import cv2
        import numpy as np

        # Get leaf mask
        mask, _ = self.make_mask(rgb)
        if mask is None:
            return rgb

        leaf_mask = (mask > 0) if mask.ndim == 2 else (mask[..., 0] > 0)

        # Create saliency map
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        saliency = np.zeros_like(gray, dtype=np.float32)

        # 1. Edge saliency (veins, leaf borders)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150, L2gradient=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        saliency += edges_dilated.astype(np.float32) * 0.4

        # 2. Texture/gradient saliency
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = cv2.magnitude(grad_x, grad_y)
        gradient_norm = cv2.normalize(
            gradient_mag, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        saliency += gradient_norm.astype(np.float32) * 0.3

        # 3. Brown/disease region saliency
        if hasattr(self.cfg, "brown_hue_range"):
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            lo, hi = self.cfg.brown_hue_range
            brown_regions = (
                (h >= lo)
                & (h <= hi)
                & (s >= self.cfg.brown_s_min)
                & (v <= self.cfg.brown_v_max)
                & leaf_mask
            )

            # Clean and dilate brown regions for better visibility
            brown_clean = cv2.morphologyEx(
                brown_regions.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel
            )
            brown_dilated = cv2.dilate(brown_clean, kernel, iterations=2)
            saliency += (
                brown_dilated.astype(np.float32) * 0.6
            )  # Higher weight for disease

        # 4. Color variation saliency (areas with unusual colors)
        blurred_rgb = cv2.GaussianBlur(rgb, (15, 15), 0)
        color_diff = np.mean(
            np.abs(rgb.astype(np.float32) - blurred_rgb.astype(np.float32)), axis=2
        )
        color_diff_norm = cv2.normalize(color_diff, None, 0, 255, cv2.NORM_MINMAX)
        saliency += color_diff_norm * 0.2

        # Normalize saliency map
        saliency_norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        # Apply Gaussian blur for smooth transitions
        saliency_blurred = cv2.GaussianBlur(
            saliency_norm, (5, 5), self.cfg.gaussian_sigma
        )

        # Mask to leaf area only
        result = np.zeros_like(gray)
        result[leaf_mask] = saliency_blurred[leaf_mask]

        # Convert to 3-channel for consistency with other outputs
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        return result_rgb

    def make_mask(self, rgb):  # noqa: C901
        import cv2
        import numpy as np
        from plantcv import plantcv as pcv  # type: ignore

        # Optional pre-upscale to improve edge quality on small images
        oh, ow = rgb.shape[:2]
        s = 1.0
        if self.cfg.mask_upscale_factor and self.cfg.mask_upscale_factor > 1.0:
            s = float(self.cfg.mask_upscale_factor)
        elif self.cfg.mask_upscale_long_side and self.cfg.mask_upscale_long_side > 0:
            ls = max(oh, ow)
            if ls < self.cfg.mask_upscale_long_side:
                s = float(self.cfg.mask_upscale_long_side) / float(ls)
        rgb_work = (
            rgb
            if abs(s - 1.0) < 1e-6
            else cv2.resize(
                rgb,
                (int(round(ow * s)), int(round(oh * s))),
                interpolation=cv2.INTER_CUBIC,
            )
        )

        def postprocess(bin_img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            # Ensure binary uint8
            b = (bin_img > 0).astype(np.uint8) * 255
            filled = pcv.fill(bin_img=b, size=self.cfg.fill_size)
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.cfg.morph_kernel, self.cfg.morph_kernel)
            )
            closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k)
            cnt, _ = self._largest_contour(opened)
            if cnt is None:
                return opened, None
            mask = self._contour_to_mask(opened.shape[:2], cnt)
            return mask, cnt

        def mask_hsv_s(object_type: str = "light"):
            g = pcv.rgb2gray_hsv(rgb_img=rgb_work, channel="s")
            th = pcv.threshold.otsu(gray_img=g, object_type=object_type)
            return th

        def mask_hsv_v(object_type: str = "dark"):
            g = pcv.rgb2gray_hsv(rgb_img=rgb_work, channel="v")
            th = pcv.threshold.otsu(gray_img=g, object_type=object_type)
            return th

        def mask_hsv_green():
            hsv = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            lo, hi = self.cfg.green_hue_range
            m = ((h >= lo) & (h <= hi) & (s >= 40)).astype(np.uint8) * 255
            return m

        def mask_lab_green():
            lab = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2LAB)
            L, a, b = cv2.split(lab)
            m = ((a <= 135) & (b >= 115) & (b <= 170)).astype(np.uint8) * 255
            return m

        def mask_kmeans(k: int = 3):
            cv2.setRNGSeed(12345)
            h, w = rgb_work.shape[:2]
            scale = 256 / max(h, w)
            small = cv2.resize(
                rgb_work,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            Z = small.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
            _K, labels, centers = cv2.kmeans(
                Z, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS
            )
            centers = centers.astype(np.uint8)
            lbl = labels.reshape(small.shape[:2])
            hsv_c = cv2.cvtColor(centers.reshape((1, k, 3)), cv2.COLOR_RGB2HSV)[0]
            lo, hi = self.cfg.green_hue_range
            green_score = np.array(
                [(1 if (lo <= hv[0] <= hi and hv[1] >= 40) else 0) for hv in hsv_c]
            )
            if self.cfg.bg_bias == "dark_bg":
                pick = int(np.argmax(centers.mean(axis=1)))
            elif self.cfg.bg_bias == "light_bg":
                pick = int(np.argmin(centers.mean(axis=1)))
            elif green_score.any():
                pick = int(np.argmax(green_score))
            else:
                sat = hsv_c[:, 1]
                pick = int(np.argmax(sat))
            ms = (lbl == pick).astype(np.uint8) * 255
            m = cv2.resize(ms, (w, h), interpolation=cv2.INTER_NEAREST)
            return m

        def score_mask(mask_bin: np.ndarray, cnt: Optional[np.ndarray]) -> float:
            if cnt is None:
                return -1.0
            h, w = mask_bin.shape[:2]
            area = float(cv2.contourArea(cnt))
            if area <= 1:
                return -1.0
            area_ratio = area / float(h * w)
            if (
                area_ratio < self.cfg.min_object_area_ratio
                or area_ratio > self.cfg.max_object_area_ratio
            ):
                return 0.01
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = (area / hull_area) if hull_area > 1 else 0.0
            gray = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mag = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dil = cv2.dilate(mask_bin, k, iterations=1)
            ero = cv2.erode(mask_bin, k, iterations=1)
            boundary = (dil > 0) ^ (ero > 0)
            b_strength = float(mag[boundary].mean()) if boundary.sum() > 0 else 0.0
            # Green consistency term
            hsv = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2HSV)
            Hc, Sc, _Vc = cv2.split(hsv)
            lo, hi = self.cfg.green_hue_range
            green = (Hc >= lo) & (Hc <= hi) & (Sc >= 40)
            denom = max(1, int(mask_bin.sum() // 255))
            green_frac = float((green & (mask_bin > 0)).sum()) / float(denom)
            # Border-touch penalty
            x, y, ww, hh = cv2.boundingRect(cnt)
            touches_border = (
                (x <= 0) or (y <= 0) or (x + ww >= w - 1) or (y + hh >= h - 1)
            )
            # Weighted score
            target = 0.35
            area_term = max(0.0, 1.0 - abs(area_ratio - target) / target)
            score = (
                0.35 * area_term
                + 0.25 * solidity
                + 0.25 * b_strength
                + 0.15 * green_frac
            )
            if touches_border:
                score *= 0.75
            return float(score)

        def suppress_shadow(
            mask_bin: np.ndarray,
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            # ULTRA-AGGRESSIVE shadow suppression - remove ALL dark regions
            hsv = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2HSV)
            Hc, Sc, Vc = cv2.split(hsv)

            # Convert to LAB for better shadow detection
            lab = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2LAB)
            L_ch, a_ch, b_ch = cv2.split(lab)

            # EXTREMELY AGGRESSIVE shadow detection thresholds

            # Method 1: Very aggressive dark region detection (LAB L < 40th percentile)
            l_threshold = np.percentile(L_ch, 40)  # Much more aggressive
            very_dark_lab = L_ch < l_threshold

            # Method 2: High saturation threshold + higher value threshold
            low_sat_dark = (Sc < 50) & (Vc < 100)  # Much higher thresholds

            # Method 3: Even more aggressive LAB + HSV combination
            aggressive_shadow = (
                (L_ch < np.percentile(L_ch, 45)) & (Sc < 60) & (Vc < 120)
            )

            # Method 4: Detect ALL low-brightness regions
            very_low_brightness = Vc < 90  # Remove anything darker than 90/255

            # Method 5: LAB lightness aggressive threshold
            lab_dark = L_ch < np.percentile(L_ch, 50)  # Remove bottom 50% of brightness

            # Method 6: Uniform area detection - more aggressive shadow detection
            gray = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            texture_diff = cv2.absdiff(gray, blur)
            uniform_areas = texture_diff < 15  # Higher threshold for uniformity
            shadow_uniform = uniform_areas & (Vc < 100)

            # Method 7: K-means with more clusters for better shadow separation
            h_img, w_img = rgb_work.shape[:2]

            # Downsample for K-means efficiency
            scale = min(1.0, 150.0 / max(h_img, w_img))
            small_h, small_w = int(h_img * scale), int(w_img * scale)
            rgb_small = cv2.resize(
                rgb_work, (small_w, small_h), interpolation=cv2.INTER_AREA
            )

            data = rgb_small.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

            try:
                # Use 5 clusters instead of 4 for better separation
                _, labels, centers = cv2.kmeans(
                    data, 5, None, criteria, 10, cv2.KMEANS_PP_CENTERS
                )

                # Find the TWO darkest clusters
                brightness_scores = centers.mean(axis=1)
                sorted_indices = np.argsort(brightness_scores)
                dark_clusters = sorted_indices[:2]  # Two darkest clusters

                shadow_kmeans_small = np.isin(labels.flatten(), dark_clusters).reshape(
                    (small_h, small_w)
                )

                # Resize back to original size
                shadow_kmeans = cv2.resize(
                    shadow_kmeans_small.astype(np.uint8),
                    (w_img, h_img),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            except Exception:
                # Fallback if K-means fails
                shadow_kmeans = np.zeros((h_img, w_img), dtype=bool)

            # More restrictive green preservation - only preserve clearly green regions
            lo, hi = self.cfg.green_hue_range
            green_regions = (
                (Hc >= lo) & (Hc <= hi) & (Sc >= 40) & (Vc >= 60)
            )  # Must be bright enough

            # Combine ALL shadow detection methods for MAXIMUM removal
            comprehensive_shadow = (
                very_dark_lab
                | low_sat_dark
                | aggressive_shadow
                | very_low_brightness
                | lab_dark
                | shadow_uniform
                | shadow_kmeans
            ) & (
                ~green_regions
            )  # Only preserve clearly bright green regions

            # More aggressive morphological cleaning
            shadow_kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            shadow_kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # Dilate shadows to catch transition regions
            comprehensive_shadow = cv2.dilate(
                comprehensive_shadow.astype(np.uint8), shadow_kernel_small, iterations=1
            )
            comprehensive_shadow = cv2.morphologyEx(
                comprehensive_shadow, cv2.MORPH_CLOSE, shadow_kernel_large
            )
            comprehensive_shadow = comprehensive_shadow.astype(bool)

            # Remove comprehensive shadows from the mask
            refined = (mask_bin > 0) & (~comprehensive_shadow)
            refined = refined.astype(np.uint8) * 255

            # Aggressive cleaning of the refined mask
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, shadow_kernel_small)
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, shadow_kernel_large)

            return postprocess(refined)

        # Build candidate masks as before, but on rgb_work
        candidates: List[Tuple[str, np.ndarray]] = []
        bias = (self.cfg.bg_bias or "auto").lower()
        if self.cfg.mask_strategy == "hsv_s":
            obj = "light" if bias != "dark_bg" else "dark"
            candidates.append(("hsv_s", mask_hsv_s(object_type=obj)))
        elif self.cfg.mask_strategy == "hsv_v_dark":
            candidates.append(("hsv_v_dark", mask_hsv_v(object_type="dark")))
        elif self.cfg.mask_strategy == "hsv_h":
            candidates.append(("hsv_h", mask_hsv_green()))
        elif self.cfg.mask_strategy == "lab":
            candidates.append(("lab", mask_lab_green()))
        elif self.cfg.mask_strategy == "kmeans":
            candidates.append(("kmeans", mask_kmeans()))
        else:
            obj = "light" if bias != "dark_bg" else "dark"
            candidates.extend(
                [
                    ("hsv_s", mask_hsv_s(object_type=obj)),
                    ("hsv_v_dark", mask_hsv_v(object_type="dark")),
                    ("hsv_h", mask_hsv_green()),
                    ("lab", mask_lab_green()),
                    ("kmeans", mask_kmeans()),
                ]
            )

        best_mask = None
        best_cnt = None
        best_score = -1.0
        for _, raw in candidates:
            m, cnt = postprocess(raw)
            sc = score_mask(m, cnt)
            if sc > best_score:
                best_score = sc
                best_mask, best_cnt = m, cnt

        # Shadow suppression refinement if enabled
        if best_mask is not None and self.cfg.shadow_suppression:
            try:
                m_ref, c_ref = suppress_shadow(best_mask)
                sc_ref = score_mask(m_ref, c_ref)
                if sc_ref >= best_score:
                    best_mask, best_cnt, best_score = m_ref, c_ref, sc_ref
            except Exception:
                pass

        # Optional GrabCut refinement, done at work scale
        if best_mask is not None and self.cfg.grabcut_refine:
            try:
                h, w = best_mask.shape[:2]
                gc_mask = np.zeros((h, w), np.uint8)
                gc_mask[best_mask > 0] = cv2.GC_PR_FGD
                gc_mask[best_mask == 0] = cv2.GC_BGD
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                cv2.grabCut(
                    rgb_work,
                    gc_mask,
                    None,
                    bgdModel,
                    fgdModel,
                    1,
                    cv2.GC_INIT_WITH_MASK,
                )
                gc_bin = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(
                    np.uint8
                ) * 255
                m2, cnt2 = postprocess(gc_bin)
                sc2 = score_mask(m2, cnt2)
                if sc2 >= best_score:
                    best_mask, best_cnt, best_score = m2, cnt2, sc2
            except Exception:
                pass

        # Map back to original resolution
        if best_mask is None:
            # Fallback to legacy simple pipeline at work scale
            gray = pcv.rgb2gray_hsv(
                rgb_img=rgb_work, channel=self.cfg.hsv_channel_for_mask
            )
            th = pcv.threshold.otsu(gray_img=gray, object_type="light")
            filled = pcv.fill(bin_img=th, size=self.cfg.fill_size)
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.cfg.morph_kernel, self.cfg.morph_kernel)
            )
            closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k)
            cnt, _ = self._largest_contour(opened)
            if cnt is None:
                # Return resized opened to original size
                ret_mask = cv2.resize(opened, (ow, oh), interpolation=cv2.INTER_NEAREST)
                return ret_mask, None
            mask = self._contour_to_mask(opened.shape[:2], cnt)
            best_mask, best_cnt = mask, cnt

        # Extend mask to include brown/diseased areas (but only near the detected leaf)
        if best_mask is not None:
            # Create a dilated mask to define the search area for brown regions
            search_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            search_area = cv2.dilate(best_mask, search_kernel, iterations=2)
            search_constraint = search_area > 0

            # Detect brown regions constrained to the search area
            if self.cfg.use_lab_brown:
                lab = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                brown_regions = (
                    (a_ch >= self.cfg.lab_a_min)
                    & (b_ch >= self.cfg.lab_b_min)
                    & search_constraint
                )  # Constrain to search area
            else:
                hsv = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2HSV)
                h_ch, s_ch, v_ch = cv2.split(hsv)
                lo, hi = self.cfg.brown_hue_range
                brown_regions = (
                    (h_ch >= lo)
                    & (h_ch <= hi)
                    & (s_ch >= self.cfg.brown_s_min)
                    & (v_ch <= self.cfg.brown_v_max)
                ) & search_constraint  # Constrain to search area

            # Clean brown regions
            k_brown = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.cfg.brown_morph_kernel, self.cfg.brown_morph_kernel),
            )
            brown_clean = cv2.morphologyEx(
                brown_regions.astype(np.uint8) * 255, cv2.MORPH_OPEN, k_brown
            )
            brown_clean = cv2.morphologyEx(brown_clean, cv2.MORPH_CLOSE, k_brown)

            # Filter brown regions by size
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                brown_clean, connectivity=8
            )
            filtered_brown = np.zeros_like(brown_clean)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.cfg.brown_min_area_px:
                    filtered_brown[labels == i] = 255

            # Extend original mask to include brown regions (only those near the leaf)
            extended_mask = ((best_mask > 0) | (filtered_brown > 0)).astype(
                np.uint8
            ) * 255

            # Find new contour for extended mask
            cnts, _ = cv2.findContours(
                extended_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts:
                extended_cnt = max(cnts, key=cv2.contourArea)
                best_mask, best_cnt = extended_mask, extended_cnt

        if abs(s - 1.0) < 1e-6:
            return best_mask, best_cnt
        # Resize mask back
        ret_mask = cv2.resize(best_mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        # Rescale contour back
        cnt_scaled = None
        if best_cnt is not None:
            cnt_scaled = (best_cnt.astype(np.float32) / s).astype(np.int32)
        return ret_mask, cnt_scaled

    def roi(self, rgb, contour):
        import cv2
        import numpy as np

        if contour is None:
            return rgb, None, None
        x, y, w, h = cv2.boundingRect(contour)
        roi_img = rgb[y : y + h, x : x + w]
        # Resize to standard size (padding to keep aspect ratio)
        H, W = self.cfg.roi_size
        if roi_img.size == 0:
            return rgb, None, None
        # Letterbox to target size
        scale = min(W / max(w, 1), H / max(h, 1))
        nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
        resized = cv2.resize(roi_img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((H, W, 3), dtype=resized.dtype)
        oy, ox = (H - nh) // 2, (W - nw) // 2
        canvas[oy : oy + nh, ox : ox + nw] = resized
        # ROI visualization on original
        vis = rgb.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return canvas, vis, (x, y, w, h)

    def analyze(self, rgb, mask, contour):
        import cv2
        import numpy as np
        from plantcv import plantcv as pcv  # type: ignore

        if contour is None or mask is None:
            return draw_text(rgb, "Analyze: no object")
        # Use PlantCV to compute shape metrics; also build a visual overlay
        try:
            _ = pcv.analyze_object(
                img=rgb, obj=contour, mask=mask
            )  # populates pcv.outputs
        except Exception:
            pass

        overlay = rgb.copy()
        # Contour
        cv2.drawContours(overlay, [contour], -1, (255, 0, 0), 2)

        # Centroid
        M = cv2.moments(contour)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"] / M["m00"])  # type: ignore
            cy = int(M["m01"] / M["m00"])  # type: ignore
        else:
            cmean = contour[:, 0, :].mean(axis=0)
            cx, cy = int(cmean[0]), int(cmean[1])
        cv2.drawMarker(
            overlay,
            (cx, cy),
            (255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )

        # Extreme points (left, right, top, bottom) and rays from centroid
        pts = contour[:, 0, :]
        left = tuple(pts[pts[:, 0].argmin()])
        right = tuple(pts[pts[:, 0].argmax()])
        top = tuple(pts[pts[:, 1].argmin()])
        bottom = tuple(pts[pts[:, 1].argmax()])
        for p in (left, right, top, bottom):
            cv2.circle(overlay, (int(p[0]), int(p[1])), 3, (255, 255, 0), -1)
            cv2.line(
                overlay,
                (cx, cy),
                (int(p[0]), int(p[1])),
                (255, 255, 0),
                1,
                lineType=cv2.LINE_AA,
            )

        # Convex hull
        hull = cv2.convexHull(contour)
        cv2.polylines(
            overlay,
            [hull],
            isClosed=True,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # Major/minor axes via PCA
        data_pts = pts.astype(np.float32)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=None)
        if eigenvectors is not None and eigenvectors.shape[0] >= 2:
            v0 = eigenvectors[0]  # major axis direction
            v1 = eigenvectors[1]  # minor axis direction
            proj0 = data_pts @ v0
            proj1 = data_pts @ v1
            p0_min = data_pts[int(proj0.argmin())]
            p0_max = data_pts[int(proj0.argmax())]
            p1_min = data_pts[int(proj1.argmin())]
            p1_max = data_pts[int(proj1.argmax())]
            cv2.line(
                overlay,
                (int(p0_min[0]), int(p0_min[1])),
                (int(p0_max[0]), int(p0_max[1])),
                (255, 255, 0),
                2,
            )
            cv2.line(
                overlay,
                (int(p1_min[0]), int(p1_min[1])),
                (int(p1_max[0]), int(p1_max[1])),
                (255, 0, 255),
                2,
            )

        # Edge features (e.g., veins) within mask using Canny
        if mask.ndim == 2:
            m2 = mask > 0
        else:
            m2 = mask[..., 0] > 0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=80, threshold2=160, L2gradient=True)
        edge_bool = (edges > 0) & m2
        overlay[edge_bool] = (0, 255, 255)  # cyan edges

        # Textual metrics (calculated but not displayed)
        # area = float(cv2.contourArea(contour))
        # peri = float(cv2.arcLength(contour, True))
        # circularity = (4.0 * 3.1415926535 * area / (peri * peri)) if peri > 0 else 0.0

        return overlay

    def pseudolandmarks(self, rgb, contour):  # noqa: C901
        import cv2
        import numpy as np

        if contour is None:
            return draw_text(rgb, "Landmarks: no object")

        # Create enhanced leaf mask that includes brown spots
        mask, _ = self.make_mask(rgb)
        if mask is not None:
            # Get brown spots
            if mask.ndim == 2:
                leaf_mask = mask > 0
            else:
                leaf_mask = mask[..., 0] > 0

            # Detect brown regions
            if self.cfg.use_lab_brown:
                lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                brown_regions = (
                    (a >= self.cfg.lab_a_min) & (b >= self.cfg.lab_b_min) & leaf_mask
                )
            else:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                lo, hi = self.cfg.brown_hue_range
                brown_regions = (
                    (h >= lo)
                    & (h <= hi)
                    & (s >= self.cfg.brown_s_min)
                    & (v <= self.cfg.brown_v_max)
                    & leaf_mask
                )

            # Combine original leaf mask with brown regions
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            brown_clean = cv2.morphologyEx(
                brown_regions.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k
            )

            # Create enhanced mask
            enhanced_mask = (leaf_mask.astype(np.uint8) * 255) | brown_clean
            enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, k)

            # Find enhanced contour
            cnts, _ = cv2.findContours(
                enhanced_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts:
                enhanced_contour = max(cnts, key=cv2.contourArea)
                contour = enhanced_contour  # Use enhanced contour for landmarks

            mask_bool = enhanced_mask > 0
        else:
            mask_bool = None

        vis = rgb.copy()
        total = max(1, int(self.cfg.landmarks_count))
        # Rebalanced quotas (border/veins/disease seulement)
        border_quota = max(1, total // 3)
        vein_quota = max(1, total // 3)
        disease_quota = max(
            1, total - border_quota - vein_quota
        )  # Le reste pour diseases

        # Colors
        COL_BORDER = (255, 0, 0)  # blue when saved
        COL_VEIN = (0, 0, 255)  # red when saved
        # COL_ANOM = (42, 42, 165)  # brown-ish when saved
        COL_DISEASE = (139, 69, 19)  # dark brown for disease spots

        # 1) Border points: resample along contour
        c_pts = self._resample_contour(contour, border_quota)
        for x, y in c_pts:
            cv2.circle(vis, (int(x), int(y)), 2, COL_BORDER, -1, lineType=cv2.LINE_AA)
        cv2.polylines(
            vis,
            [contour],
            isClosed=True,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # Prepare helpers
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        inner_mask = None
        if mask_bool is not None:
            # Lightly erode for inner mask to avoid border influence
            k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            inner_mask = cv2.erode(
                (mask_bool.astype(np.uint8) * 255), k_inner, iterations=1
            )
            # Stricter inner mask for anomaly placement (commented out as unused)
            # k_inner5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # inner_mask_strict = cv2.erode(
            #     (mask_bool.astype(np.uint8) * 255), k_inner5, iterations=1
            # )

        # 2) Vein points (edges inside leaf) — moderate, robust settings
        try:
            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            # Edge maps with typical Canny thresholds
            edges1 = cv2.Canny(gray_eq, threshold1=30, threshold2=90, L2gradient=True)
            gray_bil = cv2.bilateralFilter(gray_eq, d=5, sigmaColor=50, sigmaSpace=50)
            edges2 = cv2.Canny(gray_bil, threshold1=50, threshold2=130, L2gradient=True)
            # Gradient magnitude threshold
            sx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(sx, sy)
            mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, edges3 = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)
            # Combine cues
            edges = cv2.bitwise_or(edges1, edges2)
            edges = cv2.bitwise_or(edges, edges3)
            if inner_mask is not None:
                edges = cv2.bitwise_and(edges, edges, mask=inner_mask)
            # Connect thin edges slightly
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges_d = cv2.dilate(edges, k3, iterations=1)

            # Corners on edges with reasonable density
            max_corners = max(1, vein_quota * 8)
            corners = cv2.goodFeaturesToTrack(
                image=gray_eq,
                maxCorners=max_corners,
                qualityLevel=0.002,
                minDistance=2,
                mask=edges_d,
                blockSize=3,
                useHarrisDetector=False,
                k=0.04,
            )
            placed = 0
            if corners is not None and len(corners) > 0:
                corners = np.squeeze(corners, axis=1)
                for x, y in corners[:vein_quota]:
                    cv2.circle(
                        vis, (int(x), int(y)), 2, COL_VEIN, -1, lineType=cv2.LINE_AA
                    )
                    placed += 1
            # Deterministic fallback from edge pixels if needed
            if placed < vein_quota:
                ys, xs = np.where(edges_d > 0)
                need = vein_quota - placed
                if xs.size > 0 and need > 0:
                    idxs = np.linspace(0, xs.size - 1, num=need, dtype=int)
                    for i in idxs:
                        cv2.circle(
                            vis,
                            (int(xs[i]), int(ys[i])),
                            2,
                            COL_VEIN,
                            -1,
                            lineType=cv2.LINE_AA,
                        )
                        placed += 1
        except Exception:
            pass

        # 3) Disease points (brown/diseased areas) - remplace les anomalies
        try:
            # Detect brown regions using the same logic as detect_brown_spots
            if self.cfg.use_lab_brown:
                lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                brown_regions = (a >= self.cfg.lab_a_min) & (b >= self.cfg.lab_b_min)
            else:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                lo, hi = self.cfg.brown_hue_range
                brown_regions = (
                    (h >= lo)
                    & (h <= hi)
                    & (s >= self.cfg.brown_s_min)
                    & (v <= self.cfg.brown_v_max)
                )

            # Constrain to leaf area
            if mask_bool is not None:
                brown_regions = brown_regions & mask_bool

            # Clean up noise
            k_brown = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.cfg.brown_morph_kernel, self.cfg.brown_morph_kernel),
            )
            brown_clean = cv2.morphologyEx(
                brown_regions.astype(np.uint8) * 255, cv2.MORPH_OPEN, k_brown
            )
            brown_clean = cv2.morphologyEx(brown_clean, cv2.MORPH_CLOSE, k_brown)

            # Find disease components and place landmarks
            num_brown, labels_brown, stats_brown, centroids_brown = (
                cv2.connectedComponentsWithStats(brown_clean, connectivity=8)
            )
            brown_comps = [
                (i, stats_brown[i, cv2.CC_STAT_AREA], tuple(centroids_brown[i]))
                for i in range(1, num_brown)
                if stats_brown[i, cv2.CC_STAT_AREA] >= self.cfg.brown_min_area_px
            ]
            brown_comps.sort(key=lambda t: t[1], reverse=True)

            disease_placed = 0
            total_brown_area = sum(comp[1] for comp in brown_comps)

            # Calculate proportional disease quota based on brown area
            # Densité modérée : 1 point par 50px (compromis entre précision et
            # faisabilité)
            calculated_disease_quota = max(len(brown_comps), total_brown_area // 50)
            actual_disease_quota = min(
                calculated_disease_quota, disease_quota * 5
            )  # Allow more than base quota

            logging.info(
                f"Brown area analysis: {total_brown_area} px in "
                f"{len(brown_comps)} regions → calculated quota: "
                f"{calculated_disease_quota}, using: {actual_disease_quota}"
            )

            for i, area, (cx, cy) in brown_comps:
                if disease_placed >= actual_disease_quota:
                    break

                # For larger disease areas, place multiple points proportionally
                comp_mask = (labels_brown == i).astype(np.uint8) * 255
                points_for_comp = max(
                    1, min(area // 40, actual_disease_quota - disease_placed)
                )  # 1 point per 40px

                disease_corners = cv2.goodFeaturesToTrack(
                    image=gray,
                    maxCorners=points_for_comp * 3,  # Get more candidates
                    qualityLevel=0.005,  # Lower quality threshold
                    minDistance=3,  # Reduced min distance for higher density
                    mask=comp_mask,
                    blockSize=3,
                    useHarrisDetector=False,
                    k=0.04,
                )

                if disease_corners is not None and len(disease_corners) > 0:
                    disease_corners = np.squeeze(disease_corners, axis=1)
                    for x, y in disease_corners[:points_for_comp]:
                        cv2.circle(
                            vis,
                            (int(x), int(y)),
                            4,
                            COL_DISEASE,
                            -1,
                            lineType=cv2.LINE_AA,
                        )
                        disease_placed += 1
                        if disease_placed >= disease_quota:
                            break
                else:
                    # Fallback to centroid
                    cv2.circle(
                        vis,
                        (int(cx), int(cy)),
                        4,
                        COL_DISEASE,
                        -1,
                        lineType=cv2.LINE_AA,
                    )
                    disease_placed += 1

            if disease_placed > 0:
                total_brown_px = sum(comp[1] for comp in brown_comps)
                logging.info(
                    f"Disease landmarks placed: {disease_placed} points on "
                    f"{len(brown_comps)} brown regions "
                    f"(total brown area: {total_brown_px} px)"
                )
            else:
                logging.info(
                    "No disease landmarks placed - no brown regions detected "
                    "or regions too small"
                )

        except Exception as e:
            logging.warning(f"Failed to detect disease landmarks: {e}")

        # Final summary log
        total_landmarks = border_quota + vein_quota + disease_placed
        logging.info(
            f"Landmarks summary: {border_quota} border + {vein_quota} veins + "
            f"{disease_placed} disease points = {total_landmarks} total landmarks"
        )

        return vis

    def detect_brown_spots(self, rgb, mask):
        """Detect brown/diseased areas in leaf"""
        import cv2
        import numpy as np

        if mask is None:
            return rgb, 0.0, 0

        # Create leaf-only region
        if mask.ndim == 2:
            leaf_mask = mask > 0
        else:
            leaf_mask = mask[..., 0] > 0

        if self.cfg.use_lab_brown:
            # LAB-based detection
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            brown_regions = (
                (a >= self.cfg.lab_a_min) & (b >= self.cfg.lab_b_min) & leaf_mask
            )
        else:
            # HSV-based detection
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            lo, hi = self.cfg.brown_hue_range
            brown_regions = (
                (h >= lo)
                & (h <= hi)
                & (s >= self.cfg.brown_s_min)
                & (v <= self.cfg.brown_v_max)
                & leaf_mask
            )

        # Morphological operations to clean up noise
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cfg.brown_morph_kernel, self.cfg.brown_morph_kernel),
        )
        brown_clean = cv2.morphologyEx(
            brown_regions.astype(np.uint8) * 255, cv2.MORPH_OPEN, k
        )
        brown_clean = cv2.morphologyEx(brown_clean, cv2.MORPH_CLOSE, k)

        # Filter by area
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            brown_clean, connectivity=8
        )
        filtered_brown = np.zeros_like(brown_clean)
        brown_count = 0
        total_brown_area = 0

        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.cfg.brown_min_area_px:
                filtered_brown[labels == i] = 255
                brown_count += 1
                total_brown_area += area

        # Calculate percentage of leaf affected
        leaf_area = np.sum(leaf_mask)
        brown_percentage = (total_brown_area / max(leaf_area, 1)) * 100

        # Create visualization
        vis = rgb.copy()
        vis[filtered_brown > 0] = [255, 100, 0]  # Orange overlay for brown spots

        logging.info(
            f"Brown spots detected: {brown_count} regions, "
            f"{brown_percentage:.1f}% of leaf area ({total_brown_area} pixels)"
        )

        return vis, brown_percentage, brown_count

    def histogram_hsv(self, rgb):
        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        fig = plt.figure(figsize=(6, 4))
        bins = 50
        plt.hist(h.ravel(), bins=bins, color="r", alpha=0.5, label="H")
        plt.hist(s.ravel(), bins=bins, color="g", alpha=0.5, label="S")
        plt.hist(v.ravel(), bins=bins, color="b", alpha=0.5, label="V")
        plt.title("HSV Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        fig.tight_layout()
        # Render to RGB image array
        fig.canvas.draw()
        w, h2 = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            (h2, w, 4)
        )
        rgb_img = rgba[..., :3].copy()
        plt.close(fig)
        return rgb_img

    # --- utilities ---
    def _largest_contour(self, mask):
        import cv2

        cnts, _ = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            return None, None
        cnt = max(cnts, key=cv2.contourArea)
        return cnt, None

    def _contour_to_mask(self, shape_hw: Tuple[int, int], contour):
        import cv2
        import numpy as np

        h, w = shape_hw
        out = np.zeros((h, w), dtype="uint8")
        cv2.drawContours(out, [contour], -1, color=255, thickness=-1)
        return out

    def _resample_contour(self, contour, n: int):
        import numpy as np

        pts = contour[:, 0, :].astype(float)  # (N,2)
        # Ensure closed loop
        if not (pts[0] == pts[-1]).all():
            pts = np.vstack([pts, pts[0]])
        # Cumulative arc length
        seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        cum = np.concatenate([[0.0], seg.cumsum()])
        total = cum[-1]
        if total == 0:
            return pts[:n]
        targets = np.linspace(0, total, num=n, endpoint=False)
        res = []
        j = 0
        for t in targets:
            while j + 1 < len(cum) and cum[j + 1] < t:
                j += 1
            # interpolate between pts[j] and pts[j+1]
            if j + 1 >= len(pts):
                res.append(pts[-1])
            else:
                dt = cum[j + 1] - cum[j]
                a = 0.0 if dt == 0 else (t - cum[j]) / dt
                res.append((1 - a) * pts[j] + a * pts[j + 1])
        return res


def build_types_filter(arg: Optional[str]) -> Tuple[str, ...]:
    if not arg:
        return DEFAULT_TYPES
    items = [s.strip() for s in str(arg).split(",") if s.strip()]
    result: List[str] = []
    for s in items:
        key = s.strip().lower()
        if key in CANONICAL_TYPES:
            result.append(CANONICAL_TYPES[key])
        else:
            logging.warning("Unknown transform type skipped: %s", s)
    # Preserve order, drop duplicates
    dedup = []
    for _name in result:
        if _name not in dedup:
            dedup.append(_name)
    return tuple(dedup) if dedup else DEFAULT_TYPES


def output_names(stem: str) -> Dict[str, str]:
    return {
        "Blur": f"{stem}__T_Blur.jpg",
        "Mask": f"{stem}__T_Mask.jpg",
        "ROI": f"{stem}__T_ROI.jpg",
        "Analyze": f"{stem}__T_Analyze.jpg",
        "Landmarks": f"{stem}__T_Landmarks.jpg",
        "Hist": f"{stem}__T_Hist.jpg",
        "Brown": f"{stem}__T_Brown.jpg",
    }


def process_single_image(params: ProcessArgs) -> List[Path]:  # noqa: C901
    # Read
    try:
        rgb = pil_read_rgb(params.img_path)
    except Exception as exc:
        logging.error("Failed to read %s (%s)", params.img_path, exc)
        return []

    pipe = TransformPipeline(params.cfg)
    saved: List[Path] = []

    names = output_names(params.img_path.stem)

    # Blur
    if "Blur" in params.types:
        blur_img = pipe.blur(rgb)
        out = params.out_dir / names["Blur"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, blur_img)
            saved.append(out)

    # Mask
    mask_img, contour = (None, None)
    if (
        "Mask" in params.types
        or "ROI" in params.types
        or "Analyze" in params.types
        or "Landmarks" in params.types
        or "Brown" in params.types
    ):
        mask_img, contour = pipe.make_mask(rgb)
    if "Mask" in params.types:
        if mask_img is not None:
            # Apply mask to RGB
            # Build a 2D boolean mask, then broadcast to 3 channels via multiplication
            if mask_img.ndim == 2:
                m2 = mask_img > 0
            else:
                m2 = mask_img[..., 0] > 0
            masked_rgb = (rgb * m2[..., None]).astype(rgb.dtype)
            out = params.out_dir / names["Mask"]
            if params.overwrite or (not params.skip_existing or not out.exists()):
                imwrite_bgr(out, masked_rgb)
                saved.append(out)
        else:
            out = params.out_dir / names["Mask"]
            if params.overwrite or (not params.skip_existing or not out.exists()):
                imwrite_bgr(out, draw_text(rgb, "Mask: none"))
                saved.append(out)

    # ROI
    if "ROI" in params.types:
        roi_img, roi_vis, _ = pipe.roi(rgb, contour)
        out = params.out_dir / names["ROI"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, roi_vis if roi_vis is not None else rgb)
            saved.append(out)
        # Optionally save standardized ROI crop if needed
        # roi_save = params.out_dir / f"{params.img_path.stem}__T_ROI_crop.jpg"
        # imwrite_bgr(roi_save, roi_img)

    # Analyze
    if "Analyze" in params.types:
        analyze_img = pipe.analyze(rgb, mask_img, contour)
        out = params.out_dir / names["Analyze"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, analyze_img)
            saved.append(out)

    # Landmarks
    if "Landmarks" in params.types:
        lm_img = pipe.pseudolandmarks(rgb, contour)
        out = params.out_dir / names["Landmarks"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, lm_img)
            saved.append(out)

    # Hist
    if "Hist" in params.types:
        hist_img = pipe.histogram_hsv(rgb)
        out = params.out_dir / names["Hist"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, hist_img)
            saved.append(out)

    # Brown spots detection
    if "Brown" in params.types:
        brown_img, brown_percentage, brown_count = pipe.detect_brown_spots(
            rgb, mask_img
        )
        out = params.out_dir / names["Brown"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, brown_img)
            saved.append(out)

    return saved


def iter_images_in_dir(src: Path) -> Iterable[Path]:
    for p in sorted(src.rglob("*")):
        if is_image(p):
            yield p


# Top-level worker for multiprocessing portability
def _worker_run(
    ip_str: str,
    out_dir_str: str,
    types: Sequence[str],
    cfg: TransformConfig,
    skip_existing: bool,
    overwrite: bool,
) -> Tuple[str, List[str]]:
    ip = Path(ip_str)
    out_d = Path(out_dir_str)
    params = ProcessArgs(
        img_path=ip,
        out_dir=out_d,
        types=tuple(types),
        cfg=cfg,
        skip_existing=skip_existing,
        overwrite=overwrite,
    )
    saved_paths = process_single_image(params)
    return (str(ip), [str(p) for p in saved_paths])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Image transformation pipeline (PlantCV).\n"
            "- Single image: Transformation.py path/to/image.jpg\n"
            "- Folder mode: Transformation.py -src DIR -dst OUTDIR [--workers N]"
        )
    )
    p.add_argument("image", nargs="?", help="Path to a single image for preview mode")
    p.add_argument(
        "--out-dir", default=None, help="Output directory for single image preview"
    )
    p.add_argument("-src", "--src", default=None, help="Source directory (folder mode)")
    p.add_argument(
        "-dst", "--dst", default=None, help="Destination directory (folder mode)"
    )
    p.add_argument(
        "--types",
        default=",".join(DEFAULT_TYPES),
        help="Comma-separated transforms to run",
    )
    p.add_argument(
        "--config", default="transform/config.yaml", help="YAML config path (optional)"
    )
    p.add_argument(
        "--workers", type=int, default=0, help="Number of processes (0=auto)"
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose outputs already exist",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )
    p.add_argument(
        "--preview", action="store_true", help="Force saving outputs (no GUI popups)"
    )
    return p.parse_args()


def main() -> None:  # noqa: C901
    args = parse_args()
    setup_logging()

    types = build_types_filter(args.types)
    cfg = load_config(Path(args.config) if args.config else None)

    # Single-image mode
    if args.image and not args.src and not args.dst:
        img_path = Path(args.image)
        if not img_path.exists():
            logging.error("Image not found: %s", img_path)
            return
        stem_dir = img_path.stem
        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else Path("artifacts/transformations") / stem_dir
        )
        ensure_dir(out_dir)
        params = ProcessArgs(
            img_path=img_path,
            out_dir=out_dir,
            types=tuple(types),
            cfg=cfg,
            skip_existing=args.skip_existing,
            overwrite=args.overwrite,
        )
        saved = process_single_image(params)
        if not saved:
            logging.warning("No outputs produced")
        else:
            for p in saved:
                logging.info("Wrote: %s", p.resolve())
        return

    # Folder mode
    if not args.src or not args.dst:
        logging.error(
            "Provide either a single IMAGE or both -src and -dst for folder mode"
        )
        return

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists() or not src.is_dir():
        logging.error("Source directory not found: %s", src)
        return

    all_imgs = list(iter_images_in_dir(src))
    if not all_imgs:
        logging.warning("No images found in: %s", src)
        return

    workers = (
        args.workers
        if args.workers and args.workers > 0
        else max(os.cpu_count() or 1, 1) - 1
    )
    workers = max(1, workers)

    # Prepare tasks
    tasks: List[Tuple[Path, Path]] = []
    for ip in all_imgs:
        rel = ip.parent.relative_to(src)
        # Place each image's outputs under dst/<rel>/<stem>/
        out_dir = dst / rel / ip.stem
        if args.skip_existing and not args.overwrite:
            names = output_names(ip.stem)
            outs = [out_dir / names[k] for k in types if k in names]
            if outs and all(p.exists() for p in outs):
                continue
        tasks.append((ip, out_dir))

    if not tasks:
        logging.info("All up to date, nothing to do.")
        return

    logging.info("Processing %d images with %d worker(s)", len(tasks), workers)

    if workers == 1:
        for ip, out_d in tasks:
            params = ProcessArgs(
                img_path=ip,
                out_dir=out_d,
                types=tuple(types),
                cfg=cfg,
                skip_existing=args.skip_existing,
                overwrite=args.overwrite,
            )
            _, outs = (str(ip), [str(p) for p in process_single_image(params)])
            for o in outs:
                logging.info("Wrote: %s", Path(o).resolve())
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _worker_run,
                    str(ip),
                    str(od),
                    types,
                    cfg,
                    args.skip_existing,
                    args.overwrite,
                )
                for ip, od in tasks
            ]
            for fut in as_completed(futs):
                try:
                    _, outs = fut.result()
                    for o in outs:
                        logging.info("Wrote: %s", Path(o).resolve())
                except Exception as exc:
                    logging.error("Worker failed: %s", exc)


if __name__ == "__main__":
    main()
