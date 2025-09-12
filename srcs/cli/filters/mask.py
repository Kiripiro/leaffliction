from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from plantcv import plantcv as pcv

try:
    from ..Transformation import TransformConfig, contour_to_mask, largest_contour
except ImportError:
    import sys
    from pathlib import Path

    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from Transformation import TransformConfig, contour_to_mask, largest_contour


def _prepare_working_image(
    rgb: np.ndarray, cfg: TransformConfig
) -> Tuple[np.ndarray, float]:
    oh, ow = rgb.shape[:2]
    s = 1.0
    if cfg.mask_upscale_factor and cfg.mask_upscale_factor > 1.0:
        s = float(cfg.mask_upscale_factor)
    elif cfg.mask_upscale_long_side and cfg.mask_upscale_long_side > 0:
        ls = max(oh, ow)
        if ls < cfg.mask_upscale_long_side:
            s = float(cfg.mask_upscale_long_side) / float(ls)

    rgb_work = (
        rgb
        if abs(s - 1.0) < 1e-6
        else cv2.resize(
            rgb,
            (int(round(ow * s)), int(round(oh * s))),
            interpolation=cv2.INTER_CUBIC,
        )
    )
    return rgb_work, s


def _postprocess_mask(
    bin_img: np.ndarray, cfg: TransformConfig
) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    # Ensure binary uint8
    b = (bin_img > 0).astype(np.uint8) * 255
    filled = pcv.fill(bin_img=b, size=cfg.fill_size)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
    )
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k)
    cnt, _ = largest_contour(opened)
    if cnt is None:
        return opened, None
    mask = contour_to_mask(opened.shape[:2], cnt)
    return mask, cnt


def _create_hsv_masks(
    rgb_work: np.ndarray, cfg: TransformConfig, bias: str
) -> List[Tuple[str, np.ndarray]]:

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
        lo, hi = cfg.green_hue_range
        m = ((h >= lo) & (h <= hi) & (s >= 40)).astype(np.uint8) * 255
        return m

    obj = "light" if bias != "dark_bg" else "dark"
    return [
        ("hsv_s", mask_hsv_s(object_type=obj)),
        ("hsv_v_dark", mask_hsv_v(object_type="dark")),
        ("hsv_h", mask_hsv_green()),
    ]


def _create_lab_mask(rgb_work: np.ndarray) -> np.ndarray:

    lab = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    m = ((a <= 135) & (b >= 115) & (b <= 170)).astype(np.uint8) * 255
    return m


def _create_kmeans_mask(rgb_work: np.ndarray, cfg: TransformConfig) -> np.ndarray:

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
    _K, labels, centers = cv2.kmeans(Z, 3, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    lbl = labels.reshape(small.shape[:2])
    hsv_c = cv2.cvtColor(centers.reshape((1, 3, 3)), cv2.COLOR_RGB2HSV)[0]
    lo, hi = cfg.green_hue_range
    green_score = np.array(
        [(1 if (lo <= hv[0] <= hi and hv[1] >= 40) else 0) for hv in hsv_c]
    )
    if cfg.bg_bias == "dark_bg":
        pick = int(np.argmax(centers.mean(axis=1)))
    elif cfg.bg_bias == "light_bg":
        pick = int(np.argmin(centers.mean(axis=1)))
    elif green_score.any():
        pick = int(np.argmax(green_score))
    else:
        sat = hsv_c[:, 1]
        pick = int(np.argmax(sat))
    ms = (lbl == pick).astype(np.uint8) * 255
    m = cv2.resize(ms, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def _score_mask(
    mask_bin: np.ndarray,
    cnt: Optional[np.ndarray],
    rgb_work: np.ndarray,
    cfg: TransformConfig,
) -> float:

    if cnt is None:
        return -1.0
    h, w = mask_bin.shape[:2]
    area = float(cv2.contourArea(cnt))
    if area <= 1:
        return -1.0
    area_ratio = area / float(h * w)
    if area_ratio < cfg.min_object_area_ratio or area_ratio > cfg.max_object_area_ratio:
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
    lo, hi = cfg.green_hue_range
    green = (Hc >= lo) & (Hc <= hi) & (Sc >= 40)
    denom = max(1, int(mask_bin.sum() // 255))
    green_frac = float((green & (mask_bin > 0)).sum()) / float(denom)
    # Border-touch penalty
    x, y, ww, hh = cv2.boundingRect(cnt)
    touches_border = (x <= 0) or (y <= 0) or (x + ww >= w - 1) or (y + hh >= h - 1)
    # Weighted score
    target = 0.35
    area_term = max(0.0, 1.0 - abs(area_ratio - target) / target)
    score = 0.35 * area_term + 0.25 * solidity + 0.25 * b_strength + 0.15 * green_frac
    if touches_border:
        score *= 0.75
    return float(score)


def _suppress_shadow(
    mask_bin: np.ndarray, rgb_work: np.ndarray, cfg: TransformConfig
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
    aggressive_shadow = (L_ch < np.percentile(L_ch, 45)) & (Sc < 60) & (Vc < 120)

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
    rgb_small = cv2.resize(rgb_work, (small_w, small_h), interpolation=cv2.INTER_AREA)

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
    lo, hi = cfg.green_hue_range
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

    return _postprocess_mask(refined, cfg)


def _apply_grabcut_refinement(
    best_mask: np.ndarray, rgb_work: np.ndarray, cfg: TransformConfig
) -> Tuple[np.ndarray, Optional[np.ndarray]]:

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
        return _postprocess_mask(gc_bin, cfg)
    except Exception:
        return best_mask, None


def _extend_mask_with_brown_regions(
    best_mask: np.ndarray, rgb_work: np.ndarray, cfg: TransformConfig
) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    # Create a dilated mask to define the search area for brown regions
    search_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    search_area = cv2.dilate(best_mask, search_kernel, iterations=2)
    search_constraint = search_area > 0

    # Detect brown regions constrained to the search area
    if cfg.use_lab_brown:
        lab = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        brown_regions = (
            (a_ch >= cfg.lab_a_min) & (b_ch >= cfg.lab_b_min) & search_constraint
        )  # Constrain to search area
    else:
        hsv = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        lo, hi = cfg.brown_hue_range
        brown_regions = (
            (h_ch >= lo)
            & (h_ch <= hi)
            & (s_ch >= cfg.brown_s_min)
            & (v_ch <= cfg.brown_v_max)
        ) & search_constraint  # Constrain to search area

    # Clean brown regions
    k_brown = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.brown_morph_kernel, cfg.brown_morph_kernel),
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
        if area >= cfg.brown_min_area_px:
            filtered_brown[labels == i] = 255

    # Extend original mask to include brown regions (only those near the leaf)
    extended_mask = ((best_mask > 0) | (filtered_brown > 0)).astype(np.uint8) * 255

    # Find new contour for extended mask
    cnts, _ = cv2.findContours(
        extended_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if cnts:
        extended_cnt = max(cnts, key=cv2.contourArea)
        return extended_mask, extended_cnt
    return best_mask, None


def _create_fallback_mask(
    rgb_work: np.ndarray, cfg: TransformConfig
) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    gray = pcv.rgb2gray_hsv(rgb_img=rgb_work, channel=cfg.hsv_channel_for_mask)
    th = pcv.threshold.otsu(gray_img=gray, object_type="light")
    filled = pcv.fill(bin_img=th, size=cfg.fill_size)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
    )
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k)
    cnt, _ = largest_contour(opened)
    if cnt is None:
        return opened, None
    mask = contour_to_mask(opened.shape[:2], cnt)
    return mask, cnt


def _build_mask_candidates(
    rgb_work: np.ndarray, cfg: TransformConfig
) -> List[Tuple[str, np.ndarray]]:

    candidates: List[Tuple[str, np.ndarray]] = []
    bias = (cfg.bg_bias or "auto").lower()

    if cfg.mask_strategy == "hsv_s":
        candidates.extend(_create_hsv_masks(rgb_work, cfg, bias)[:1])
    elif cfg.mask_strategy == "hsv_v_dark":
        candidates.extend(_create_hsv_masks(rgb_work, cfg, bias)[1:2])
    elif cfg.mask_strategy == "hsv_h":
        candidates.extend(_create_hsv_masks(rgb_work, cfg, bias)[2:3])
    elif cfg.mask_strategy == "lab":
        candidates.append(("lab", _create_lab_mask(rgb_work)))
    elif cfg.mask_strategy == "kmeans":
        candidates.append(("kmeans", _create_kmeans_mask(rgb_work, cfg)))
    elif cfg.mask_strategy == "enhanced":
        candidates.append(("enhanced", _create_enhanced_mask(rgb_work, cfg)))
    elif cfg.mask_strategy == "inclusive":
        candidates.append(("inclusive", _create_inclusive_mask(rgb_work, cfg)))
    else:
        # Auto mode: try all strategies including enhanced
        candidates.extend(_create_hsv_masks(rgb_work, cfg, bias))
        candidates.append(("lab", _create_lab_mask(rgb_work)))
        candidates.append(("kmeans", _create_kmeans_mask(rgb_work, cfg)))
        candidates.append(("enhanced", _create_enhanced_mask(rgb_work, cfg)))
        candidates.append(("inclusive", _create_inclusive_mask(rgb_work, cfg)))

    return candidates


def _find_best_mask(
    candidates: List[Tuple[str, np.ndarray]], rgb_work: np.ndarray, cfg: TransformConfig
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:

    best_mask = None
    best_cnt = None
    best_score = -1.0

    for _, raw in candidates:
        m, cnt = _postprocess_mask(raw, cfg)
        sc = _score_mask(m, cnt, rgb_work, cfg)
        if sc > best_score:
            best_score = sc
            best_mask, best_cnt = m, cnt

    return best_mask, best_cnt, best_score


def _apply_refinements(
    best_mask: Optional[np.ndarray],
    best_cnt: Optional[np.ndarray],
    best_score: float,
    rgb_work: np.ndarray,
    cfg: TransformConfig,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    current_mask, current_cnt, current_score = best_mask, best_cnt, best_score

    # Shadow suppression refinement
    if current_mask is not None and cfg.shadow_suppression:
        try:
            m_ref, c_ref = _suppress_shadow(current_mask, rgb_work, cfg)
            sc_ref = _score_mask(m_ref, c_ref, rgb_work, cfg)
            if sc_ref >= current_score:
                current_mask, current_cnt, current_score = m_ref, c_ref, sc_ref
        except Exception:
            pass

    # GrabCut refinement
    if current_mask is not None and cfg.grabcut_refine:
        m2, cnt2 = _apply_grabcut_refinement(current_mask, rgb_work, cfg)
        if m2 is not None:
            sc2 = _score_mask(m2, cnt2, rgb_work, cfg)
            if sc2 >= current_score:
                current_mask, current_cnt = m2, cnt2

    return current_mask, current_cnt


def _handle_fallback_and_extension(
    best_mask: Optional[np.ndarray],
    best_cnt: Optional[np.ndarray],
    rgb_work: np.ndarray,
    cfg: TransformConfig,
    ow: int,
    oh: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    current_mask, current_cnt = best_mask, best_cnt

    # Create fallback if no mask found
    if current_mask is None:
        current_mask, current_cnt = _create_fallback_mask(rgb_work, cfg)
        if current_mask is None:
            ret_mask = cv2.resize(
                np.zeros((rgb_work.shape[0], rgb_work.shape[1]), dtype=np.uint8),
                (ow, oh),
                interpolation=cv2.INTER_NEAREST,
            )
            return ret_mask, None

    # Extend with brown regions
    if current_mask is not None:
        current_mask, current_cnt = _extend_mask_with_brown_regions(
            current_mask, rgb_work, cfg
        )

    return current_mask, current_cnt


def _resize_results_to_original(
    mask: Optional[np.ndarray],
    cnt: Optional[np.ndarray],
    scale_factor: float,
    ow: int,
    oh: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    if abs(scale_factor - 1.0) < 1e-6:
        return mask, cnt

    # Resize mask back
    ret_mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

    # Rescale contour back
    cnt_scaled = None
    if cnt is not None:
        cnt_scaled = (cnt.astype(np.float32) / scale_factor).astype(np.int32)

    return ret_mask, cnt_scaled


def make_mask(
    rgb: np.ndarray, cfg: TransformConfig
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create a robust leaf mask using multiple strategies.

    Args:
        rgb: Input RGB image
        cfg: Transform configuration

    Returns:
        Tuple of (mask, contour)
    """
    # Prepare working image with optional upscaling
    rgb_work, scale_factor = _prepare_working_image(rgb, cfg)
    oh, ow = rgb.shape[:2]

    # Build candidate masks based on strategy
    candidates = _build_mask_candidates(rgb_work, cfg)

    # Find the best mask among candidates
    best_mask, best_cnt, best_score = _find_best_mask(candidates, rgb_work, cfg)

    # Apply refinements (shadow suppression, GrabCut)
    best_mask, best_cnt = _apply_refinements(
        best_mask, best_cnt, best_score, rgb_work, cfg
    )

    # Handle fallback and extension with brown regions
    best_mask, best_cnt = _handle_fallback_and_extension(
        best_mask, best_cnt, rgb_work, cfg, ow, oh
    )

    # Resize results back to original dimensions
    return _resize_results_to_original(best_mask, best_cnt, scale_factor, ow, oh)


def apply_mask_filter(
    rgb: np.ndarray, cfg: TransformConfig, make_mask_func: callable
) -> np.ndarray:
    """
    Apply mask to RGB image using PlantCV-style masking with black background.

    Args:
        rgb: Input RGB image
        cfg: Transform configuration
        make_mask_func: Function to create leaf mask

    Returns:
        Masked RGB image with black background (PlantCV style)
    """
    mask_img, _ = make_mask_func(rgb)

    if mask_img is not None:
        # Use the new PlantCV-style apply_mask function with black background
        from srcs.utils.mask_utils import apply_mask

        return apply_mask(rgb, mask_img, mask_color="black")
    else:
        return rgb


def _create_enhanced_mask(rgb_work: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    """
    Create enhanced mask that properly isolates the entire plant from background.

    This improved method:
    1. Uses more inclusive color space thresholds to capture all plant tissue
    2. Combines multiple approaches with OR logic instead of AND
    3. Includes brown/diseased areas as part of the plant
    4. Better handles shadows and variations in lighting
    """
    h, w = rgb_work.shape[:2]

    # Convert to multiple color spaces
    hsv = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2LAB)

    # 1. More inclusive HSV-based vegetation detection
    h_chan, s_chan, v_chan = cv2.split(hsv)
    lo, hi = cfg.green_hue_range

    # More inclusive hue range - capture green and yellowish areas
    hue_mask = ((h_chan >= lo) & (h_chan <= hi)).astype(np.uint8)

    # Lower saturation threshold to include less vivid plant areas
    sat_mask = (s_chan >= 25).astype(np.uint8)  # Reduced from 80 to 25

    # More inclusive brightness range to capture shadows and highlights
    val_mask = ((v_chan >= 20) & (v_chan <= 240)).astype(np.uint8)  # More inclusive

    # Combine HSV conditions with AND
    hsv_vegetation = hue_mask & sat_mask & val_mask

    # 2. LAB-based vegetation detection (more inclusive)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # More inclusive LAB detection to capture all plant tissue
    lab_vegetation = (
        (a_chan <= 135)  # Relaxed from 125 to 135
        & (b_chan >= 105)  # Relaxed from 115 to 105
        & (l_chan >= 30)  # Relaxed from 50 to 30
        & (l_chan <= 220)  # More inclusive brightness
    ).astype(np.uint8)

    # 3. Brown/diseased area detection to include in plant mask
    brown_areas = np.zeros_like(h_chan, dtype=np.uint8)

    if cfg.use_lab_brown:
        # LAB-based brown detection
        brown_areas = (
            (a_chan >= cfg.lab_a_min - 10)  # Slightly more inclusive
            & (b_chan >= cfg.lab_b_min - 10)
            & (l_chan >= 20)  # Include darker brown areas
        ).astype(np.uint8)
    else:
        # HSV-based brown detection with expanded range
        lo_brown, hi_brown = cfg.brown_hue_range
        # Expand brown range to include more yellow-brown variations
        brown_hue1 = (h_chan >= lo_brown) & (h_chan <= hi_brown + 20)
        brown_hue2 = (h_chan >= 160) & (h_chan <= 180)  # Red-brown range
        brown_areas = (
            (brown_hue1 | brown_hue2)
            & (s_chan >= cfg.brown_s_min - 10)  # More inclusive
            & (v_chan <= cfg.brown_v_max + 30)  # Include lighter brown areas
        ).astype(np.uint8)

    # 4. Edge-based detection for plant boundaries
    gray = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2GRAY)

    # Multi-scale edge detection
    edges1 = cv2.Canny(gray, 30, 100)  # Lower thresholds for more edges
    edges2 = cv2.Canny(gray, 50, 150)
    edges_combined = cv2.bitwise_or(edges1, edges2)

    # Dilate edges to create broader edge regions
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_regions = cv2.dilate(edges_combined, edge_kernel, iterations=2)

    # 5. Combine all vegetation indicators with OR logic (more inclusive)
    vegetation_mask = hsv_vegetation | lab_vegetation | brown_areas

    # Add edge information to capture plant boundaries
    edge_enhanced = vegetation_mask.astype(np.float32)
    edge_enhanced += (edge_regions.astype(np.float32) / 255.0) * 0.3

    # Lower threshold for more inclusive mask
    enhanced_mask = (edge_enhanced > 0.3).astype(np.uint8) * 255

    # 6. Morphological operations to clean and connect plant regions
    # Fill small holes first
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, fill_kernel)

    # Remove very small noise
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, noise_kernel)

    # Connect nearby plant regions
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, connect_kernel)

    # 7. Keep largest connected component (the main plant)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        enhanced_mask, connectivity=8
    )

    if num_labels > 1:
        # Find the largest component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        enhanced_mask = (labels == largest_label).astype(np.uint8) * 255

    # 8. Final smoothing to create clean boundaries
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, smooth_kernel)

    return enhanced_mask


def _create_inclusive_mask(rgb_work: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    """
    Create highly effective mask to isolate leaf from gray/purple backgrounds.

    Focus on strong green detection and aggressive background removal.
    """
    h, w = rgb_work.shape[:2]

    # Convert to multiple color spaces
    hsv = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2LAB)

    h_chan, s_chan, v_chan = cv2.split(hsv)
    l_chan, a_chan, b_chan = cv2.split(lab)
    r_chan, g_chan, b_chan_rgb = cv2.split(rgb_work)

    # Strategy 1: Strong green detection in HSV
    lo, hi = cfg.green_hue_range
    # Expand green range slightly to catch all green variations
    expanded_lo = max(0, lo - 10)
    expanded_hi = min(179, hi + 15)

    green_hue = (h_chan >= expanded_lo) & (h_chan <= expanded_hi)
    strong_green = (green_hue & (s_chan >= 30) & (v_chan >= 30)).astype(np.uint8)

    # Strategy 2: Green channel dominance (very effective for leaves)
    green_dominant = (
        (g_chan > r_chan + 15)
        | (g_chan > b_chan_rgb + 15)
        | ((g_chan > r_chan + 5) & (g_chan > b_chan_rgb + 5) & (s_chan >= 20))
    ).astype(np.uint8)

    # Strategy 3: LAB green detection (robust to lighting)
    lab_green = (
        (a_chan <= 125)
        & (b_chan >= 120)  # Green-yellow in LAB
        & (l_chan >= 20)
        & (l_chan <= 240)  # Reasonable brightness
    ).astype(np.uint8)

    # Strategy 4: Aggressive background removal
    # Detect gray/purple backgrounds typical in plant images
    gray = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (15, 15), 0)
    texture_diff = cv2.absdiff(gray, blur_gray)

    gray_purple_bg = (
        # Gray backgrounds (low saturation)
        ((s_chan <= 25) & (v_chan >= 50) & (v_chan <= 220))
        # Purple/violet backgrounds (specific hue range)
        | (
            (h_chan >= 120)
            & (h_chan <= 160)
            & (s_chan >= 20)
            & (r_chan > g_chan)
            & (b_chan_rgb > g_chan)
        )
        # Very uniform areas (no texture)
        | ((s_chan <= 15) & (texture_diff < 10))
    ).astype(np.uint8)

    # Strategy 5: Edge-enhanced detection
    edges = cv2.Canny(gray, 30, 100)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_edges = cv2.dilate(edges, edge_kernel)

    # Combine all positive indicators
    plant_candidates = (
        strong_green | green_dominant | lab_green | (dilated_edges > 0)
    ).astype(np.uint8)

    # Remove background explicitly
    plant_mask = plant_candidates & (~gray_purple_bg.astype(bool)).astype(np.uint8)

    # Convert to full intensity
    plant_mask = plant_mask * 255

    # Morphological cleaning - aggressive to get clean boundaries
    # Remove small noise first
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, noise_kernel)

    # Fill gaps in the leaf
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, fill_kernel)

    # Final connection of nearby regions
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, connect_kernel)

    # Keep only the largest connected component (main leaf)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        plant_mask, connectivity=8
    )

    if num_labels > 1:
        # Find the largest component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        plant_mask = (labels == largest_label).astype(np.uint8) * 255

    # Final boundary smoothing
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, smooth_kernel)

    return plant_mask
