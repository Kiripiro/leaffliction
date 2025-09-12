from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

try:
    from srcs.cli.Transformation import (
        TransformConfig,
        draw_text,
        resample_contour,
    )
except Exception:
    import sys
    from pathlib import Path

    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from srcs.cli.Transformation import (
        TransformConfig,
        draw_text,
        resample_contour,
    )


def _create_enhanced_mask(rgb, cfg, mask_bool):
    if cfg.use_lab_brown:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        brown_regions = (a >= cfg.lab_a_min) & (b >= cfg.lab_b_min) & mask_bool
    else:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        lo, hi = cfg.brown_hue_range
        brown_regions = (
            (h >= lo)
            & (h <= hi)
            & (s >= cfg.brown_s_min)
            & (v <= cfg.brown_v_max)
            & mask_bool
        )

    # Clean up brown regions
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    brown_clean = cv2.morphologyEx(
        brown_regions.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k
    )

    # Create enhanced mask
    enhanced_mask = (mask_bool.astype(np.uint8) * 255) | brown_clean
    enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, k)

    return enhanced_mask


def _place_border_landmarks(vis, contour, border_quota):

    COL_BORDER = (255, 0, 0)
    c_pts = resample_contour(contour, border_quota)
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
    return border_quota


def _place_vein_landmarks(vis, rgb, mask_bool, vein_quota):

    COL_VEIN = (0, 0, 255)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Create inner mask
    inner_mask = None
    if mask_bool is not None:
        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_mask = cv2.erode(
            (mask_bool.astype(np.uint8) * 255), k_inner, iterations=1
        )

    placed = 0
    try:
        # Enhance contrast and detect edges
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        edges1 = cv2.Canny(gray_eq, threshold1=30, threshold2=90, L2gradient=True)
        gray_bil = cv2.bilateralFilter(gray_eq, d=5, sigmaColor=50, sigmaSpace=50)
        edges2 = cv2.Canny(gray_bil, threshold1=50, threshold2=130, L2gradient=True)

        sx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges3 = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)

        edges = cv2.bitwise_or(edges1, edges2)
        edges = cv2.bitwise_or(edges, edges3)

        if inner_mask is not None:
            edges = cv2.bitwise_and(edges, edges, mask=inner_mask)

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_d = cv2.dilate(edges, k3, iterations=1)

        # Find corners
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

        if corners is not None and len(corners) > 0:
            corners = np.squeeze(corners, axis=1)
            for x, y in corners[:vein_quota]:
                cv2.circle(vis, (int(x), int(y)), 2, COL_VEIN, -1, lineType=cv2.LINE_AA)
                placed += 1

        # Fallback if not enough corners found
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

    return placed


def _place_disease_landmarks(vis, rgb, cfg, mask_bool, disease_quota):

    COL_DISEASE = (139, 69, 19)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    disease_placed = 0

    try:
        # Detect brown regions
        if cfg.use_lab_brown:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            brown_regions = (a >= cfg.lab_a_min) & (b >= cfg.lab_b_min)
        else:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            lo, hi = cfg.brown_hue_range
            brown_regions = (
                (h >= lo) & (h <= hi) & (s >= cfg.brown_s_min) & (v <= cfg.brown_v_max)
            )

        if mask_bool is not None:
            brown_regions = brown_regions & mask_bool

        # Clean up regions
        k_brown = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.brown_morph_kernel, cfg.brown_morph_kernel)
        )
        brown_clean = cv2.morphologyEx(
            brown_regions.astype(np.uint8) * 255, cv2.MORPH_OPEN, k_brown
        )
        brown_clean = cv2.morphologyEx(brown_clean, cv2.MORPH_CLOSE, k_brown)

        # Find components
        num_brown, labels_brown, stats_brown, centroids_brown = (
            cv2.connectedComponentsWithStats(brown_clean, connectivity=8)
        )
        brown_comps = [
            (i, stats_brown[i, cv2.CC_STAT_AREA], tuple(centroids_brown[i]))
            for i in range(1, num_brown)
            if stats_brown[i, cv2.CC_STAT_AREA] >= cfg.brown_min_area_px
        ]
        brown_comps.sort(key=lambda t: t[1], reverse=True)

        total_brown_area = sum(comp[1] for comp in brown_comps)
        calculated_disease_quota = max(len(brown_comps), total_brown_area // 50)
        actual_disease_quota = min(calculated_disease_quota, disease_quota * 5)

        logging.info(
            f"Brown area analysis: {total_brown_area} px in "
            f"{len(brown_comps)} regions → calculated quota: "
            f"{calculated_disease_quota}, using: {actual_disease_quota}"
        )

        # Place landmarks on disease areas
        for i, area, (cx, cy) in brown_comps:
            if disease_placed >= actual_disease_quota:
                break

            comp_mask = (labels_brown == i).astype(np.uint8) * 255
            points_for_comp = max(
                1, min(area // 40, actual_disease_quota - disease_placed)
            )

            disease_corners = cv2.goodFeaturesToTrack(
                image=gray,
                maxCorners=points_for_comp * 3,
                qualityLevel=0.005,
                minDistance=3,
                mask=comp_mask,
                blockSize=3,
                useHarrisDetector=False,
                k=0.04,
            )

            if disease_corners is not None and len(disease_corners) > 0:
                disease_corners = np.squeeze(disease_corners, axis=1)
                for x, y in disease_corners[:points_for_comp]:
                    cv2.circle(
                        vis, (int(x), int(y)), 4, COL_DISEASE, -1, lineType=cv2.LINE_AA
                    )
                    disease_placed += 1
                    if disease_placed >= disease_quota:
                        break
            else:
                cv2.circle(
                    vis, (int(cx), int(cy)), 4, COL_DISEASE, -1, lineType=cv2.LINE_AA
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
            logging.info("No disease landmarks placed - no brown regions detected")

    except Exception as e:
        logging.warning(f"Failed to detect disease landmarks: {e}")

    return disease_placed


def apply_landmarks_filter(
    rgb: np.ndarray,
    contour: Optional[np.ndarray],
    cfg: TransformConfig,
    make_mask_func: callable,
) -> np.ndarray:

    if contour is None:
        return draw_text(rgb, "Landmarks: no object")

    # Create enhanced leaf mask
    mask, _ = make_mask_func(rgb)
    if mask is not None:
        if mask.ndim == 2:
            leaf_mask = mask > 0
        else:
            leaf_mask = mask[..., 0] > 0

        enhanced_mask = _create_enhanced_mask(rgb, cfg, leaf_mask)

        # Find enhanced contour
        cnts, _ = cv2.findContours(
            enhanced_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if cnts:
            enhanced_contour = max(cnts, key=cv2.contourArea)
            contour = enhanced_contour

        mask_bool = enhanced_mask > 0
    else:
        mask_bool = None

    # Initialize visualization
    vis = rgb.copy()
    total = max(1, int(cfg.landmarks_count))

    # Calculate quotas
    border_quota = max(1, total // 3)
    vein_quota = max(1, total // 3)
    disease_quota = max(1, total - border_quota - vein_quota)

    # Place different types of landmarks
    border_placed = _place_border_landmarks(vis, contour, border_quota)
    vein_placed = _place_vein_landmarks(vis, rgb, mask_bool, vein_quota)
    disease_placed = _place_disease_landmarks(vis, rgb, cfg, mask_bool, disease_quota)

    # Log summary
    total_landmarks = border_placed + vein_placed + disease_placed
    logging.info(
        f"Landmarks summary: {border_placed} border + {vein_placed} veins + "
        f"{disease_placed} disease points = {total_landmarks} total landmarks"
    )

    return vis
