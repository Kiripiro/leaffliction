from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    from ..Transformation import TransformConfig, draw_text
except ImportError:
    import sys
    from pathlib import Path

    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from Transformation import TransformConfig, draw_text


def apply_analyze_filter(
    rgb: np.ndarray,
    mask: Optional[np.ndarray],
    contour: Optional[np.ndarray],
    cfg: TransformConfig,
) -> np.ndarray:
    from plantcv import plantcv as pcv  # type: ignore

    if contour is None or mask is None:
        return draw_text(rgb, "Analyze: no object")

    # Use PlantCV to compute shape metrics; also build a visual overlay
    try:
        _ = pcv.analyze_object(img=rgb, obj=contour, mask=mask)  # populates pcv.outputs
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

    return overlay
