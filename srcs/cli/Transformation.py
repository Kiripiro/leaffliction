#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
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

# Lazy imports to avoid circular imports
import cv2
import numpy as np
import yaml

# Limit thread oversubscription before importing numpy/opencv
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


# === COMMON UTILITY FUNCTIONS ===


def is_image(path: Path) -> bool:
    """Check if path is a valid image file"""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)


def imwrite_bgr(path: Path, rgb_img) -> None:
    """Save RGB image to disk as BGR (OpenCV format)"""
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
    """Read image as RGB using PIL with EXIF orientation correction"""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        return np.array(im)


def draw_text(img, text: str, org=(10, 24)):
    """Draw text overlay on image"""
    if img is None:
        return None
    out = img.copy()
    cv2.putText(
        out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA
    )
    return out


def largest_contour(mask):
    """Find the largest contour in a binary mask"""
    cnts, _ = cv2.findContours(
        mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return None, None
    cnt = max(cnts, key=cv2.contourArea)
    return cnt, None


def contour_to_mask(shape_hw: Tuple[int, int], contour):
    """Convert contour to binary mask"""
    h, w = shape_hw
    out = np.zeros((h, w), dtype="uint8")
    cv2.drawContours(out, [contour], -1, color=255, thickness=-1)
    return out


def resample_contour(contour, n: int):
    """Resample contour to n equally spaced points"""
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
        from srcs.cli.filters.blur import apply_blur_filter

        return apply_blur_filter(rgb, self.cfg, self.make_mask)

    def make_mask(self, rgb):
        """Create robust leaf mask using various strategies"""
        from srcs.cli.filters.mask import make_mask

        return make_mask(rgb, self.cfg)

    def roi(self, rgb, contour):
        """Extract ROI (Region of Interest) from the image"""
        from srcs.cli.filters.roi import apply_roi_filter

        return apply_roi_filter(rgb, contour, self.cfg)

    def analyze(self, rgb, mask, contour):
        """Analyze leaf shape and create visual overlay"""
        from srcs.cli.filters.analyze import apply_analyze_filter

        return apply_analyze_filter(rgb, mask, contour, self.cfg)

    def pseudolandmarks(self, rgb, contour):
        """Place pseudolandmarks on leaf features"""
        from srcs.cli.filters.landmarks import apply_landmarks_filter

        return apply_landmarks_filter(rgb, contour, self.cfg, self.make_mask)

    def detect_brown_spots(self, rgb, mask):
        """Detect brown/diseased areas in leaf"""
        from srcs.cli.filters.brown import apply_brown_filter

        return apply_brown_filter(rgb, mask, self.cfg)

    def histogram_hsv(self, rgb):
        """Create HSV histogram visualization"""
        from srcs.cli.filters.hist import apply_histogram_filter

        return apply_histogram_filter(rgb, self.cfg)


# === OUTPUT AND PROCESSING FUNCTIONS ===


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
            from srcs.cli.filters.mask import apply_mask_filter

            masked_rgb = apply_mask_filter(rgb, params.cfg, pipe.make_mask)
            out = params.out_dir / names["Mask"]
            if params.overwrite or (not params.skip_existing or not out.exists()):
                imwrite_bgr(out, masked_rgb)
                saved.append(out)
        else:
            out = params.out_dir / names["Mask"]
            if params.overwrite or (not params.skip_existing or not out.exists()):
                imwrite_bgr(out, rgb)
                saved.append(out)

    # ROI
    if "ROI" in params.types:
        roi_img, roi_vis, _ = pipe.roi(rgb, contour)
        out = params.out_dir / names["ROI"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, roi_vis if roi_vis is not None else rgb)
            saved.append(out)

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
        ip = Path(args.image)
        if not is_image(ip):
            logging.error("Not a valid image: %s", ip)
            return

        # Extract image number from filename (e.g., "image (100).JPG" -> "100")
        import re

        match = re.search(r"image \((\d+)\)", ip.stem)
        if match:
            image_number = match.group(1)
        else:
            # Fallback to using the full stem
            image_number = ip.stem

        # Use artifacts/transformations/{imageNumber} as output directory
        repo_root = Path(__file__).resolve().parents[2]
        out_d = (
            Path(args.out_dir)
            if args.out_dir
            else repo_root / "artifacts" / "transformations" / image_number
        )
        ensure_dir(out_d)

        params = ProcessArgs(
            img_path=ip,
            out_dir=out_d,
            types=types,
            cfg=cfg,
            skip_existing=args.skip_existing,
            overwrite=args.overwrite,
        )
        saved = process_single_image(params)
        print(f"Saved {len(saved)} outputs to {out_d}")
        for s in saved:
            print(f"  - {s}")
        return

    # Folder mode
    if args.src and args.dst:
        src = Path(args.src)
        dst = Path(args.dst)
        if not src.exists():
            logging.error("Source directory does not exist: %s", src)
            return

        ensure_dir(dst)
        imgs = list(iter_images_in_dir(src))
        if not imgs:
            logging.warning("No images found in %s", src)
            return

        logging.info("Found %d images in %s", len(imgs), src)

        # Multiprocessing setup
        n_proc = (
            args.workers if args.workers > 0 else min(8, max(1, mp.cpu_count() // 2))
        )
        logging.info("Using %d processes", n_proc)

        if n_proc == 1:
            # Single-threaded
            for img_path in imgs:
                params = ProcessArgs(
                    img_path=img_path,
                    out_dir=dst,
                    types=types,
                    cfg=cfg,
                    skip_existing=args.skip_existing,
                    overwrite=args.overwrite,
                )
                process_single_image(params)
        else:
            # Multi-threaded
            with mp.Pool(processes=n_proc) as pool:
                tasks = [
                    (str(ip), str(dst), types, cfg, args.skip_existing, args.overwrite)
                    for ip in imgs
                ]
                results = pool.starmap(_worker_run, tasks)
                total_saved = sum(len(saved) for _, saved in results)
                logging.info(
                    "Processed %d images, saved %d outputs", len(imgs), total_saved
                )
        return

    # No valid mode
    logging.error("Must specify either single image or --src/--dst for folder mode")


if __name__ == "__main__":
    main()
