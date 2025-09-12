#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from srcs.utils.common import setup_logging
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from srcs.utils.common import setup_logging

import random
import re

import cv2
import numpy as np
import yaml

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
    gaussian_sigma: float
    hsv_channel_for_mask: str
    fill_size: int
    morph_kernel: int
    landmarks_count: int
    roi_size: Tuple[int, int]
    mask_strategy: str
    bg_bias: Optional[str]
    grabcut_refine: bool
    green_hue_range: Tuple[int, int]
    min_object_area_ratio: float
    max_object_area_ratio: float
    mask_upscale_factor: float
    mask_upscale_long_side: int
    shadow_suppression: bool
    shadow_s_max: int
    shadow_v_method: str
    shadow_v_percentile: int
    shadow_morphology_kernel: int
    brown_hue_range: Tuple[int, int]
    brown_s_min: int
    brown_v_max: int
    brown_min_area_px: int
    brown_morph_kernel: int
    use_lab_brown: bool
    lab_b_min: int
    lab_a_min: int
    debug_shadow_visualization: bool


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
        logging.error("No configuration file path provided")
        sys.exit(1)
    if not path.exists():
        logging.error("Configuration file not found: %s", path)
        sys.exit(1)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Validate that all required fields are present
        required_fields = [
            "gaussian_sigma",
            "hsv_channel_for_mask",
            "fill_size",
            "morph_kernel",
            "landmarks_count",
            "roi_size",
            "mask_strategy",
            "bg_bias",
            "grabcut_refine",
            "green_hue_range",
            "min_object_area_ratio",
            "max_object_area_ratio",
            "mask_upscale_factor",
            "mask_upscale_long_side",
            "shadow_suppression",
            "shadow_s_max",
            "shadow_v_method",
            "shadow_v_percentile",
            "shadow_morphology_kernel",
            "brown_hue_range",
            "brown_s_min",
            "brown_v_max",
            "brown_min_area_px",
            "brown_morph_kernel",
            "use_lab_brown",
            "lab_b_min",
            "lab_a_min",
            "debug_shadow_visualization",
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logging.error("Missing required configuration fields: %s", missing_fields)
            sys.exit(1)

        return TransformConfig(
            gaussian_sigma=float(data["gaussian_sigma"]),
            hsv_channel_for_mask=str(data["hsv_channel_for_mask"]),
            fill_size=int(data["fill_size"]),
            morph_kernel=int(data["morph_kernel"]),
            landmarks_count=int(data["landmarks_count"]),
            roi_size=tuple(data["roi_size"]),  # type: ignore
            mask_strategy=str(data["mask_strategy"]),
            bg_bias=data["bg_bias"],
            grabcut_refine=bool(data["grabcut_refine"]),
            green_hue_range=tuple(data["green_hue_range"]),  # type: ignore
            min_object_area_ratio=float(data["min_object_area_ratio"]),
            max_object_area_ratio=float(data["max_object_area_ratio"]),
            mask_upscale_factor=float(data["mask_upscale_factor"]),
            mask_upscale_long_side=int(data["mask_upscale_long_side"]),
            shadow_suppression=bool(data["shadow_suppression"]),
            shadow_s_max=int(data["shadow_s_max"]),
            shadow_v_method=str(data["shadow_v_method"]),
            shadow_v_percentile=int(data["shadow_v_percentile"]),
            shadow_morphology_kernel=int(data["shadow_morphology_kernel"]),
            brown_hue_range=tuple(data["brown_hue_range"]),  # type: ignore
            brown_s_min=int(data["brown_s_min"]),
            brown_v_max=int(data["brown_v_max"]),
            brown_min_area_px=int(data["brown_min_area_px"]),
            brown_morph_kernel=int(data["brown_morph_kernel"]),
            use_lab_brown=bool(data["use_lab_brown"]),
            lab_b_min=int(data["lab_b_min"]),
            lab_a_min=int(data["lab_a_min"]),
            debug_shadow_visualization=bool(data["debug_shadow_visualization"]),
        )
    except Exception as exc:
        logging.error("Failed to read configuration file (%s)", exc)
        sys.exit(1)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def imwrite_bgr(path: Path, rgb_img) -> None:
    if rgb_img is None:
        return
    arr = np.asarray(rgb_img)
    if arr.ndim == 2:
        bgr = arr
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), bgr)


def create_mosaic(original_rgb, filter_results, image_number):
    target_size = 300
    original_resized = cv2.resize(original_rgb, (target_size, target_size))
    images = [("Original", original_resized)]
    for filter_name, filter_image in filter_results.items():
        if filter_image is not None:
            # S'assurer que l'image est en RGB
            if filter_image.ndim == 2:
                filter_image = cv2.cvtColor(filter_image, cv2.COLOR_GRAY2RGB)
            elif filter_image.shape[2] == 1:
                filter_image = cv2.cvtColor(filter_image, cv2.COLOR_GRAY2RGB)

            resized_filter = cv2.resize(filter_image, (target_size, target_size))
            images.append((filter_name, resized_filter))

    cols = 3
    rows = (len(images) + cols - 1) // cols

    mosaic_width = cols * target_size
    mosaic_height = rows * target_size
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

    for idx, (title, img) in enumerate(images):
        row = idx // cols
        col = idx % cols

        y_start = row * target_size
        x_start = col * target_size

        # Ajouter l'image
        mosaic[y_start : y_start + target_size, x_start : x_start + target_size] = img

        # Ajouter le titre au-dessus de l'image (sur un fond semi-transparent)
        text_height = 25
        overlay = mosaic.copy()
        cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_start + target_size, y_start + text_height),
            (0, 0, 0),
            -1,
        )
        mosaic = cv2.addWeighted(mosaic, 0.7, overlay, 0.3, 0)

        # Texte en blanc
        cv2.putText(
            mosaic,
            title,
            (x_start + 10, y_start + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return mosaic


def pil_read_rgb(path: Path):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        return np.array(im)


def draw_text(img, text: str, org=(10, 24)):
    if img is None:
        return None
    out = img.copy()
    cv2.putText(
        out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA
    )
    return out


def largest_contour(mask):
    cnts, _ = cv2.findContours(
        mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return None, None
    cnt = max(cnts, key=cv2.contourArea)
    return cnt, None


def contour_to_mask(shape_hw: Tuple[int, int], contour):
    h, w = shape_hw
    out = np.zeros((h, w), dtype="uint8")
    cv2.drawContours(out, [contour], -1, color=255, thickness=-1)
    return out


def resample_contour(contour, n: int):
    pts = contour[:, 0, :].astype(float)
    if not (pts[0] == pts[-1]).all():
        pts = np.vstack([pts, pts[0]])
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
        from plantcv import plantcv as pcv

        pcv.params.debug = None

    def blur(self, rgb):
        from srcs.transform.filters.blur import apply_blur_filter

        return apply_blur_filter(rgb, self.cfg, self.make_mask)

    def make_mask(self, rgb):
        from srcs.transform.filters.mask import make_mask

        return make_mask(rgb, self.cfg)

    def create_masked_rgb(self, rgb, mask):
        """Create masked RGB image using PlantCV-style masking."""
        if mask is None:
            return rgb

        from srcs.utils.mask_utils import apply_mask

        return apply_mask(rgb, mask, mask_color="white")

    def apply_mask(self, rgb, mask, mask_color="white"):
        """Apply mask to RGB image following PlantCV logic.

        Args:
            rgb: RGB image array
            mask: Binary mask array
            mask_color: 'white' or 'black' background color

        Returns:
            Masked image array
        """
        from srcs.utils.mask_utils import apply_mask

        return apply_mask(rgb, mask, mask_color)

    def roi(self, rgb, contour):
        from srcs.transform.filters.roi import apply_roi_filter

        return apply_roi_filter(rgb, contour, self.cfg)

    def analyze(self, rgb, mask, contour):
        from srcs.transform.filters.analyze import apply_analyze_filter

        return apply_analyze_filter(rgb, mask, contour, self.cfg)

    def pseudolandmarks(self, rgb, contour):
        from srcs.transform.filters.landmarks import apply_landmarks_filter

        return apply_landmarks_filter(rgb, contour, self.cfg, self.make_mask)

    def detect_brown_spots(self, rgb, mask):
        from srcs.transform.filters.brown import apply_brown_filter

        return apply_brown_filter(rgb, mask, self.cfg)

    def histogram_hsv(self, rgb):
        from srcs.transform.filters.hist import apply_histogram_filter

        return apply_histogram_filter(rgb, self.cfg)


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
    try:
        rgb = pil_read_rgb(params.img_path)
    except Exception as exc:
        logging.error("Failed to read %s (%s)", params.img_path, exc)
        return []

    pipe = TransformPipeline(params.cfg)
    saved: List[Path] = []

    filter_results = {}

    names = output_names(params.img_path.stem)

    mask_img, contour = (None, None)
    masked_rgb = rgb  # Default to original image

    if (
        "Mask" in params.types
        or "ROI" in params.types
        or "Analyze" in params.types
        or "Landmarks" in params.types
        or "Brown" in params.types
        or "Blur" in params.types
    ):
        mask_img, contour = pipe.make_mask(rgb)
        # Create masked RGB for all other filters to use
        if mask_img is not None:
            masked_rgb = pipe.create_masked_rgb(rgb, mask_img)

    if "Mask" in params.types:
        if mask_img is not None:
            # Apply mask to RGB for visualization
            from srcs.transform.filters.mask import apply_mask_filter

            mask_vis = apply_mask_filter(rgb, params.cfg, pipe.make_mask)
            filter_results["Mask"] = mask_vis
            out = params.out_dir / names["Mask"]
            if params.overwrite or (not params.skip_existing or not out.exists()):
                imwrite_bgr(out, mask_vis)
                saved.append(out)
        else:
            filter_results["Mask"] = rgb
            out = params.out_dir / names["Mask"]
            if params.overwrite or (not params.skip_existing or not out.exists()):
                imwrite_bgr(out, rgb)
                saved.append(out)

    if "Blur" in params.types:
        blur_img = pipe.blur(masked_rgb)
        filter_results["Blur"] = blur_img
        out = params.out_dir / names["Blur"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, blur_img)
            saved.append(out)

    if "ROI" in params.types:
        roi_img, roi_vis, _ = pipe.roi(masked_rgb, contour)
        filter_results["ROI"] = roi_vis if roi_vis is not None else masked_rgb
        out = params.out_dir / names["ROI"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, roi_vis if roi_vis is not None else masked_rgb)
            saved.append(out)

    if "Analyze" in params.types:
        analyze_img = pipe.analyze(masked_rgb, mask_img, contour)
        filter_results["Analyze"] = analyze_img
        out = params.out_dir / names["Analyze"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, analyze_img)
            saved.append(out)

    if "Landmarks" in params.types:
        lm_img = pipe.pseudolandmarks(masked_rgb, contour)
        filter_results["Landmarks"] = lm_img
        out = params.out_dir / names["Landmarks"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, lm_img)
            saved.append(out)

    if "Hist" in params.types:
        hist_img = pipe.histogram_hsv(masked_rgb)
        filter_results["Hist"] = hist_img
        out = params.out_dir / names["Hist"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, hist_img)
            saved.append(out)

    if "Brown" in params.types:
        brown_img, brown_percentage, brown_count = pipe.detect_brown_spots(
            masked_rgb, mask_img
        )
        filter_results["Brown"] = brown_img
        out = params.out_dir / names["Brown"]
        if params.overwrite or (not params.skip_existing or not out.exists()):
            imwrite_bgr(out, brown_img)
            saved.append(out)

    if filter_results:
        # Extraire le numéro d'image du nom de fichier

        match = re.search(r"image \((\d+)\)", params.img_path.stem)
        image_number = match.group(1) if match else params.img_path.stem

        # Créer la mosaïque
        mosaic = create_mosaic(rgb, filter_results, image_number)

        # Sauvegarder la mosaïque
        mosaic_path = params.out_dir / f"image{image_number}_mosaic.jpg"
        imwrite_bgr(mosaic_path, mosaic)
        saved.append(mosaic_path)
        print(f"Mosaïque créée : {mosaic_path}")

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
        "--config",
        default="srcs/transform/config.yaml",
        help="YAML config path (optional)",
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

    if args.image and not args.src and not args.dst:
        ip = Path(args.image)
        if not is_image(ip):
            logging.error("Not a valid image: %s", ip)
            return

        # Extract image number from filename (e.g., "image (100).JPG" -> "100")

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

    logging.error("Must specify either single image or --src/--dst for folder mode")


# === TRAINING-SPECIFIC FUNCTIONS ===


def transform_single_image_for_training(  # noqa: C901
    img_path: Path,
    item: Any,  # ManifestItem
    img_size: int,
    cfg: Optional[TransformConfig] = None,
    transform_types: Optional[Tuple[str, ...]] = None,
    apply_augmentation: bool = True,
    extern_cache: Optional[Dict[Any, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a single image for model training/inference.

    Args:
        img_path: Path to the image file
        item: ManifestItem with metadata (unused for interface compatibility)
        img_size: Target size for model input (e.g., 224, 256, etc.)
        cfg: Configuration for transformations (if None, loads default)
        transform_types: Which transformations to apply
            (default: optimized for training)
        apply_augmentation: Whether to apply data augmentation
        extern_cache: External cache dict to reuse results across calls
        logger: Optional logger for messages

    Returns:
        Tuple of (original_uint8, transformed_float32) both resized to img_size
    """
    log = logger or logging.getLogger(__name__)

    if cfg is None:
        # Load default config if not provided
        try:
            repo_root = Path(__file__).resolve().parents[2]
            default_config_path = repo_root / "transform" / "config.yaml"
            cfg = load_config(default_config_path)
        except Exception as exc:
            log.warning(
                "Failed to load default config, using minimal transformations: %s",
                exc,
            )
            # Fallback: simple resize without transformation
            from keras.utils import img_to_array, load_img

            resized = img_to_array(
                load_img(img_path, target_size=(img_size, img_size), color_mode="rgb")
            )
            x_float32 = (resized / 255.0).astype("float32")
            orig_uint8 = np.clip(resized, 0, 255).astype("uint8")
            return orig_uint8, x_float32

    if transform_types is None:
        transform_types = DEFAULT_TYPES

    canonical: List[str] = []
    seen: set[str] = set()
    for t in transform_types:
        key = str(t).strip().lower()
        if key in CANONICAL_TYPES:
            name = CANONICAL_TYPES[key]
        else:
            name = t
        if name not in seen:
            canonical.append(name)
            seen.add(name)
        else:
            # Prevent duplicate filter application on the same image
            log.info("Duplicate transform '%s' ignored for %s", name, img_path)
    transform_types = tuple(canonical)

    cache_key = (str(img_path), int(img_size), tuple(transform_types))

    if extern_cache is not None and cache_key in extern_cache:
        cached = extern_cache[cache_key]
        return cached[0], cached[1]

    try:
        # Load and cache original RGB to avoid re-reading the same image
        rgb_cache_key = ("__rgb__", str(img_path))
        if extern_cache is not None and rgb_cache_key in extern_cache:
            rgb = extern_cache[rgb_cache_key]
        else:
            rgb = pil_read_rgb(img_path)
            if extern_cache is not None:
                extern_cache[rgb_cache_key] = rgb

        # Keep the original resized (independent of transform list)
        orig_resize_key = ("__orig__", str(img_path), int(img_size))
        if extern_cache is not None and orig_resize_key in extern_cache:
            orig_uint8 = extern_cache[orig_resize_key]
        else:
            orig_resized = cv2.resize(
                rgb, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4
            )
            orig_uint8 = np.clip(orig_resized, 0, 255).astype("uint8")
            if extern_cache is not None:
                extern_cache[orig_resize_key] = orig_uint8

        # Apply transformations in sequence
        pipe = TransformPipeline(cfg)
        transformed_img = rgb.copy()
        original_rgb = rgb  # preserve original for detectors like Brown

        # Reuse intermediate computations in a per-call dict to avoid redundant work
        step_cache: Dict[str, Any] = {}

        mask_img, contour = None, None
        mask_needed = ["Mask", "Brown", "ROI", "Analyze", "Landmarks"]
        if any(t in transform_types for t in mask_needed):
            # Try cache first (compute mask on original rgb)
            mask_key = ("__mask__", str(img_path))
            contour_key = ("__contour__", str(img_path))
            if (
                extern_cache is not None
                and mask_key in extern_cache
                and contour_key in extern_cache
            ):
                mask_img = extern_cache[mask_key]
                contour = extern_cache[contour_key]
            else:
                mask_img, contour = pipe.make_mask(rgb)
                if extern_cache is not None:
                    extern_cache[mask_key] = mask_img
                    extern_cache[contour_key] = contour
            step_cache["Mask:binary"] = mask_img
            step_cache["Mask:contour"] = contour

        if "Blur" in transform_types:
            blur_img = pipe.blur(original_rgb)
            transformed_img = blur_img
            step_cache["Blur"] = blur_img

        if "Mask" in transform_types and mask_img is not None:
            from srcs.transform.filters.mask import apply_mask_filter

            masked_rgb = apply_mask_filter(original_rgb, cfg, pipe.make_mask)
            transformed_img = masked_rgb
            step_cache["Mask:applied"] = masked_rgb

        if "ROI" in transform_types and contour is not None:
            _roi_img, roi_vis, _ = pipe.roi(original_rgb, contour)
            if roi_vis is not None:
                transformed_img = roi_vis
                step_cache["ROI"] = transformed_img

        if "Analyze" in transform_types:
            analyze_img = pipe.analyze(original_rgb, mask_img, contour)
            if analyze_img is not None:
                transformed_img = analyze_img
                step_cache["Analyze"] = transformed_img

        if "Landmarks" in transform_types and contour is not None:
            lm_img = pipe.pseudolandmarks(original_rgb, contour)
            if lm_img is not None:
                transformed_img = lm_img
                step_cache["Landmarks"] = transformed_img

        if "Hist" in transform_types:
            hist_img = pipe.histogram_hsv(original_rgb)
            if hist_img is not None:
                transformed_img = hist_img
                step_cache["Hist"] = transformed_img

        # Apply Brown last to mirror the CLI processing order
        if "Brown" in transform_types and mask_img is not None:
            # Use original RGB for brown detection (parity with CLI)
            brown_img, _, _ = pipe.detect_brown_spots(original_rgb, mask_img)
            if brown_img is not None:
                transformed_img = brown_img
                step_cache["Brown"] = transformed_img

        # Optionally save all transformation steps for a random subset of images
        # Controlled via environment variables to avoid changing call sites
        #   LEAF_SAVE_TRANSFORMS: "1"/"true" to enable (default off)
        #   LEAF_SAVE_TRANSFORMS_RATE: sampling probability in [0,1] (default 0.05)
        #   LEAF_SAVE_TRANSFORMS_DIR: output directory
        #       (default artifacts/sequence_previews)
        try:

            save_flag = os.environ.get("LEAF_SAVE_TRANSFORMS", "0").lower() in {
                "1",
                "true",
                "yes",
            }
            save_rate = float(os.environ.get("LEAF_SAVE_TRANSFORMS_RATE", "0.05"))
            if save_flag and random.random() < max(0.0, min(1.0, save_rate)):
                repo_root = Path(__file__).resolve().parents[2]
                base_dir = os.environ.get(
                    "LEAF_SAVE_TRANSFORMS_DIR",
                    str(repo_root / "artifacts" / "sequence_previews"),
                )
                # Try to preserve a bit of folder structure for readability
                stem = img_path.stem
                parent = img_path.parent.name
                out_dir = Path(base_dir) / parent / stem
                ensure_dir(out_dir)

                # Save original and each step (if present)
                # Helper to save safely
                def _save(name: str, arr: Any) -> None:
                    if arr is None:
                        return
                    if isinstance(arr, np.ndarray):
                        imwrite_bgr(out_dir / f"T_{name}.jpg", arr)
                    else:
                        try:
                            imwrite_bgr(out_dir / f"T_{name}.jpg", np.asarray(arr))
                        except Exception:
                            pass

                _save("00_Input", original_rgb)
                if "Blur" in step_cache:
                    _save("10_Blur", step_cache.get("Blur"))
                if step_cache.get("Mask:binary") is not None:
                    _save("20_MaskBinary", step_cache.get("Mask:binary"))
                if step_cache.get("Mask:applied") is not None:
                    _save("21_MaskApplied", step_cache.get("Mask:applied"))
                if step_cache.get("ROI") is not None:
                    _save("30_ROI", step_cache.get("ROI"))
                if step_cache.get("Analyze") is not None:
                    _save("40_Analyze", step_cache.get("Analyze"))
                if step_cache.get("Landmarks") is not None:
                    _save("50_Landmarks", step_cache.get("Landmarks"))
                if step_cache.get("Hist") is not None:
                    _save("60_Hist", step_cache.get("Hist"))
                if step_cache.get("Brown") is not None:
                    _save("70_Brown", step_cache.get("Brown"))
                # Final image fed to the model (after resize happens below)
                _save("99_FinalPreResize", transformed_img)
        except Exception as dbg_exc:
            log.debug("Transform preview save skipped: %s", dbg_exc)

        # Resize and normalize for the model
        transformed_resized = cv2.resize(
            transformed_img, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4
        )

        if apply_augmentation:
            transformed_resized = _apply_light_augmentation(transformed_resized)

        x_float32 = (transformed_resized / 255.0).astype("float32")

        # Save in external cache if provided
        if extern_cache is not None:
            extern_cache[cache_key] = (orig_uint8, x_float32)

        return orig_uint8, x_float32

    except Exception as exc:
        log.error(
            "Failed to transform %s (%s), falling back to simple resize",
            img_path,
            exc,
        )
        # Fallback en cas d'erreur
        from keras.utils import img_to_array, load_img

        try:
            resized = img_to_array(
                load_img(img_path, target_size=(img_size, img_size), color_mode="rgb")
            )
            x_float32 = (resized / 255.0).astype("float32")
            orig_uint8 = np.clip(resized, 0, 255).astype("uint8")
            if extern_cache is not None:
                extern_cache[cache_key] = (orig_uint8, x_float32)
            return orig_uint8, x_float32
        except Exception as fallback_exc:
            log.error("Complete failure to load %s (%s)", img_path, fallback_exc)
            # Retourner des images noires en dernier recours
            black_img = np.zeros((img_size, img_size, 3), dtype="uint8")
            black_float = np.zeros((img_size, img_size, 3), dtype="float32")
            if extern_cache is not None:
                extern_cache[cache_key] = (black_img, black_float)
            return black_img, black_float


def _apply_light_augmentation(img: np.ndarray) -> np.ndarray:
    """
    Appliquer une augmentation légère pour l'entraînement.

    Args:
        img: Image RGB en uint8

    Returns:
        Image augmentée
    """

    if random.random() < 0.3:  # 30% de chance
        # Ajustement léger de la luminosité
        brightness_factor = random.uniform(0.8, 1.2)
        img = np.clip(img * brightness_factor, 0, 255).astype("uint8")

    if random.random() < 0.2:  # 20% de chance
        # Ajustement léger du contraste
        contrast_factor = random.uniform(0.8, 1.2)
        img = np.clip((img - 127.5) * contrast_factor + 127.5, 0, 255).astype("uint8")

    return img


def create_transform_function(
    config_path: Optional[str] = None,
    transform_types: Optional[Tuple[str, ...]] = None,
    apply_augmentation: bool = True,
):
    """
    Créer une fonction de transformation compatible avec ManifestSequence.

    Args:
        config_path: Chemin vers le fichier de configuration YAML
        transform_types: Types de transformations à appliquer
        apply_augmentation: Activer l'augmentation de données

    Returns:
        Fonction de transformation utilisable avec ManifestSequence
    """
    cfg = None
    if config_path:
        cfg = load_config(Path(config_path))

    _internal_cache: Dict[Any, Any] = {}

    def transform_fn(
        img_path: Path,
        item: Any,
        img_size: int,
        transformations: Optional[Tuple[str, ...]] = None,
        cache: Optional[Dict[Any, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Prefer caller-provided list, else the default configured at creation
        chosen = transformations if transformations is not None else transform_types
        # Prefer external cache if provided, else internal shared cache
        cache_dict = cache if cache is not None else _internal_cache
        return transform_single_image_for_training(
            img_path=img_path,
            item=item,
            img_size=img_size,
            cfg=cfg,
            transform_types=chosen,
            apply_augmentation=apply_augmentation,
            extern_cache=cache_dict,
            logger=logger,
        )

    return transform_fn


if __name__ == "__main__":
    main()
