from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from srcs.utils.common import get_logger

logger = get_logger(__name__)


class PredictionVisualizer:

    def __init__(self, font_size: int = 20):
        self.font_size = font_size

    def create_montage(self, result: Dict[str, Any], output_path: str | Path) -> None:
        output_path = Path(output_path)

        original = result["original_array"]
        processed = result["processed_array"]

        processed_display = (processed * 255).astype(np.uint8)

        original_image = Image.fromarray(original)
        processed_image = Image.fromarray(processed_display)

        display_size = (224, 224)
        original_image = original_image.resize(display_size, Image.Resampling.LANCZOS)
        processed_image = processed_image.resize(display_size, Image.Resampling.LANCZOS)

        montage_width = display_size[0] * 2 + 20
        text_height = 60
        montage_height = display_size[1] + text_height

        montage = Image.new("RGB", (montage_width, montage_height), "white")

        montage.paste(original_image, (0, 0))
        montage.paste(processed_image, (display_size[0] + 20, 0))

        draw = ImageDraw.Draw(montage)

        try:
            font = ImageFont.truetype("arial.ttf", self.font_size)
        except OSError:
            font = ImageFont.load_default()

        prediction_text = (
            f"Prediction: {result['top_prediction']} " f"({result['confidence']:.1%})"
        )

        text_bbox = draw.textbbox((0, 0), prediction_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (montage_width - text_width) // 2
        text_y = display_size[1] + 20

        draw.text((text_x, text_y), prediction_text, font=font, fill="black")

        draw.text((10, display_size[1] + 5), "Original", font=font, fill="gray")
        draw.text(
            (display_size[0] + 30, display_size[1] + 5),
            "Processed",
            font=font,
            fill="gray",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        montage.save(output_path)

        logger.info(f"Montage saved to {output_path}")
