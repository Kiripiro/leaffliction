from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from srcs.utils.common import get_logger

logger = get_logger(__name__)


class ModelLoader:

    def __init__(self, learnings_dir: str | Path):
        self.learnings_dir = Path(learnings_dir)
        self.meta_data: Dict[str, Any] = {}
        self.model = None

    def load(self):
        self._load_meta_data()
        self._load_model()
        logger.info("Model and metadata loaded successfully")

    def _load_meta_data(self):
        meta_path = self.learnings_dir / "meta.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta_data = json.load(f)

        logger.info(f"Loaded metadata: {len(self.meta_data['labels'])} classes")

    def _load_model(self):
        model_file = self.meta_data.get("model_file")
        if not model_file:
            raise ValueError("Model file not specified in metadata")

        model_path = Path(model_file)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        import keras

        self.model = keras.models.load_model(str(model_path))
        logger.info(f"Model loaded from {model_path}")

    @property
    def labels(self) -> List[str]:
        return self.meta_data.get("labels", [])

    @property
    def img_size(self) -> int:
        return self.meta_data.get("data", {}).get("img_size", 224)

    @property
    def num_classes(self) -> int:
        return len(self.labels)
