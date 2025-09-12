from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from srcs.predict.image_processor import ImageProcessor
from srcs.predict.model_loader import ModelLoader
from srcs.utils.common import get_logger

logger = get_logger(__name__)


class Predictor:

    def __init__(self, learnings_dir: str | Path):
        self.learnings_dir = Path(learnings_dir)
        self.model_loader = None
        self.image_processor = None
        self._initialized = False

    def load(self):
        logger.info(f"Initializing Predictor with learnings from: {self.learnings_dir}")

        self.model_loader = ModelLoader(self.learnings_dir)
        self.model_loader.load()

        self.image_processor = ImageProcessor(img_size=self.model_loader.img_size)

        self._initialized = True
        logger.info("Predictor initialized successfully")

    def predict_single(self, image_path: str | Path) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Predictor not initialized. Call load() first.")

        image_path = Path(image_path)
        logger.info(f"Predicting image: {image_path}")

        original_array, processed_array = self.image_processor.process_image(image_path)
        probabilities = self.model_loader.model.predict(processed_array, verbose=0)[0]

        top_idx = int(np.argmax(probabilities))
        top_class = self.model_loader.labels[top_idx]
        confidence = float(probabilities[top_idx])

        all_probabilities = {
            self.model_loader.labels[i]: float(probabilities[i])
            for i in range(len(self.model_loader.labels))
        }

        logger.info(f"Prediction: {top_class} ({confidence:.2f})")

        return {
            "image_path": image_path,
            "top_prediction": top_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "original_array": original_array,
            "processed_array": processed_array.squeeze(0),
        }

    def predict_batch(self, image_paths: List[str | Path]) -> List[Dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("Predictor not initialized. Call load() first.")

        paths = [Path(p) for p in image_paths]
        logger.info(f"Predicting batch of {len(paths)} images")

        processed_data = []
        original_arrays = []

        for img_path in paths:
            try:
                original, processed = self.image_processor.process_image(img_path)
                processed_data.append(processed[0])
                original_arrays.append((img_path, original, processed[0]))
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
                continue

        if not processed_data:
            logger.warning("No valid images to predict.")
            return []

        batch_array = np.stack(processed_data)
        logger.debug(f"Batch array shape: {batch_array.shape}")

        batch_probabilities = self.model_loader.model.predict(batch_array, verbose=0)

        results = []
        for i, (img_path, original, processed) in enumerate(original_arrays):
            probabilities = batch_probabilities[i]

            top_idx = int(np.argmax(probabilities))
            top_class = self.model_loader.labels[top_idx]
            confidence = float(probabilities[top_idx])

            all_probabilities = {
                self.model_loader.labels[j]: float(probabilities[j])
                for j in range(len(self.model_loader.labels))
            }

            results.append(
                {
                    "image_path": img_path,
                    "top_prediction": top_class,
                    "confidence": confidence,
                    "all_probabilities": all_probabilities,
                    "original_array": original,
                    "processed_array": processed,
                }
            )

            logger.debug(
                f"Image: {img_path}, Prediction: {top_class} ({confidence:.2f})"
            )

        logger.info("Batch prediction completed")
        return results
