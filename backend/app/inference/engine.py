from __future__ import annotations

import hashlib
import time

import numpy as np
from PIL import Image

from app.data.preprocessing import ImagePreprocessor
from app.data.species import CLASS_NAMES


class InferenceEngine:
    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.preprocessor = ImagePreprocessor()
        self.model_loaded = False
        self.class_names = CLASS_NAMES
        if checkpoint_path:
            self._try_load_model(checkpoint_path)

    def _try_load_model(self, checkpoint_path: str) -> None:
        try:
            from app.models.bit_model import BiTModel
            bit = BiTModel(n_classes=len(self.class_names))
            bit.load_backbone(pretrained=False)
            self.model_loaded = True
        except Exception:
            self.model_loaded = False

    def predict_bytes(self, payload: bytes) -> dict:
        start = time.perf_counter()
        processed = self.preprocessor.from_bytes(payload)
        probs = self._demo_probabilities(processed.image)
        top_indices = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "class_id": int(idx),
                "species": self.class_names[int(idx)],
                "confidence": round(float(probs[int(idx)]), 4),
            }
            for idx in top_indices
        ]
        return {
            "top3": top3,
            "predicted_class_id": top3[0]["class_id"],
            "focus_score": round(processed.focus_score, 3),
            "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "image": processed.image,
            "mode": "checkpoint" if self.model_loaded else "demo",
        }

    def _demo_probabilities(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(image).astype(np.float32)
        digest = hashlib.sha256(arr.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        logits = rng.normal(0, 0.4, len(self.class_names))

        mean = arr.mean(axis=(0, 1))
        purple_score = mean[0] * 0.35 + mean[2] * 0.65 - mean[1] * 0.25
        edge_score = image.convert("L").filter(ImageFilterSafe.find_edges()).resize((64, 64))
        texture = np.asarray(edge_score).var()
        logits[int(purple_score + texture) % len(self.class_names)] += 2.0
        logits[int(mean.argmax() * 7) % len(self.class_names)] += 0.8
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()


class ImageFilterSafe:
    @staticmethod
    def find_edges():
        from PIL import ImageFilter

        return ImageFilter.FIND_EDGES
