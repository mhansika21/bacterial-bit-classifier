from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PreprocessedImage:
    image: Image.Image
    tensor: np.ndarray
    focus_score: float


class MacenkoNormalizer:
    """Lightweight stain normalization approximation for demo/runtime use."""

    def normalize(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb).astype(np.float32)
        means = arr.reshape(-1, 3).mean(axis=0)
        stds = arr.reshape(-1, 3).std(axis=0) + 1e-6
        target_mean = np.array([180.0, 145.0, 175.0])
        target_std = np.array([45.0, 40.0, 45.0])
        normalized = (arr - means) / stds * target_std + target_mean
        return Image.fromarray(np.clip(normalized, 0, 255).astype(np.uint8))


class ImagePreprocessor:
    def __init__(self, image_size: int = 448) -> None:
        self.image_size = image_size
        self.stain_normalizer = MacenkoNormalizer()

    def from_bytes(self, payload: bytes) -> PreprocessedImage:
        try:
            image = Image.open(BytesIO(payload)).convert("RGB")
        except Exception as exc:
            raise ValueError("Uploaded file is not a valid image") from exc
        return self.preprocess(image)

    def preprocess(self, image: Image.Image) -> PreprocessedImage:
        normalized = self.stain_normalizer.normalize(image)
        resized = self._resize_with_padding(normalized)
        tensor = self._to_normalized_tensor(resized)
        focus = self.focus_score(resized)
        return PreprocessedImage(image=resized, tensor=tensor, focus_score=focus)

    def _resize_with_padding(self, image: Image.Image) -> Image.Image:
        image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        left = (self.image_size - image.width) // 2
        top = (self.image_size - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas

    def _to_normalized_tensor(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(arr, (2, 0, 1))

    def focus_score(self, image: Image.Image) -> float:
        gray = ImageOps.grayscale(image).filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(gray).astype(np.float32)
        return float(arr.var())
