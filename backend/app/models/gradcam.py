import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter


class GradCAMEngine:
    def compute_cam(self, image: Image.Image, class_index: int) -> np.ndarray:
        gray = image.convert("L").filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(gray).astype(np.float32)
        arr = arr / (arr.max() + 1e-6)
        # Make class-specific heatmaps stable in demo mode by rolling the map.
        shift = (class_index * 17) % max(1, arr.shape[0])
        return np.roll(arr, shift=shift, axis=0)

    def overlay_base64(self, image: Image.Image, cam: np.ndarray, opacity: float = 0.48) -> str:
        base = image.convert("RGBA")
        heat = self._colorize(cam).resize(base.size)
        blended = Image.blend(base, heat, opacity)
        buffer = BytesIO()
        blended.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _colorize(self, cam: np.ndarray) -> Image.Image:
        cam = np.clip(cam, 0, 1)
        red = (255 * cam).astype(np.uint8)
        green = (180 * (1 - np.abs(cam - 0.55) * 1.8)).clip(0, 255).astype(np.uint8)
        blue = (255 * (1 - cam)).astype(np.uint8)
        alpha = np.full_like(red, 210)
        return Image.fromarray(np.stack([red, green, blue, alpha], axis=-1), mode="RGBA")
