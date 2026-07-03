"""Letterbox resizing: fit an image into a square canvas without stretching.

The image is scaled preserving aspect ratio and padded symmetrically with a
neutral grey, YOLO-style. Returns the canvas plus the scale/padding needed to
map coordinates back to the original image.
"""

import cv2
import numpy as np

DEFAULT_PAD_COLOR = (114, 114, 114)


def letterbox(image: np.ndarray, target_size: int = 224, pad_color=DEFAULT_PAD_COLOR):
    """Resize `image` into a target_size x target_size canvas, no stretching.

    Returns (canvas, scale, (pad_x, pad_y)).
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image passed to letterbox")

    scale = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((target_size, target_size, 3), pad_color, dtype=image.dtype)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def unletterbox_coords(coords_xy: np.ndarray, scale: float, padding) -> np.ndarray:
    """Map (x, y) points from letterboxed space back to the original image."""
    pad_x, pad_y = padding
    coords = np.asarray(coords_xy, dtype=np.float32).copy()
    coords[..., 0] = (coords[..., 0] - pad_x) / scale
    coords[..., 1] = (coords[..., 1] - pad_y) / scale
    return coords
