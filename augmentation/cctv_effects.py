"""CCTV-specific effects that albumentations does not cover:

- fisheye/barrel lens distortion
- synthetic surveillance timestamp overlay (camera id + date + time + REC dot)
- scanlines and vignette (mild, optional)

All effects are driven by a caller-provided random.Random instance so the
augmentation stage is reproducible.
"""

import random

import cv2
import numpy as np


def apply_fisheye(image: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Apply a radial lens distortion of the given strength (0.05-0.5)."""
    h, w = image.shape[:2]
    focal = float(w)
    camera_matrix = np.array(
        [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.array([strength, strength * 0.1, 0, 0], dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
    )
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def add_timestamp_overlay(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Burn a synthetic 'CAM NN  YYYY-MM-DD HH:MM:SS' overlay into a corner.

    The timestamp is synthesized from the rng (not the wall clock) so the
    output is reproducible for a fixed seed.
    """
    img = image.copy()
    h, w = img.shape[:2]

    cam_id = rng.randint(1, 16)
    year = rng.randint(2024, 2026)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour, minute, second = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
    text = f"CAM {cam_id:02d}  {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"

    scale = max(0.4, w / 900.0)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

    margin = int(0.02 * min(h, w)) + 4
    corners = {
        "top_left": (margin, margin + th),
        "top_right": (w - tw - margin, margin + th),
        "bottom_left": (margin, h - margin),
        "bottom_right": (w - tw - margin, h - margin),
    }
    x, y = corners[rng.choice(list(corners))]

    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if rng.random() < 0.5:  # blinking REC dot
        cv2.circle(img, (x - margin // 2 if x > w // 2 else x + tw + margin // 2, y - th // 2), max(2, th // 4), (0, 0, 255), -1)
    return img


def add_scanlines(image: np.ndarray, intensity: float = 0.06) -> np.ndarray:
    img = image.astype(np.float32)
    img[::2, :, :] *= 1.0 - intensity
    return np.clip(img, 0, 255).astype(np.uint8)


def add_vignette(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    h, w = image.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2.0, w / 2.0
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    dist /= dist.max()
    mask = 1.0 - strength * dist**2
    return np.clip(image.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def apply_cctv_effects(
    image: np.ndarray,
    rng: random.Random,
    fisheye_probability: float = 0.35,
    timestamp_probability: float = 0.9,
    scanline_probability: float = 0.3,
    vignette_probability: float = 0.3,
) -> np.ndarray:
    """Compose the CCTV effects stochastically (BGR or RGB uint8 in/out)."""
    img = image
    if rng.random() < fisheye_probability:
        img = apply_fisheye(img, strength=rng.uniform(0.08, 0.35))
    if rng.random() < scanline_probability:
        img = add_scanlines(img, intensity=rng.uniform(0.03, 0.1))
    if rng.random() < vignette_probability:
        img = add_vignette(img, strength=rng.uniform(0.15, 0.4))
    if rng.random() < timestamp_probability:
        img = add_timestamp_overlay(img, rng)
    return img
