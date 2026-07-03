"""Augmentation pipeline for the raw synthetic dataset.

Albumentations covers: gaussian noise, motion blur, brightness/contrast
shifts, JPEG compression artifacts, perspective transform and random crop
jitter. augmentation/cctv_effects.py adds fisheye distortion and a burned-in
CCTV timestamp overlay.

Each raw image gets `variants_per_image` augmented copies with identical
labels; the combined annotations are written to
dataset/augmented_annotations.json.
"""

import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from augmentation.cctv_effects import apply_cctv_effects  # noqa: E402
from common import ensure_dir, get_logger, load_json, load_yaml, resolve_path, save_json  # noqa: E402

logger = get_logger("augmentation")

SEVERITY_PRESETS = {
    "light": {"noise": (5.0, 25.0), "blur": (3, 5), "bc": 0.15, "jpeg": (50, 85), "persp": 0.04},
    "medium": {"noise": (10.0, 60.0), "blur": (3, 9), "bc": 0.3, "jpeg": (25, 70), "persp": 0.08},
    "heavy": {"noise": (20.0, 90.0), "blur": (5, 13), "bc": 0.4, "jpeg": (15, 55), "persp": 0.12},
}


def build_augmentation_pipeline(severity: str = "medium") -> A.Compose:
    """Albumentations pipeline covering the required degradation families."""
    p = SEVERITY_PRESETS[severity]
    return A.Compose(
        [
            # crop jitter: small random shift/scale, mild rotation
            A.ShiftScaleRotate(
                shift_limit=0.06, scale_limit=0.12, rotate_limit=3,
                border_mode=cv2.BORDER_REPLICATE, p=0.6,
            ),
            A.Perspective(scale=(0.02, p["persp"]), pad_mode=cv2.BORDER_REPLICATE, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=p["bc"], contrast_limit=p["bc"], p=0.8),
            A.MotionBlur(blur_limit=p["blur"], p=0.5),
            A.GaussNoise(var_limit=p["noise"], p=0.7),
            A.ImageCompression(quality_lower=p["jpeg"][0], quality_upper=p["jpeg"][1], p=0.7),
        ]
    )


def augment_image(image_bgr: np.ndarray, pipeline: A.Compose, rng: random.Random, cfg: dict) -> np.ndarray:
    """One augmented variant: albumentations pass + CCTV effects pass."""
    augmented = pipeline(image=image_bgr)["image"]
    return apply_cctv_effects(
        augmented,
        rng,
        fisheye_probability=cfg.get("fisheye_probability", 0.35),
        timestamp_probability=cfg.get("timestamp_probability", 0.9),
    )


def augment_dataset(
    annotations_path="dataset/synthetic_annotations.json",
    output_dir="dataset/augmented",
    output_annotations_path="dataset/augmented_annotations.json",
    variants_per_image: int = 2,
    severity: str = "medium",
    seed: int = 42,
    config: dict = None,
) -> list:
    """Augment every raw image; returns combined (raw + augmented) annotations."""
    config = config or {}
    annotations = load_json(annotations_path)
    output_dir = ensure_dir(output_dir)
    pipeline = build_augmentation_pipeline(severity)

    combined = list(annotations)
    for idx, entry in enumerate(tqdm(annotations, desc="augment")):
        src_path = resolve_path(entry["image_path"])
        image = cv2.imread(str(src_path))
        if image is None:
            logger.warning("Unreadable image, skipping: %s", src_path)
            continue

        for v in range(variants_per_image):
            # Derived per-variant seed keeps the whole stage reproducible.
            variant_seed = seed * 1_000_003 + idx * 31 + v
            rng = random.Random(variant_seed)
            random.seed(variant_seed)
            np.random.seed(variant_seed % (2**32))

            augmented = augment_image(image, pipeline, rng, config)
            stem = Path(entry["image_path"]).stem
            rel_path = f"dataset/augmented/{stem}_aug{v}.jpg"
            cv2.imwrite(str(resolve_path(rel_path)), augmented, [cv2.IMWRITE_JPEG_QUALITY, 90])

            combined.append(
                {
                    **{k: entry[k] for k in ("labels", "primary_label", "seed", "backend") if k in entry},
                    "image_path": rel_path,
                    "source_image": entry["image_path"],
                    "augmentation": {"variant": v, "severity": severity, "seed": variant_seed},
                    "source": "augmented",
                }
            )

    save_json(combined, output_annotations_path)
    logger.info(
        "Augmentation complete: %d raw + %d augmented = %d entries -> %s",
        len(annotations), len(combined) - len(annotations), len(combined),
        resolve_path(output_annotations_path),
    )
    return combined


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Augment the synthetic dataset")
    parser.add_argument("--config", default="configs/diffusion.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    aug_cfg = cfg.get("augmentation", {})
    augment_dataset(
        annotations_path=cfg.get("annotations_path", "dataset/synthetic_annotations.json"),
        output_dir=aug_cfg.get("output_dir", "dataset/augmented"),
        output_annotations_path=aug_cfg.get("annotations_path", "dataset/augmented_annotations.json"),
        variants_per_image=aug_cfg.get("variants_per_image", 2),
        severity=aug_cfg.get("severity", "medium"),
        seed=cfg.get("seed", 42),
        config=aug_cfg,
    )


if __name__ == "__main__":
    main()
