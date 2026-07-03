"""Person cropping: expand YOLO box by 20%, crop, letterbox to 224x224.

build_cropped_dataset() consumes the combined (raw + augmented) annotations,
runs the person detector on every image, crops the best person box and writes
the final training annotations to dataset/annotations.json in the required
format: {"image_path": ..., "labels": {...}}  (plus provenance metadata).

If no person is detected (a known failure mode of diffusion outputs), the
full frame is letterboxed as a fallback and flagged, so no labeled sample is
silently lost. Set fallback_full_image: false to drop such images instead.
"""

import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ensure_dir, get_logger, load_json, resolve_path, save_json  # noqa: E402
from preprocessing.letterbox import letterbox  # noqa: E402

logger = get_logger("preprocessing.crop_person")


def expand_box(box, img_w: int, img_h: int, expand: float = 0.2):
    """Expand [x1,y1,x2,y2] by `expand` fraction per side, clipped to image."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * expand / 2.0, bh * expand / 2.0
    x1 = max(0, int(round(x1 - dx)))
    y1 = max(0, int(round(y1 - dy)))
    x2 = min(img_w, int(round(x2 + dx)))
    y2 = min(img_h, int(round(y2 + dy)))
    return x1, y1, x2, y2


def crop_person(image, box, expand: float = 0.2, target_size: int = 224):
    """Crop one person region (expanded) and letterbox it. Returns the crop
    canvas and the expanded box actually used."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = expand_box(box, w, h, expand)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, (x1, y1, x2, y2)
    crop = image[y1:y2, x1:x2]
    canvas, _, _ = letterbox(crop, target_size=target_size)
    return canvas, (x1, y1, x2, y2)


def _pick_best_detection(detections):
    """Highest confidence wins; area breaks near-ties (the labeled subject is
    assumed to be the most prominent person in the frame)."""
    if not detections:
        return None
    top_conf = detections[0]["conf"]
    candidates = [d for d in detections if d["conf"] >= top_conf - 0.1]
    return max(candidates, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))


def build_cropped_dataset(
    annotations_path="dataset/augmented_annotations.json",
    output_dir="dataset/cropped",
    output_annotations_path="dataset/annotations.json",
    weights="yolov8n.pt",
    conf: float = 0.35,
    expand: float = 0.2,
    target_size: int = 224,
    fallback_full_image: bool = True,
) -> list:
    from yolo.detect_people import PersonDetector

    annotations = load_json(annotations_path)
    output_dir = ensure_dir(output_dir)
    detector = PersonDetector(weights=weights, conf=conf)

    cropped_annotations = []
    stats = {"detected": 0, "fallback": 0, "dropped": 0}

    for entry in tqdm(annotations, desc="crop"):
        src_path = resolve_path(entry["image_path"])
        image = cv2.imread(str(src_path))
        if image is None:
            logger.warning("Unreadable image, skipping: %s", src_path)
            stats["dropped"] += 1
            continue

        best = _pick_best_detection(detector.detect(image))
        if best is not None:
            canvas, used_box = crop_person(image, best["box"], expand=expand, target_size=target_size)
            det_conf, fallback = best["conf"], False
        else:
            canvas, used_box, det_conf, fallback = None, None, 0.0, True

        if canvas is None:
            if not fallback_full_image:
                stats["dropped"] += 1
                continue
            canvas, _, _ = letterbox(image, target_size=target_size)
            used_box = [0, 0, image.shape[1], image.shape[0]]
            fallback = True

        stats["fallback" if fallback else "detected"] += 1
        rel_path = f"dataset/cropped/{Path(entry['image_path']).stem}_crop.jpg"
        cv2.imwrite(str(resolve_path(rel_path)), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

        cropped_annotations.append(
            {
                "image_path": rel_path,
                "labels": entry["labels"],
                "primary_label": entry.get("primary_label"),
                "source_image": entry["image_path"],
                "box": [int(v) for v in used_box],
                "det_conf": round(float(det_conf), 4),
                "fallback_full_image": fallback,
            }
        )

    save_json(cropped_annotations, output_annotations_path)
    logger.info(
        "Cropping complete: %(detected)d detected, %(fallback)d full-image fallbacks, "
        "%(dropped)d dropped", stats,
    )
    logger.info("Final annotations: %s", resolve_path(output_annotations_path))
    return cropped_annotations


def main():
    import argparse

    from common import load_yaml

    parser = argparse.ArgumentParser(description="Crop person regions and letterbox to 224x224")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--annotations", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    yolo_cfg, crop_cfg = cfg.get("yolo", {}), cfg.get("crop", {})
    build_cropped_dataset(
        annotations_path=args.annotations or crop_cfg.get("input_annotations", "dataset/augmented_annotations.json"),
        output_dir=crop_cfg.get("output_dir", "dataset/cropped"),
        output_annotations_path=cfg.get("annotations_path", "dataset/annotations.json"),
        weights=yolo_cfg.get("weights", "yolov8n.pt"),
        conf=yolo_cfg.get("conf", 0.35),
        expand=crop_cfg.get("expand", 0.2),
        target_size=crop_cfg.get("target_size", 224),
        fallback_full_image=crop_cfg.get("fallback_full_image", True),
    )


if __name__ == "__main__":
    main()
