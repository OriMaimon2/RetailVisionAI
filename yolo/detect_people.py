"""Person detection with pretrained YOLOv8 (Ultralytics).

PersonDetector wraps a YOLO checkpoint, restricts detections to COCO class 0
(person) and returns [x1, y1, x2, y2, confidence] boxes per frame.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_device, get_logger, resolve_path, save_json  # noqa: E402

logger = get_logger("yolo.detect_people")

PERSON_CLASS_ID = 0  # COCO


class PersonDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, device: str = "auto"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.device = get_device(device)

    def detect(self, image_bgr: np.ndarray) -> list:
        """Detect people in one BGR frame.

        Returns a list of dicts: {"box": [x1, y1, x2, y2], "conf": float},
        sorted by confidence descending.
        """
        results = self.model.predict(
            image_bgr,
            classes=[PERSON_CLASS_ID],
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                detections.append({"box": [float(v) for v in box], "conf": float(conf)})
        detections.sort(key=lambda d: d["conf"], reverse=True)
        return detections

    def detect_file(self, image_path) -> list:
        image = cv2.imread(str(resolve_path(image_path)))
        if image is None:
            logger.warning("Unreadable image: %s", image_path)
            return []
        return self.detect(image)


def detect_directory(image_dir, output_json=None, weights="yolov8n.pt", conf=0.35) -> dict:
    """Run detection over every image in a directory tree.

    Returns {relative_image_path: [detections]} and optionally saves it.
    """
    image_dir = resolve_path(image_dir)
    detector = PersonDetector(weights=weights, conf=conf)
    results = {}
    paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for path in paths:
        results[str(path)] = detector.detect_file(path)
    logger.info("Detected people in %d images", len(results))
    if output_json:
        save_json(results, output_json)
    return results


def draw_detections(image_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Utility for demos: draw person boxes on a copy of the frame."""
    img = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"person {det['conf']:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Detect people with YOLOv8")
    parser.add_argument("--image-dir", default="dataset/synthetic_raw")
    parser.add_argument("--output", default="dataset/detections.json")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    detect_directory(args.image_dir, output_json=args.output, weights=args.weights, conf=args.conf)


if __name__ == "__main__":
    main()
