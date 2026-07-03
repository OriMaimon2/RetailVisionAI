"""Inference on full frames: YOLO person detection -> 20% box expansion ->
letterboxed 224x224 crops -> multi-label sigmoid classifier.

Returns per-person probabilities for all 10 labels and can render an
annotated frame. Usable as a library (InferencePipeline) or CLI.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_device, get_logger, load_yaml, resolve_path, save_json  # noqa: E402
from labels import LABELS  # noqa: E402
from model.classifier import IMAGENET_MEAN, IMAGENET_STD, load_checkpoint  # noqa: E402
from preprocessing.crop_person import crop_person  # noqa: E402
from yolo.detect_people import PersonDetector  # noqa: E402

logger = get_logger("model.inference")


class InferencePipeline:
    def __init__(
        self,
        checkpoint_path="model/checkpoints/best.pt",
        yolo_weights: str = "yolov8n.pt",
        yolo_conf: float = 0.35,
        expand: float = 0.2,
        target_size: int = 224,
        threshold: float = 0.5,
        device: str = "auto",
    ):
        self.device = get_device(device)
        self.model = load_checkpoint(resolve_path(checkpoint_path), device=self.device)
        self.detector = PersonDetector(weights=yolo_weights, conf=yolo_conf, device=self.device)
        self.expand = expand
        self.target_size = target_size
        self.threshold = threshold
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def _to_tensor(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        return torch.from_numpy(rgb.transpose(2, 0, 1))

    @torch.no_grad()
    def predict_frame(self, frame_bgr: np.ndarray) -> list:
        """Classify every detected person in a frame.

        Returns [{"box": [...], "det_conf": float,
                  "probabilities": {label: p}, "active_labels": [...]}].
        """
        detections = self.detector.detect(frame_bgr)
        if not detections:
            return []

        crops, kept = [], []
        for det in detections:
            canvas, used_box = crop_person(frame_bgr, det["box"], self.expand, self.target_size)
            if canvas is None:
                continue
            crops.append(self._to_tensor(canvas))
            kept.append((det, used_box))
        if not crops:
            return []

        batch = torch.stack(crops).to(self.device)
        probs = torch.sigmoid(self.model(batch)).cpu().numpy()  # multi-label sigmoid

        results = []
        for (det, used_box), p in zip(kept, probs):
            probabilities = {label: round(float(v), 4) for label, v in zip(LABELS, p)}
            results.append(
                {
                    "box": [int(v) for v in used_box],
                    "det_conf": round(float(det["conf"]), 4),
                    "probabilities": probabilities,
                    "active_labels": [l for l, v in probabilities.items() if v >= self.threshold],
                }
            )
        return results

    def annotate_frame(self, frame_bgr: np.ndarray, results: list = None) -> np.ndarray:
        """Draw boxes + active labels on a copy of the frame."""
        if results is None:
            results = self.predict_frame(frame_bgr)
        img = frame_bgr.copy()
        for r in results:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
            for k, label in enumerate(r["active_labels"] or ["<none>"]):
                y = y1 + 16 + k * 16
                cv2.putText(img, label, (x1 + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, label, (x1 + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 0), 1, cv2.LINE_AA)
        return img


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on frames")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--image-dir", help="Directory of images")
    parser.add_argument("--checkpoint", default="model/checkpoints/best.pt")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", default="dataset/inference_output")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    pipeline = InferencePipeline(
        checkpoint_path=args.checkpoint,
        yolo_weights=cfg.get("yolo", {}).get("weights", "yolov8n.pt"),
        yolo_conf=cfg.get("yolo", {}).get("conf", 0.35),
        expand=cfg.get("crop", {}).get("expand", 0.2),
        target_size=cfg.get("crop", {}).get("target_size", 224),
        threshold=args.threshold if args.threshold is not None else cfg.get("threshold", 0.5),
    )

    if args.image:
        paths = [Path(args.image)]
    elif args.image_dir:
        paths = sorted(p for p in resolve_path(args.image_dir).rglob("*")
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    else:
        parser.error("Provide --image or --image-dir")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for path in paths:
        frame = cv2.imread(str(resolve_path(path)))
        if frame is None:
            logger.warning("Unreadable: %s", path)
            continue
        results = pipeline.predict_frame(frame)
        all_results[str(path)] = results
        annotated = pipeline.annotate_frame(frame, results)
        out_path = output_dir / f"{Path(path).stem}_pred.jpg"
        cv2.imwrite(str(out_path), annotated)
        logger.info("%s -> %d person(s), saved %s", path, len(results), out_path)
        for r in results:
            logger.info("  box=%s labels=%s", r["box"], r["active_labels"])

    save_json(all_results, output_dir / "predictions.json")


if __name__ == "__main__":
    main()
