"""Optional extension: multi-object person tracking with ByteTrack.

Uses Ultralytics' built-in ByteTrack integration to assign stable track IDs
across video frames — the foundation for temporal smoothing and video-level
label aggregation on top of the per-crop classifier.
"""

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_device, get_logger, save_json  # noqa: E402

logger = get_logger("yolo.track_people")


class PersonTracker:
    """Streaming ByteTrack person tracker over consecutive frames."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, device: str = "auto"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.device = get_device(device)

    def update(self, frame_bgr: np.ndarray) -> list:
        """Track people in the next frame.

        Returns [{"track_id": int, "box": [x1,y1,x2,y2], "conf": float}].
        """
        results = self.model.track(
            frame_bgr,
            classes=[0],
            conf=self.conf,
            device=self.device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        tracks = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            for box, conf, track_id in zip(
                boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.id.cpu().numpy()
            ):
                tracks.append(
                    {"track_id": int(track_id), "box": [float(v) for v in box], "conf": float(conf)}
                )
        return tracks


class TemporalLabelSmoother:
    """Optional extension: per-track exponential/majority smoothing of the
    classifier's per-frame probabilities."""

    def __init__(self, window: int = 15):
        self.window = window
        self.history = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, probabilities: np.ndarray) -> np.ndarray:
        self.history[track_id].append(np.asarray(probabilities, dtype=np.float32))
        return np.mean(np.stack(self.history[track_id]), axis=0)


def track_video(video_path, output_json=None, weights="yolov8n.pt", conf=0.35) -> list:
    """Track people through a video file; returns per-frame track lists."""
    import cv2

    tracker = PersonTracker(weights=weights, conf=conf)
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append({"frame": frame_idx, "tracks": tracker.update(frame)})
        frame_idx += 1
    capture.release()
    logger.info("Tracked %d frames from %s", frame_idx, video_path)
    if output_json:
        save_json(frames, output_json)
    return frames


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track people in a video with ByteTrack")
    parser.add_argument("video")
    parser.add_argument("--output", default="dataset/tracks.json")
    args = parser.parse_args()
    track_video(args.video, output_json=args.output)
