"""End-to-end pipeline orchestrator.

Stages (run all or any subset, in order):
  generate  -> synthetic CCTV images via diffusion (>=100 per label)
  augment   -> CCTV-style augmentation variants
  crop      -> YOLO person detection + 20% box expansion + letterbox 224x224
  train     -> multi-label classifier (BCEWithLogitsLoss, sigmoid outputs)
  evaluate  -> report best validation metrics from the training log
  infer     -> inference demo on sample raw frames

Examples:
  python main_pipeline.py --stage all
  python main_pipeline.py --stage all --backend mock --images-per-label 10 --epochs 3   # fast smoke test
  python main_pipeline.py --stage generate --backend sd
  python main_pipeline.py --stage train
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import get_logger, load_json, load_yaml, resolve_path, set_global_seed  # noqa: E402
from labels import LABELS  # noqa: E402

logger = get_logger("main_pipeline")

STAGES = ["generate", "augment", "crop", "train", "evaluate", "infer"]


def stage_generate(diffusion_cfg):
    from diffusion.dataset_generator import DatasetGenerator

    logger.info("=== STAGE: generate (backend=%s, %d images/label x %d labels) ===",
                diffusion_cfg.get("backend"), diffusion_cfg.get("images_per_label"), len(LABELS))
    return DatasetGenerator(diffusion_cfg).run()


def stage_augment(diffusion_cfg):
    from augmentation.augmentations import augment_dataset

    aug_cfg = diffusion_cfg.get("augmentation", {})
    logger.info("=== STAGE: augment ===")
    return augment_dataset(
        annotations_path=diffusion_cfg.get("annotations_path", "dataset/synthetic_annotations.json"),
        output_dir=aug_cfg.get("output_dir", "dataset/augmented"),
        output_annotations_path=aug_cfg.get("annotations_path", "dataset/augmented_annotations.json"),
        variants_per_image=aug_cfg.get("variants_per_image", 2),
        severity=aug_cfg.get("severity", "medium"),
        seed=diffusion_cfg.get("seed", 42),
        config=aug_cfg,
    )


def stage_crop(training_cfg, diffusion_cfg):
    from preprocessing.crop_person import build_cropped_dataset

    yolo_cfg = training_cfg.get("yolo", {})
    crop_cfg = training_cfg.get("crop", {})
    aug_annotations = diffusion_cfg.get("augmentation", {}).get(
        "annotations_path", "dataset/augmented_annotations.json"
    )
    input_annotations = aug_annotations if resolve_path(aug_annotations).exists() \
        else diffusion_cfg.get("annotations_path", "dataset/synthetic_annotations.json")
    logger.info("=== STAGE: crop (YOLO detect -> expand 20%% -> letterbox 224) ===")
    return build_cropped_dataset(
        annotations_path=input_annotations,
        output_dir=crop_cfg.get("output_dir", "dataset/cropped"),
        output_annotations_path=training_cfg.get("annotations_path", "dataset/annotations.json"),
        weights=yolo_cfg.get("weights", "yolov8n.pt"),
        conf=yolo_cfg.get("conf", 0.35),
        expand=crop_cfg.get("expand", 0.2),
        target_size=crop_cfg.get("target_size", 224),
        fallback_full_image=crop_cfg.get("fallback_full_image", True),
    )


def stage_train(training_cfg):
    from model.train import train

    logger.info("=== STAGE: train (%s, BCEWithLogitsLoss, sigmoid outputs) ===",
                training_cfg.get("backbone"))
    return train(training_cfg)


def stage_evaluate(training_cfg):
    logger.info("=== STAGE: evaluate ===")
    log_path = resolve_path(training_cfg.get("log_path", "model/training_log.json"))
    if not log_path.exists():
        logger.warning("No training log at %s — run the train stage first", log_path)
        return None
    log = load_json(log_path)
    best = max(log["history"], key=lambda h: h["macro_f1"])
    logger.info("Best epoch %d | macro_f1=%.4f | mAP=%s", best["epoch"], best["macro_f1"], best["mAP"])
    for label, metrics in best["per_label"].items():
        logger.info("  %-28s P=%.3f R=%.3f F1=%.3f", label,
                    metrics["precision"], metrics["recall"], metrics["f1"])
    return best


def stage_infer(training_cfg, n_samples: int = 5):
    from model.inference import InferencePipeline

    logger.info("=== STAGE: infer (demo on %d raw frames) ===", n_samples)
    import cv2

    pipeline = InferencePipeline(
        checkpoint_path=str(resolve_path(training_cfg.get("checkpoint_dir", "model/checkpoints")) / "best.pt"),
        yolo_weights=training_cfg.get("yolo", {}).get("weights", "yolov8n.pt"),
        yolo_conf=training_cfg.get("yolo", {}).get("conf", 0.35),
        expand=training_cfg.get("crop", {}).get("expand", 0.2),
        target_size=training_cfg.get("crop", {}).get("target_size", 224),
        threshold=training_cfg.get("threshold", 0.5),
    )

    raw_dir = resolve_path("dataset/synthetic_raw")
    samples = sorted(raw_dir.rglob("*.png"))[:n_samples]
    output_dir = resolve_path("dataset/inference_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in samples:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        results = pipeline.predict_frame(frame)
        annotated = pipeline.annotate_frame(frame, results)
        out = output_dir / f"{path.stem}_pred.jpg"
        cv2.imwrite(str(out), annotated)
        logger.info("%s -> %d person(s) %s", path.name, len(results),
                    [r["active_labels"] for r in results])
    logger.info("Inference demo outputs in %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="CCTV supermarket multi-label pipeline")
    parser.add_argument("--stage", default="all", choices=["all"] + STAGES)
    parser.add_argument("--diffusion-config", default="configs/diffusion.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--backend", default=None, help="Override diffusion backend (sd|openai|gemini|mock)")
    parser.add_argument("--images-per-label", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    diffusion_cfg = load_yaml(args.diffusion_config)
    training_cfg = load_yaml(args.training_config)
    if args.backend:
        diffusion_cfg["backend"] = args.backend
    if args.images_per_label:
        diffusion_cfg["images_per_label"] = args.images_per_label
    if args.epochs:
        training_cfg["epochs"] = args.epochs

    set_global_seed(int(diffusion_cfg.get("seed", 42)))
    stages = STAGES if args.stage == "all" else [args.stage]

    for stage in stages:
        if stage == "generate":
            stage_generate(diffusion_cfg)
        elif stage == "augment":
            stage_augment(diffusion_cfg)
        elif stage == "crop":
            stage_crop(training_cfg, diffusion_cfg)
        elif stage == "train":
            stage_train(training_cfg)
        elif stage == "evaluate":
            stage_evaluate(training_cfg)
        elif stage == "infer":
            stage_infer(training_cfg)

    logger.info("Pipeline finished: %s", " -> ".join(stages))


if __name__ == "__main__":
    main()
