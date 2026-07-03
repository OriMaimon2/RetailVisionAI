"""Synthetic dataset generation orchestrator.

For each of the 10 labels, generates N images (default 100) with the selected
backend (local Stable Diffusion, OpenAI, Gemini, or the procedural mock),
saves them under dataset/synthetic_raw/<label>/ and logs full metadata
(prompt, seed, backend, diversity attributes, multi-label ground truth) to
dataset/synthetic_annotations.json.

Generation is resumable: images already on disk with a matching metadata
entry are skipped.
"""

import hashlib
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ensure_dir, get_logger, load_json, load_yaml, resolve_path, save_json  # noqa: E402
from diffusion.prompt_generator import PromptGenerator  # noqa: E402
from labels import LABELS  # noqa: E402

logger = get_logger("diffusion.dataset_generator")


def create_runner(backend: str, config: dict):
    """Instantiate an image-generation backend by name."""
    if backend == "sd":
        from diffusion.local_sd_runner import LocalSDRunner

        return LocalSDRunner(
            model_id=config.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
            fallback_model_id=config.get("fallback_model_id", "runwayml/stable-diffusion-v1-5"),
            device=config.get("device", "auto"),
            offline=config.get("offline", False),
            num_inference_steps=config.get("num_inference_steps", 30),
            guidance_scale=config.get("guidance_scale", 6.5),
            height=config.get("height", 768),
            width=config.get("width", 768),
        )
    if backend == "openai":
        from diffusion.openai_api import OpenAIImageRunner

        return OpenAIImageRunner(model=config.get("openai_model", "gpt-image-1"))
    if backend == "gemini":
        from diffusion.gemini_api import GeminiImageRunner

        return GeminiImageRunner(model=config.get("gemini_model", "imagen-3.0-generate-002"))
    if backend == "mock":
        from diffusion.local_sd_runner import MockDiffusionRunner

        return MockDiffusionRunner(height=config.get("height", 512), width=config.get("width", 512))
    raise ValueError(f"Unknown backend: {backend!r} (expected sd | openai | gemini | mock)")


def _image_seed(base_seed: int, label: str, index: int) -> int:
    """Deterministic per-image seed derived from base seed, label and index."""
    digest = hashlib.sha256(f"{base_seed}:{label}:{index}".encode()).hexdigest()
    return int(digest[:8], 16)


class DatasetGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.base_seed = int(config.get("seed", 42))
        self.images_per_label = int(config.get("images_per_label", 100))
        self.output_dir = ensure_dir(config.get("output_dir", "dataset/synthetic_raw"))
        self.annotations_path = resolve_path(
            config.get("annotations_path", "dataset/synthetic_annotations.json")
        )
        self.backend_name = config.get("backend", "sd")
        self.runner = create_runner(self.backend_name, config)
        self.prompt_generator = PromptGenerator(seed=self.base_seed)

    def run(self) -> list:
        annotations = []
        if self.annotations_path.exists():
            annotations = load_json(self.annotations_path)
        existing = {entry["image_path"] for entry in annotations}

        for label in LABELS:
            label_dir = ensure_dir(self.output_dir / label)
            progress = tqdm(range(self.images_per_label), desc=f"generate:{label}")
            for i in progress:
                rel_path = f"dataset/synthetic_raw/{label}/{label}_{i:04d}.png"
                abs_path = label_dir / f"{label}_{i:04d}.png"

                # PromptGenerator must advance even on skip so resumed runs
                # keep producing the same prompt sequence.
                sample = self.prompt_generator.generate(label)
                if rel_path in existing and abs_path.exists():
                    continue

                seed = _image_seed(self.base_seed, label, i)
                try:
                    image = self.runner.generate(
                        prompt=sample["prompt"],
                        negative_prompt=sample["negative_prompt"],
                        seed=seed,
                    )
                except Exception as exc:
                    logger.error("Generation failed for %s (%s); skipping", rel_path, exc)
                    continue

                image.save(abs_path)
                annotations.append(
                    {
                        "image_path": rel_path,
                        "labels": sample["labels"],
                        "primary_label": label,
                        "prompt": sample["prompt"],
                        "negative_prompt": sample["negative_prompt"],
                        "attributes": sample["attributes"],
                        "seed": seed,
                        "backend": self.backend_name,
                        "model_id": getattr(self.runner, "model_id", self.backend_name),
                        "source": "synthetic_raw",
                    }
                )
                existing.add(rel_path)

                if (i + 1) % 20 == 0:
                    save_json(annotations, self.annotations_path)

        save_json(annotations, self.annotations_path)
        logger.info(
            "Dataset generation complete: %d images, metadata at %s",
            len(annotations),
            self.annotations_path,
        )
        return annotations


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic CCTV supermarket dataset")
    parser.add_argument("--config", default="configs/diffusion.yaml")
    parser.add_argument("--backend", default=None, help="Override: sd | openai | gemini | mock")
    parser.add_argument("--images-per-label", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.backend:
        config["backend"] = args.backend
    if args.images_per_label:
        config["images_per_label"] = args.images_per_label

    DatasetGenerator(config).run()


if __name__ == "__main__":
    main()
