"""Local Stable Diffusion runner (HuggingFace Diffusers).

Supports SDXL and SD 1.5 through AutoPipelineForText2Image, offline mode
(local_files_only), fp16 on CUDA, and per-image seeds for reproducibility.

Also provides MockDiffusionRunner: a procedural image generator that produces
CCTV-looking placeholder frames (shelves + person silhouettes + sensor noise)
so the full pipeline can be smoke-tested on machines without a GPU or model
weights.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_device, get_logger  # noqa: E402

logger = get_logger("diffusion.local_sd_runner")


class LocalSDRunner:
    """Text-to-image via diffusers. Works with SDXL and SD 1.5 checkpoints."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        fallback_model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "auto",
        offline: bool = False,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.5,
        height: int = 768,
        width: int = 768,
        low_vram: bool = True,
    ):
        import torch
        from diffusers import AutoPipelineForText2Image

        self.device = get_device(device)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        load_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": offline,
            "use_safetensors": True,
            # Without low_cpu_mem_usage, from_pretrained materializes weights
            # then casts dtype, needing fp32+fp16 buffers simultaneously.
            "low_cpu_mem_usage": True,
        }

        def _load(repo_id):
            # variant="fp16" fetches the half-precision weight files directly
            # instead of the ~2x larger fp32 checkpoint (the actual cause of
            # the Colab RAM crash: SDXL fp32 weights are ~14GB on CPU before
            # any cast happens).
            if dtype == torch.float16:
                try:
                    return AutoPipelineForText2Image.from_pretrained(repo_id, variant="fp16", **load_kwargs)
                except Exception as exc:
                    logger.warning("No cached fp16 variant for %s (%s); loading default weights", repo_id, exc)
            return AutoPipelineForText2Image.from_pretrained(repo_id, **load_kwargs)

        try:
            logger.info("Loading diffusion model: %s (device=%s, dtype=%s)", model_id, self.device, dtype)
            self.pipeline = _load(model_id)
            self.model_id = model_id
        except Exception as exc:  # e.g. SDXL not cached / not enough disk
            logger.warning("Failed to load %s (%s); trying fallback %s", model_id, exc, fallback_model_id)
            self.pipeline = _load(fallback_model_id)
            self.model_id = fallback_model_id

        if self.device == "cuda" and low_vram:
            # Keeps only the submodule currently in use on the GPU and the
            # rest on CPU, cutting both GPU and peak system RAM versus a
            # blanket .to("cuda"). Requires `accelerate` (already a dependency).
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(self.device)

        if self.device == "cuda":
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_vae_slicing()

    def generate(self, prompt: str, negative_prompt: str = "", seed: int = 0) -> Image.Image:
        import torch

        generator = torch.Generator(device=self.device if self.device != "mps" else "cpu")
        generator.manual_seed(int(seed))
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
            generator=generator,
        )
        return result.images[0]


class MockDiffusionRunner:
    """Procedural CCTV-style placeholder generator (no GPU / model required).

    Draws a grey supermarket floor, shelf blocks, 1-3 person silhouettes and
    heavy sensor noise. Deterministic for a given seed. Useful for CI and for
    validating the end-to-end pipeline before spending GPU time.
    """

    def __init__(self, height: int = 512, width: int = 512, **_unused):
        self.height = height
        self.width = width
        self.model_id = "mock"

    def generate(self, prompt: str, negative_prompt: str = "", seed: int = 0) -> Image.Image:
        import cv2

        rng = np.random.default_rng(int(seed))
        h, w = self.height, self.width

        # Floor gradient
        base = np.tile(np.linspace(70, 130, h, dtype=np.float32)[:, None], (1, w))
        img = np.stack([base, base, base * 0.98], axis=-1)

        # Shelf blocks along both sides
        shelf_color = rng.uniform(30, 80, size=3)
        for x0 in (0, int(w * 0.78)):
            cv2.rectangle(img, (x0, 0), (x0 + int(w * 0.22), h), shelf_color.tolist(), -1)
            for row in range(4):
                y = int(h * (0.15 + 0.22 * row))
                cv2.line(img, (x0, y), (x0 + int(w * 0.22), y), (140, 140, 140), 2)
                for k in range(5):
                    px = x0 + int(w * 0.03) + k * int(w * 0.035)
                    color = rng.uniform(40, 220, size=3).tolist()
                    cv2.rectangle(img, (px, y - 22), (px + 12, y - 2), color, -1)

        # Person silhouettes (torso + head + legs), so YOLO has a chance on
        # real runs and the crop stage always has a fallback region.
        for _ in range(int(rng.integers(1, 4))):
            cx = int(rng.uniform(0.3, 0.7) * w)
            cy = int(rng.uniform(0.35, 0.75) * h)
            ph = int(rng.uniform(0.22, 0.4) * h)
            pw = max(10, ph // 3)
            color = rng.uniform(20, 100, size=3).tolist()
            cv2.rectangle(img, (cx - pw // 2, cy - ph // 2), (cx + pw // 2, cy + ph // 2), color, -1)
            cv2.circle(img, (cx, cy - ph // 2 - pw // 3), pw // 3, color, -1)

        # CCTV degradation: noise + slight blur + downscale/upscale grain
        img = img + rng.normal(0, 12, size=img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        return Image.fromarray(img)


if __name__ == "__main__":
    runner = MockDiffusionRunner()
    image = runner.generate("test prompt", seed=42)
    out = Path(__file__).resolve().parents[1] / "dataset" / "mock_test.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"Saved mock test image to {out}")
