"""Optional fallback backend: Google Gemini / Imagen image generation.

Requires the `google-genai` package and the GEMINI_API_KEY (or GOOGLE_API_KEY)
environment variable. Same .generate() interface as the other runners.
"""

import io
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_logger  # noqa: E402

logger = get_logger("diffusion.gemini_api")


class GeminiImageRunner:
    def __init__(self, model: str = "imagen-3.0-generate-002", api_key: str = None, **_unused):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Install the Google GenAI SDK: pip install google-genai") from exc

        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.model_id = f"gemini:{model}"

    def generate(self, prompt: str, negative_prompt: str = "", seed: int = 0) -> Image.Image:
        from google.genai import types

        logger.debug("Gemini image generation treats seed=%s as metadata only", seed)
        full_prompt = prompt
        if negative_prompt:
            full_prompt += f". Avoid: {negative_prompt}"

        response = self.client.models.generate_images(
            model=self.model,
            prompt=full_prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )
        image_bytes = response.generated_images[0].image.image_bytes
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
