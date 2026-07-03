"""Optional fallback backend: OpenAI Images API.

Requires the `openai` package and the OPENAI_API_KEY environment variable.
Exposes the same .generate(prompt, negative_prompt, seed) interface as
LocalSDRunner so DatasetGenerator can swap backends transparently.
"""

import base64
import io
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_logger  # noqa: E402

logger = get_logger("diffusion.openai_api")


class OpenAIImageRunner:
    def __init__(self, model: str = "gpt-image-1", size: str = "1024x1024", api_key: str = None, **_unused):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the OpenAI SDK: pip install openai") from exc

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.size = size
        self.model_id = f"openai:{model}"

    def generate(self, prompt: str, negative_prompt: str = "", seed: int = 0) -> Image.Image:
        # The Images API has no negative prompt or seed; fold the negative
        # prompt into the instruction and log that the seed is advisory only.
        full_prompt = prompt
        if negative_prompt:
            full_prompt += f". Avoid: {negative_prompt}"
        logger.debug("OpenAI Images API ignores seeds; seed=%s is metadata only", seed)

        result = self.client.images.generate(model=self.model, prompt=full_prompt, size=self.size, n=1)
        data = result.data[0]
        if getattr(data, "b64_json", None):
            raw = base64.b64decode(data.b64_json)
        else:
            import urllib.request

            with urllib.request.urlopen(data.url) as resp:
                raw = resp.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")
