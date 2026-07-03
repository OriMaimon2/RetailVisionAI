"""Automatic per-label prompt generation with diversity axes.

Every prompt is built from the CCTV base prompt + a label-specific fragment +
randomly (but reproducibly) sampled diversity attributes: lighting, camera
angle, crowd density, clothing, store layout and image-quality degradation.

The generator also emits the full multi-label ground-truth dict for the image,
including deterministic implied labels and stochastic co-labels.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labels import LABELS  # noqa: E402

BASE_PROMPT = (
    "CCTV footage inside a supermarket, wide-angle security camera, "
    "grainy low resolution, realistic lighting, aisles with products, "
    "people shopping, surveillance camera perspective"
)

NEGATIVE_PROMPT = (
    "cartoon, illustration, anime, painting, drawing, 3d render, cgi, "
    "text overlay, watermark, logo, deformed hands, extra fingers, extra limbs, "
    "duplicate person, close-up portrait, bokeh, shallow depth of field, "
    "professional photography, studio lighting, high fashion"
)

LIGHTING = [
    "harsh fluorescent ceiling lighting",
    "dim evening lighting with dark corners",
    "bright uniform supermarket lighting",
    "flickering fluorescent tubes, uneven illumination",
    "cold white LED lighting",
    "warm yellowish lighting near the bakery section",
]

CAMERA_ANGLE = [
    "overhead ceiling-mounted camera looking almost straight down",
    "corner-mounted camera looking down the length of the aisle",
    "high wall-mounted camera at 45 degrees",
    "dome camera fisheye view from the ceiling",
    "camera mounted above the checkout area looking across the store",
]

CROWD = [
    "empty aisle with a single person",
    "one person in the foreground, a couple of shoppers far in the background",
    "moderately busy store with several shoppers",
    "crowded store, multiple shoppers with carts",
]

CLOTHING = [
    "wearing a dark hoodie",
    "wearing a thick winter jacket",
    "wearing a t-shirt and jeans",
    "wearing a long overcoat",
    "wearing business casual clothes",
    "wearing sportswear and a baseball cap",
    "wearing a puffer vest over a sweater",
]

LAYOUT = [
    "narrow aisle with tall shelves full of canned goods",
    "wide aisle next to refrigerated display cases",
    "produce section with fruit and vegetable stands",
    "snack aisle with colorful packaging",
    "beverage aisle with bottles and crates",
    "checkout lanes visible in the background",
]

QUALITY = [
    "slightly blurry footage",
    "heavy sensor noise",
    "mild motion blur",
    "visible compression artifacts",
    "washed-out desaturated colors",
    "slight interlacing artifacts",
]

# Label-specific prompt fragment + stochastic co-labels (probability that the
# co-label is also set to 1 and its cue appended to the prompt).
LABEL_SPECS = {
    "right_hand_visible": {
        "fragment": "a person standing in the aisle with their right hand clearly visible at their side",
        "co_labels": {"left_hand_visible": 0.6},
    },
    "right_hand_in_pocket": {
        "fragment": "a person standing in the aisle with their right hand inside their pants pocket",
        "co_labels": {"left_hand_visible": 0.7},
    },
    "right_hand_in_bag": {
        "fragment": "a person holding an open shopping bag, right hand inside the bag",
        "co_labels": {"left_hand_visible": 0.5},
    },
    "left_hand_visible": {
        "fragment": "a person standing in the aisle with their left hand clearly visible at their side",
        "co_labels": {"right_hand_visible": 0.6},
    },
    "left_hand_in_pocket": {
        "fragment": "a person standing in the aisle with their left hand inside their jacket pocket",
        "co_labels": {"right_hand_visible": 0.7},
    },
    "left_hand_in_bag": {
        "fragment": "a person holding an open tote bag, left hand reaching inside the bag",
        "co_labels": {"right_hand_visible": 0.5},
    },
    "object_in_hand": {
        "fragment": "a person holding a product taken from the shelf in their hand",
        "co_labels": {"right_hand_visible": 0.5, "interacting_with_shelf": 0.3},
    },
    "interacting_with_shelf": {
        "fragment": "a person reaching toward the supermarket shelf, touching the products",
        "co_labels": {"right_hand_visible": 0.5, "object_in_hand": 0.4},
    },
    "hand_occluded_generic": {
        "fragment": "a person partially hidden behind shelves and other shoppers so one hand is occluded from the camera",
        "co_labels": {"left_hand_visible": 0.3},
    },
    "both_hands_not_visible": {
        "fragment": "a person seen from behind with both hands completely hidden from the camera view",
        "co_labels": {"hand_occluded_generic": 0.5},
    },
}

# Extra prompt cues appended when a co-label fires, so the generated pixels
# actually match the label vector.
CO_LABEL_CUES = {
    "right_hand_visible": "right hand clearly visible",
    "left_hand_visible": "left hand clearly visible",
    "object_in_hand": "holding a small product",
    "interacting_with_shelf": "reaching toward the shelf",
    "hand_occluded_generic": "one hand occluded by the body",
}

# Labels that can never co-occur (per hand, and the both-hidden case).
MUTUAL_EXCLUSIONS = [
    {"right_hand_visible", "right_hand_in_pocket", "right_hand_in_bag"},
    {"left_hand_visible", "left_hand_in_pocket", "left_hand_in_bag"},
    {"both_hands_not_visible", "right_hand_visible"},
    {"both_hands_not_visible", "left_hand_visible"},
]


class PromptGenerator:
    """Reproducible prompt + ground-truth generator for a given seed."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _sample_co_labels(self, primary_label: str, labels: dict) -> list:
        cues = []
        for co_label, prob in LABEL_SPECS[primary_label]["co_labels"].items():
            if self.rng.random() >= prob:
                continue
            candidate = dict(labels)
            candidate[co_label] = 1
            active = {l for l, v in candidate.items() if v == 1}
            if any(len(group & active) > 1 for group in MUTUAL_EXCLUSIONS):
                continue
            labels[co_label] = 1
            if co_label in CO_LABEL_CUES:
                cues.append(CO_LABEL_CUES[co_label])
        return cues

    def generate(self, primary_label: str) -> dict:
        """Build one prompt sample for the given primary label.

        Returns a dict with: prompt, negative_prompt, labels (all 10 keys,
        0/1), primary_label and the sampled diversity attributes.
        """
        if primary_label not in LABEL_SPECS:
            raise ValueError(f"Unknown label: {primary_label}")

        attributes = {
            "lighting": self.rng.choice(LIGHTING),
            "camera_angle": self.rng.choice(CAMERA_ANGLE),
            "crowd": self.rng.choice(CROWD),
            "clothing": self.rng.choice(CLOTHING),
            "layout": self.rng.choice(LAYOUT),
            "quality": self.rng.choice(QUALITY),
        }

        labels = {label: 0 for label in LABELS}
        labels[primary_label] = 1
        co_cues = self._sample_co_labels(primary_label, labels)

        parts = [
            BASE_PROMPT,
            LABEL_SPECS[primary_label]["fragment"],
            attributes["clothing"],
        ]
        parts.extend(co_cues)
        parts.extend(
            [
                attributes["layout"],
                attributes["crowd"],
                attributes["lighting"],
                attributes["camera_angle"],
                attributes["quality"],
            ]
        )

        return {
            "prompt": ", ".join(parts),
            "negative_prompt": NEGATIVE_PROMPT,
            "labels": labels,
            "primary_label": primary_label,
            "attributes": attributes,
        }


if __name__ == "__main__":
    generator = PromptGenerator(seed=42)
    for label in LABELS:
        sample = generator.generate(label)
        active = [l for l, v in sample["labels"].items() if v]
        print(f"\n### {label} -> labels={active}")
        print(sample["prompt"])
