"""Canonical label space for the multi-label CCTV person classifier.

STRICT: this list must not be modified. Every module imports from here so the
label ordering is consistent across dataset generation, training and inference.
"""

LABELS = [
    "right_hand_visible",
    "right_hand_in_pocket",
    "right_hand_in_bag",

    "left_hand_visible",
    "left_hand_in_pocket",
    "left_hand_in_bag",

    "object_in_hand",
    "interacting_with_shelf",

    "hand_occluded_generic",
    "both_hands_not_visible",
]

NUM_LABELS = len(LABELS)
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

# Horizontal image flip turns a right hand into a left hand (and vice versa).
# Used by the training dataset to keep flip augmentation label-consistent.
FLIP_LABEL_SWAP = {
    "right_hand_visible": "left_hand_visible",
    "left_hand_visible": "right_hand_visible",
    "right_hand_in_pocket": "left_hand_in_pocket",
    "left_hand_in_pocket": "right_hand_in_pocket",
    "right_hand_in_bag": "left_hand_in_bag",
    "left_hand_in_bag": "right_hand_in_bag",
}


def labels_dict_to_vector(labels: dict) -> list:
    """Convert a {label_name: 0/1} dict to a fixed-order float vector."""
    return [float(labels.get(label, 0)) for label in LABELS]


def vector_to_labels_dict(vector) -> dict:
    """Convert a fixed-order vector back to a {label_name: 0/1} dict."""
    return {label: int(round(float(v))) for label, v in zip(LABELS, vector)}


def flip_labels_dict(labels: dict) -> dict:
    """Return the label dict after a horizontal flip (left/right swapped)."""
    return {FLIP_LABEL_SWAP.get(label, label): value for label, value in labels.items()}
