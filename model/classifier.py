"""Multi-label classifier: CNN backbone + 10 sigmoid outputs.

Design rules honored here:
- The head outputs raw logits (NO softmax). Training uses BCEWithLogitsLoss;
  inference applies an element-wise sigmoid.
- Backbones: resnet18 or efficientnet_b0 (torchvision, ImageNet-pretrained).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labels import LABELS, NUM_LABELS  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MultiLabelClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        num_labels: int = NUM_LABELS,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_labels = num_labels

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            in_features = net.classifier[1].in_features
            net.classifier = nn.Identity()
            self.backbone = net
        else:
            raise ValueError(f"Unsupported backbone: {backbone!r} (resnet18 | efficientnet_b0)")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_labels),  # raw logits — sigmoid applied outside
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (batch, num_labels)."""
        return self.head(self.backbone(x))

    @torch.no_grad()
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid probabilities per label (multi-label, NOT softmax)."""
        self.eval()
        return torch.sigmoid(self.forward(x))


def save_checkpoint(model: MultiLabelClassifier, path, config: dict = None, extra: dict = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": model.backbone_name,
            "num_labels": model.num_labels,
            "labels": LABELS,
            "config": config or {},
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path, device: str = "cpu") -> MultiLabelClassifier:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MultiLabelClassifier(
        backbone=checkpoint["backbone"],
        num_labels=checkpoint["num_labels"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model
