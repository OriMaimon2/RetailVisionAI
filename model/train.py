"""Training for the multi-label person-crop classifier.

- BCEWithLogitsLoss (multi-label, sigmoid — never softmax), optional
  pos_weight computed from label frequencies to counter imbalance.
- Horizontal-flip augmentation swaps left/right hand labels so the flip stays
  label-consistent.
- Fixed seeds everywhere (python / numpy / torch / DataLoader workers).
- Per-label precision/recall/F1 @ threshold, macro F1 and mAP each epoch;
  best checkpoint by macro F1; full history logged to model/training_log.json.
"""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_device, get_logger, load_json, load_yaml, resolve_path, save_json, set_global_seed  # noqa: E402
from labels import LABELS, flip_labels_dict, labels_dict_to_vector  # noqa: E402
from model.classifier import IMAGENET_MEAN, IMAGENET_STD, MultiLabelClassifier, save_checkpoint  # noqa: E402
from preprocessing.letterbox import letterbox  # noqa: E402

logger = get_logger("model.train")


class PersonCropDataset(Dataset):
    """Loads 224x224 person crops + 10-dim multi-label targets."""

    def __init__(self, entries: list, image_size: int = 224, train: bool = False,
                 horizontal_flip_prob: float = 0.5):
        self.entries = entries
        self.image_size = image_size
        self.train = train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = cv2.imread(str(resolve_path(entry["image_path"])))
        if image is None:
            raise FileNotFoundError(f"Missing crop: {entry['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (self.image_size, self.image_size):
            image, _, _ = letterbox(image, target_size=self.image_size)

        labels = entry["labels"]
        if self.train and random.random() < self.horizontal_flip_prob:
            image = np.ascontiguousarray(image[:, ::-1, :])
            labels = flip_labels_dict(labels)  # right<->left hand labels

        tensor = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1))
        target = torch.tensor(labels_dict_to_vector(labels), dtype=torch.float32)
        return tensor, target


def split_dataset(entries: list, val_split: float, seed: int):
    rng = random.Random(seed)
    indices = list(range(len(entries)))
    rng.shuffle(indices)
    n_val = max(1, int(len(entries) * val_split))
    val_idx = set(indices[:n_val])
    train = [e for i, e in enumerate(entries) if i not in val_idx]
    val = [e for i, e in enumerate(entries) if i in val_idx]
    return train, val


def compute_pos_weight(entries: list) -> torch.Tensor:
    """pos_weight[i] = (#negatives / #positives) for label i, clamped."""
    counts = np.zeros(len(LABELS), dtype=np.float64)
    for entry in entries:
        counts += np.array(labels_dict_to_vector(entry["labels"]))
    total = len(entries)
    pos = np.clip(counts, 1, None)
    weight = (total - counts) / pos
    return torch.tensor(np.clip(weight, 0.5, 20.0), dtype=torch.float32)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AP for one label; manual fallback if sklearn is unavailable."""
    if y_true.sum() == 0:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_score))
    except ImportError:
        order = np.argsort(-y_score)
        y_sorted = y_true[order]
        cumulative = np.cumsum(y_sorted)
        precision = cumulative / (np.arange(len(y_sorted)) + 1)
        return float((precision * y_sorted).sum() / y_sorted.sum())


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(np.float32)
    per_label, f1s, aps = {}, [], []
    for i, label in enumerate(LABELS):
        tp = float(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum())
        fp = float(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = float(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap = _average_precision(y_true[:, i], y_prob[:, i])
        per_label[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(f1, 4), "ap": round(ap, 4) if not np.isnan(ap) else None}
        f1s.append(f1)
        if not np.isnan(ap):
            aps.append(ap)
    return {
        "macro_f1": round(float(np.mean(f1s)), 4),
        "mAP": round(float(np.mean(aps)), 4) if aps else None,
        "per_label": per_label,
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_true, all_prob = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        all_prob.append(torch.sigmoid(logits).cpu().numpy())
        all_true.append(targets.numpy())
    return compute_metrics(np.concatenate(all_true), np.concatenate(all_prob), threshold)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    device = get_device(config.get("device", "auto"))
    logger.info("Training on device=%s seed=%d", device, seed)

    entries = load_json(config.get("annotations_path", "dataset/annotations.json"))
    if len(entries) < 20:
        raise RuntimeError(
            f"Only {len(entries)} samples in annotations — run the generation/crop stages first."
        )
    train_entries, val_entries = split_dataset(entries, float(config.get("val_split", 0.15)), seed)
    logger.info("Dataset: %d train / %d val", len(train_entries), len(val_entries))

    image_size = int(config.get("image_size", 224))
    train_ds = PersonCropDataset(train_entries, image_size, train=True,
                                 horizontal_flip_prob=float(config.get("horizontal_flip_prob", 0.5)))
    val_ds = PersonCropDataset(val_entries, image_size, train=False)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = dict(batch_size=int(config.get("batch_size", 32)),
                         num_workers=int(config.get("num_workers", 2)),
                         worker_init_fn=_seed_worker, generator=generator)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = MultiLabelClassifier(
        backbone=config.get("backbone", "resnet18"),
        pretrained=bool(config.get("pretrained", True)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)

    pos_weight = compute_pos_weight(train_entries).to(device) if config.get("use_pos_weight", True) else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # multi-label loss; NO softmax anywhere
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 3e-4)),
                                  weight_decay=float(config.get("weight_decay", 1e-4)))
    epochs = int(config.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = resolve_path(config.get("checkpoint_dir", "model/checkpoints"))
    threshold = float(config.get("threshold", 0.5))
    history, best_f1 = [], -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        metrics = evaluate(model, val_loader, device, threshold)
        record = {"epoch": epoch, "train_loss": round(epoch_loss / max(1, n_batches), 4),
                  "lr": scheduler.get_last_lr()[0], **metrics}
        history.append(record)
        logger.info("epoch %d | loss %.4f | macro_f1 %.4f | mAP %s",
                    epoch, record["train_loss"], metrics["macro_f1"], metrics["mAP"])

        save_checkpoint(model, checkpoint_dir / "last.pt", config, {"epoch": epoch, "metrics": metrics})
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            save_checkpoint(model, checkpoint_dir / "best.pt", config, {"epoch": epoch, "metrics": metrics})
            logger.info("New best macro_f1=%.4f -> %s", best_f1, checkpoint_dir / "best.pt")

    log = {
        "config": config,
        "n_train": len(train_entries),
        "n_val": len(val_entries),
        "best_macro_f1": best_f1,
        "history": history,
    }
    save_json(log, config.get("log_path", "model/training_log.json"))
    return log


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the multi-label classifier")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    train(config)


if __name__ == "__main__":
    main()
