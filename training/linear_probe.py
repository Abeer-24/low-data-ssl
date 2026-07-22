"""
Linear probe evaluation for a frozen SimSiam-pretrained encoder.

Loads a SimSiam checkpoint, freezes the encoder, extracts embeddings for
each label-percentage subset (1/5/10/20/50/100%) and the full test set, then
trains a logistic regression classifier on top -- repeated across 3 seeds
per label percentage (see configs/seeds.yaml, downstream_seeds).

This is the actual test of whether pretraining was worthwhile: SimSiam's
training loss plateauing does NOT by itself prove the embeddings are good.
This script produces the real evidence -- accuracy vs label percentage.

Usage:
    python training/linear_probe.py --backbone resnet18
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, mobilenet_v2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from stl10_loader import (
    get_stl10_splits, load_seed_config, make_label_percentage_subsets, DATA_ROOT
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "simsiam")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "downstream")


# ---------------------------------------------------------------------------
# Encoder loading (matches training/simsiam.py's build_backbone)
# ---------------------------------------------------------------------------

def build_backbone(name: str):
    if name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Identity()
    elif name == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return model


def load_frozen_encoder(backbone_name: str, device):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{backbone_name}_simsiam.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No SimSiam checkpoint found at {checkpoint_path}. "
            f"Run training/simsiam.py --backbone {backbone_name} first."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = build_backbone(backbone_name)

    # The saved state dict includes encoder + projector + predictor (full
    # SimSiam model). We only want the encoder.* weights.
    full_state = ckpt["model_state"]
    encoder_state = {
        k.replace("encoder.", "", 1): v
        for k, v in full_state.items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state)
    encoder.to(device)
    encoder.eval()

    print(f"Loaded encoder from checkpoint at epoch {ckpt['epoch'] + 1}")
    return encoder


# ---------------------------------------------------------------------------
# Feature extraction (plain eval transform, no SimSiam-style augmentation --
# we want one stable embedding per image, not augmented views)
# ---------------------------------------------------------------------------

def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class EvalWrapper(torch.utils.data.Dataset):
    """Applies the plain eval transform to a base dataset or Subset."""

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return self.transform(img), label


@torch.no_grad()
def extract_features(dataset, encoder, device, batch_size=128):
    wrapped = EvalWrapper(dataset, get_eval_transform())
    loader = DataLoader(wrapped, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        features = encoder(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_linear_probe(backbone_name: str):
    seed_config = load_seed_config()
    downstream_seeds = seed_config["downstream_seeds"]
    label_percentages = seed_config["label_percentages"]
    split_seed = seed_config["subsample_split_seed"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_frozen_encoder(backbone_name, device)

    print("Loading STL-10 splits...")
    _, train, test = get_stl10_splits(root=DATA_ROOT)

    print("Extracting features for the full test set (used for every run)...")
    test_features, test_labels = extract_features(test, encoder, device)

    subsets_by_seed = {
        seed: make_label_percentage_subsets(train, label_percentages, split_seed=seed)
        for seed in downstream_seeds
    }

    results = {}
    for pct in label_percentages:
        print(f"\n--- Label percentage: {pct}% ---")

        accuracies = []
        for seed in downstream_seeds:
            subset = subsets_by_seed[seed][pct]
            train_features, train_labels = extract_features(subset, encoder, device)

            clf = LogisticRegression(max_iter=2000, random_state=seed)
            clf.fit(train_features, train_labels)
            preds = clf.predict(test_features)
            acc = accuracy_score(test_labels, preds)
            accuracies.append(acc)
            print(f"  seed={seed} ({len(subset)} images): accuracy={acc:.4f}")

        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        print(f"  -> {mean_acc*100:.1f} ± {std_acc*100:.1f}%")

        results[pct] = {
            "accuracies": accuracies,
            "mean": mean_acc,
            "std": std_acc,
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"{backbone_name}_simsiam_linearprobe.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("\nSummary (accuracy mean ± std across 3 seeds):")
    for pct, r in results.items():
        print(f"  {pct:>3}% labels -> {r['mean']*100:.1f} ± {r['std']*100:.1f}%")

    print(
        "\nNext: run training/baseline.py and training/augmented.py with the "
        "same backbone to get comparison numbers -- this SSL result alone "
        "doesn't prove SimSiam was worthwhile until compared against those."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet18", "mobilenet_v2"], default="resnet18")
    args = parser.parse_args()

    run_linear_probe(args.backbone)
