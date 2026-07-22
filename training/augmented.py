"""
Data Augmentation strategy: supervised CNN trained from scratch on the
labeled subset, same as training/baseline.py, but with random crop, flip,
and color jitter applied during training.

Purpose: isolates how much of SimSiam's advantage (training/linear_probe.py)
is really about self-supervised pretraining versus just needing better
regularization on tiny datasets. If Augmented closes most of the gap to
SimSiam, that's an important, honest finding -- report it, don't bury it.

Seed policy: matches training/baseline.py -- 3 different random
label-percentage subsamples per percentage, one model trained per subsample.

Usage:
    python training/augmented.py --backbone resnet18
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, mobilenet_v2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from stl10_loader import get_stl10_splits, load_seed_config, make_label_percentage_subsets, DATA_ROOT

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "downstream")

NUM_CLASSES = 10  # STL-10


# ---------------------------------------------------------------------------
# Model (identical to training/baseline.py)
# ---------------------------------------------------------------------------

def build_model(backbone_name: str, num_classes: int = NUM_CLASSES):
    if backbone_name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return model


# ---------------------------------------------------------------------------
# Transforms -- this is the only real difference from training/baseline.py
# ---------------------------------------------------------------------------

def get_train_transform(image_size=96):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform():
    # No augmentation at test time -- standard practice.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Training + evaluation for one (percentage, seed) combination
# ---------------------------------------------------------------------------

def train_and_evaluate(subset, test_dataset, backbone_name, seed, device,
                        epochs=30, batch_size=64, lr=0.001):
    torch.manual_seed(seed)

    model = build_model(backbone_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TransformWrapper(subset, get_train_transform()),
        batch_size=min(batch_size, len(subset)), shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        TransformWrapper(test_dataset, get_eval_transform()),
        batch_size=128, shuffle=False, num_workers=0,
    )

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_augmented(backbone_name: str):
    seed_config = load_seed_config()
    downstream_seeds = seed_config["downstream_seeds"]
    label_percentages = seed_config["label_percentages"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading STL-10 splits...")
    _, train, test = get_stl10_splits(root=DATA_ROOT)

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
            start = time.time()
            acc = train_and_evaluate(subset, test, backbone_name, seed, device)
            duration = time.time() - start
            accuracies.append(acc)
            print(f"  seed={seed} ({len(subset)} images): accuracy={acc:.4f} ({duration:.1f}s)")

        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        print(f"  -> {mean_acc*100:.1f} ± {std_acc*100:.1f}%")

        results[pct] = {"accuracies": accuracies, "mean": mean_acc, "std": std_acc}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"{backbone_name}_augmented.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("\nSummary (accuracy mean ± std across 3 seeds):")
    for pct, r in results.items():
        print(f"  {pct:>3}% labels -> {r['mean']*100:.1f} ± {r['std']*100:.1f}%")

    print(
        "\nCompare against checkpoints/downstream/{backbone}_baseline.json and "
        "{backbone}_simsiam_linearprobe.json -- if Augmented closes most of "
        "the gap to SimSiam, that's a real finding worth reporting, not a "
        "reason to hide this result."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet18", "mobilenet_v2"], default="resnet18")
    args = parser.parse_args()

    run_augmented(args.backbone)
