"""
STL-10 data loader with fixed-seed label-percentage subsampling.

This module:
1. Downloads STL-10 (unlabeled + labeled train/test splits) via torchvision.
2. Creates fixed subsets of the labeled train split at 1/5/10/20/50/100%,
   using `subsample_split_seed` from configs/seeds.yaml so every downstream
   script (baseline, augmented, imagenet_transfer, linear_probe) sees the
   exact same images at each percentage.

Run this file standalone first and confirm the printed split sizes look
right (e.g. 1% of 5000 ~= 50 images) before writing any training code that
depends on it.
"""

import os
import yaml
import numpy as np
from torchvision.datasets import STL10
from torch.utils.data import Subset

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "seeds.yaml")
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "raw_data")


def load_seed_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def get_stl10_splits(root: str = DATA_ROOT):
    """
    Downloads (if needed) and returns the three STL-10 splits:
      - unlabeled: 100,000 images, no labels (used for SimSiam pretraining)
      - train:     5,000 labeled images (subsampled downstream)
      - test:      8,000 labeled images (final evaluation, never subsampled)
    """
    os.makedirs(root, exist_ok=True)

    unlabeled = STL10(root=root, split="unlabeled", download=True)
    train = STL10(root=root, split="train", download=True)
    test = STL10(root=root, split="test", download=True)

    return unlabeled, train, test


# ---------------------------------------------------------------------------
# Fixed label-percentage subsampling
# ---------------------------------------------------------------------------

def make_label_percentage_subsets(train_dataset, label_percentages, split_seed):
    """
    Returns a dict mapping each label percentage -> a torch Subset of
    train_dataset, sampled once with a fixed seed so every strategy and
    every seed run downstream sees the identical set of images at a given
    percentage. This keeps label-percentage subsampling noise separate from
    training/init noise (see docs/PROJECT_DOCUMENTATION.md, Limitations).

    Sampling is stratified by class so small percentages don't accidentally
    drop a class entirely.
    """
    labels = np.array(train_dataset.labels)
    num_classes = len(np.unique(labels))
    rng = np.random.RandomState(split_seed)

    subsets = {}
    for pct in label_percentages:
        n_total = int(len(train_dataset) * pct / 100)
        n_per_class = max(1, n_total // num_classes)

        selected_indices = []
        for cls in range(num_classes):
            cls_indices = np.where(labels == cls)[0]
            rng.shuffle(cls_indices)
            selected_indices.extend(cls_indices[:n_per_class].tolist())

        subsets[pct] = Subset(train_dataset, selected_indices)

    return subsets


# ---------------------------------------------------------------------------
# Standalone check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = load_seed_config()
    label_percentages = config["label_percentages"]
    split_seed = config["subsample_split_seed"]

    print(f"Downloading STL-10 to: {os.path.abspath(DATA_ROOT)}")
    unlabeled, train, test = get_stl10_splits()

    print(f"\nSplit sizes:")
    print(f"  Unlabeled: {len(unlabeled)} images")
    print(f"  Train (full labeled): {len(train)} images")
    print(f"  Test: {len(test)} images")

    subsets = make_label_percentage_subsets(train, label_percentages, split_seed)

    print(f"\nLabel-percentage subsets (split_seed={split_seed}):")
    for pct, subset in subsets.items():
        expected = int(len(train) * pct / 100)
        print(f"  {pct:>3}% -> {len(subset):>5} images (expected ~{expected})")

    print(
        "\nIf these numbers look wrong (e.g. 1% giving 0 images for any "
        "class, or counts far off from 'expected'), stop here and fix this "
        "file before writing any training script that depends on it."
    )
