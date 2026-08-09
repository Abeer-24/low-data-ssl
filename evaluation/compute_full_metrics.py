"""
Computes precision, recall, and F1 (macro-averaged) for every already-
trained backbone x strategy combination, using the saved checkpoints --
no retraining needed, since all 12 models already exist on disk.

Only accuracy was computed and saved during the original training runs
(training/baseline.py, augmented.py, imagenet_transfer.py, linear_probe.py).
This script fills that gap for the Dashboard's radar chart and comparison
table.

Usage:
    python evaluation/compute_full_metrics.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, mobilenet_v2, efficientnet_b0
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from stl10_loader import get_stl10_splits, DATA_ROOT

DEPLOY_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "deploy")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "downstream")

BACKBONES = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
STRATEGIES = ["baseline", "augmented", "imagenet_transfer", "simsiam"]
NUM_CLASSES = 10


def build_classifier_model(backbone_name: str):
    if backbone_name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif backbone_name == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif backbone_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def build_encoder_only(backbone_name: str):
    if backbone_name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Identity()
    elif backbone_name == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
        model.classifier = nn.Identity()
    elif backbone_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier = nn.Identity()
    return model


def get_eval_transform():
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


@torch.no_grad()
def get_predictions_classifier(backbone, strategy, test_loader, device):
    path = os.path.join(DEPLOY_DIR, f"{backbone}_{strategy}_deploy.pt")
    if not os.path.exists(path):
        return None, None

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_classifier_model(backbone).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def get_predictions_simsiam(backbone, test_loader, device):
    encoder_path = os.path.join(DEPLOY_DIR, f"{backbone}_simsiam_encoder_deploy.pt")
    weights_path = os.path.join(DEPLOY_DIR, f"{backbone}_simsiam_linear_weights.npz")
    if not os.path.exists(encoder_path) or not os.path.exists(weights_path):
        return None, None

    ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder = build_encoder_only(backbone).to(device)
    encoder.load_state_dict(ckpt["model_state"])
    encoder.eval()

    weights = np.load(weights_path)
    coef, intercept = weights["coef"], weights["intercept"]

    all_preds, all_labels = [], []
    for images, labels in test_loader:
        images = images.to(device)
        features = encoder(images).cpu().numpy()
        logits = features @ coef.T + intercept
        preds = logits.argmax(axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_full_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading STL-10 test split...")
    _, _, test = get_stl10_splits(root=DATA_ROOT)
    test_loader = DataLoader(
        TransformWrapper(test, get_eval_transform()),
        batch_size=128, shuffle=False, num_workers=0,
    )

    results = {}
    for backbone in BACKBONES:
        results[backbone] = {}
        for strategy in STRATEGIES:
            print(f"\n--- {backbone} + {strategy} ---")
            if strategy == "simsiam":
                preds, labels = get_predictions_simsiam(backbone, test_loader, device)
            else:
                preds, labels = get_predictions_classifier(backbone, strategy, test_loader, device)

            if preds is None:
                print("  Skipped -- no saved checkpoint found.")
                continue

            accuracy = accuracy_score(labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )

            results[backbone][strategy] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
            print(f"  accuracy={accuracy:.4f} precision={precision:.4f} "
                  f"recall={recall:.4f} f1={f1:.4f}")

    output_path = os.path.join(RESULTS_DIR, "full_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print("Next: copy this file into render_deploy/results/ for the Dashboard's radar chart.")


if __name__ == "__main__":
    compute_full_metrics()
