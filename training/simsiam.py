"""
SimSiam self-supervised pretraining.

Trains an encoder (ResNet18 by default) on STL-10's unlabeled split using
SimSiam: two augmented views per image, a shared encoder + projection head,
a predictor head on one branch, and a stop-gradient on the other. No
negative pairs, no large-batch requirement -- this is what makes it
tractable on a 4GB-VRAM laptop GPU.

Uses a SINGLE seed (pretraining_seed in configs/seeds.yaml) -- this stage is
the expensive one, and is documented as single-seed in
docs/PROJECT_DOCUMENTATION.md, Section 11 (Limitations).

Usage:
    python training/simsiam.py --backbone resnet18 --epochs 100
    python training/simsiam.py --backbone mobilenet_v2 --epochs 100 --resume
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, mobilenet_v2

import wandb

# Make the sibling data/ folder importable regardless of what directory
# this script is run from.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from stl10_loader import get_stl10_splits, load_seed_config, DATA_ROOT

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "simsiam")
WANDB_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "wandb_config.yaml")


# ---------------------------------------------------------------------------
# SimSiam augmentations (two independent views per image)
# ---------------------------------------------------------------------------

def get_simsiam_transform(image_size=96):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TwoViewDataset(torch.utils.data.Dataset):
    """Wraps a dataset to return two independently augmented views of each image."""

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]  # label unused -- this is unlabeled data
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


# ---------------------------------------------------------------------------
# SimSiam model: encoder + projection head + predictor head
# ---------------------------------------------------------------------------

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x):
        return self.net(x)


class PredictionMLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_backbone(name: str):
    """Returns (backbone_module, feature_dim) with the classifier head removed."""
    if name == "resnet18":
        model = resnet18(weights=None)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return model, feature_dim


class SimSiam(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.encoder, feature_dim = build_backbone(backbone_name)
        self.projector = ProjectionMLP(feature_dim)
        self.predictor = PredictionMLP()

    def forward(self, x1, x2):
        f1, f2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = self.projector(f1), self.projector(f2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def negative_cosine_similarity(p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=-1).mean()


def simsiam_loss(p1, p2, z1, z2):
    # Symmetrized loss with stop-gradient already applied to z1, z2
    return 0.5 * negative_cosine_similarity(p1, z2) + 0.5 * negative_cosine_similarity(p2, z1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(backbone_name: str, epochs: int, batch_size: int, resume: bool):
    seed_config = load_seed_config()
    seed = seed_config["pretraining_seed"]
    torch.manual_seed(seed)

    with open(WANDB_CONFIG_PATH, "r") as f:
        wandb_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available -- this will be extremely slow. "
              "Stop and check your torch install before continuing.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{backbone_name}_simsiam.pt")

    unlabeled, _, _ = get_stl10_splits(root=DATA_ROOT)
    dataset = TwoViewDataset(unlabeled, get_simsiam_transform())
    # num_workers=0 is safer on Windows to avoid multiprocessing issues;
    # increase if you confirm higher values work on your machine.
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    model = SimSiam(backbone_name).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    run_name = wandb_config["run_name_template"].format(
        backbone=backbone_name, strategy="simsiam_pretrain", label_pct="NA", seed=seed
    )
    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config.get("entity"),
        name=run_name,
        tags=wandb_config["default_tags"] + ["pretraining"],
        resume="allow",
    )

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        total_loss = 0.0

        for view1, view2 in loader:
            view1, view2 = view1.to(device, non_blocking=True), view2.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                p1, p2, z1, z2 = model(view1, view2)
                loss = simsiam_loss(p1, p2, z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        epoch_duration = time.time() - epoch_start
        gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0

        print(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | "
              f"time={epoch_duration:.1f}s | gpu_mem={gpu_mem_mb:.0f}MB")

        wandb.log({
            "training_loss": avg_loss,
            "epoch_duration_sec": epoch_duration,
            "gpu_memory_mb": gpu_mem_mb,
            "epoch": epoch,
        })

        # Checkpoint every 5 epochs -- critical for surviving interrupted
        # overnight runs on a laptop (sleep, thermal throttle, etc.)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "backbone": backbone_name,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    wandb.finish()
    print(f"\nDone. Final checkpoint: {checkpoint_path}")
    print("Next: check the W&B loss curve. If it's still dropping steadily, "
          "re-run with --resume and a higher --epochs value. If it's "
          "plateaued, move on to training/linear_probe.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet18", "mobilenet_v2"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(args.backbone, args.epochs, args.batch_size, args.resume)
