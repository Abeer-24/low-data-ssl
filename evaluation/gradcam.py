"""
Grad-CAM visualization for the deployed model (MobileNetV2 + ImageNet
transfer, trained on 100% labels -- see PROJECT_DOCUMENTATION.md, Section
11.6). Produces one figure: original image -> predicted class -> heatmap
highlighting the regions the model used to make that prediction.

Requires training/imagenet_transfer.py to have been run first (it saves
the deployment checkpoint to checkpoints/deploy/).

Usage:
    python evaluation/gradcam.py --backbone mobilenet_v2 --num_images 4
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18, mobilenet_v2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from stl10_loader import get_stl10_splits, DATA_ROOT

DEPLOY_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "deploy")
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "figures")

NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


def build_model(backbone_name: str):
    if backbone_name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif backbone_name == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return model


def get_target_layer(model, backbone_name: str):
    # Grad-CAM needs a convolutional layer with spatial resolution still
    # intact. The final layer of each backbone (features[-1] / layer4[-1])
    # was designed for 224x224 ImageNet inputs, where it still has a 7x7
    # grid -- but STL-10 images are 96x96, so that same layer collapses to
    # roughly 3x3, producing smooth, near-content-independent-looking
    # heatmaps after upsampling. Using an earlier layer (stride 16 instead
    # of 32) gives a 6x6 grid instead -- still coarse, but meaningfully
    # more localized, at a small cost in semantic depth.
    if backbone_name == "resnet18":
        return model.layer3[-1]
    elif backbone_name == "mobilenet_v2":
        return model.features[13]
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


def load_deployed_model(backbone_name: str, device):
    checkpoint_path = os.path.join(
        DEPLOY_CHECKPOINT_DIR, f"{backbone_name}_imagenet_transfer_deploy.pt"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No deployment checkpoint found at {checkpoint_path}. "
            f"Run training/imagenet_transfer.py --backbone {backbone_name} "
            f"first -- it saves this checkpoint automatically at the 100% "
            f"label percentage."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(backbone_name)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"Loaded deployment checkpoint: {backbone_name}, "
          f"strategy={ckpt['strategy']}, label_pct={ckpt['label_pct']}, "
          f"seed={ckpt['seed']}")
    return model


def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_gradcam(backbone_name: str, num_images: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_deployed_model(backbone_name, device)
    target_layer = get_target_layer(model, backbone_name)

    cam = GradCAM(model=model, target_layers=[target_layer])

    print("Loading STL-10 test split...")
    _, _, test = get_stl10_splits(root=DATA_ROOT)

    transform = get_eval_transform()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))

    for i in range(num_images):
        pil_img, true_label = test[i]
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_label = output.argmax(dim=1).item()

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(pred_label)],
        )[0]

        # Normalize the original image to [0, 1] for overlay (undo the
        # ImageNet normalization that was applied for the model's input).
        rgb_img = np.array(pil_img).astype(np.float32) / 255.0
        cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(
            f"True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]}"
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(cam_overlay)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis("off")

    plt.tight_layout()
    figure_path = os.path.join(FIGURE_DIR, f"{backbone_name}_gradcam.png")
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {figure_path}")
    print(
        "This is a static figure for the README/documentation, not a live "
        "feature -- see PROJECT_DOCUMENTATION.md, Section 8, for why "
        "Grad-CAM is excluded from the deployed Gradio app."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet18", "mobilenet_v2"], default="mobilenet_v2")
    parser.add_argument("--num_images", type=int, default=4)
    args = parser.parse_args()

    run_gradcam(args.backbone, args.num_images)
