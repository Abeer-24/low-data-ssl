"""
Export a trained model to ONNX -- decouples the served model from the
training code/environment, per the deployment plan (Section 8 of
PROJECT_DOCUMENTATION.md).

Handles two cases:
1. Classifier strategies (Baseline, Augmented, ImageNet Transfer): the
   saved checkpoint is a full CNN with a 10-class head -- export it
   directly.
2. SimSiam: the saved checkpoint is an ENCODER ONLY (no classifier head --
   see training/linear_probe.py, which saves the encoder and the linear
   classifier's raw weights separately, since a scikit-learn
   LogisticRegression isn't an ONNX-native object). This script exports
   just the encoder; the linear classifier is applied manually with numpy
   at inference time (softmax(W @ features + b)) in the served app.

Verifies every export two ways:
1. Structural validity via onnx.checker (catches malformed graphs).
2. Numerical equivalence -- runs the same input through both the original
   PyTorch model and the exported ONNX model, checks outputs match within
   floating-point tolerance. Requires onnxruntime (pip install onnxruntime
   if not already installed -- it's a separate package from onnx).

Usage:
    python export/to_onnx.py --backbone mobilenet_v2 --strategy imagenet_transfer
    python export/to_onnx.py --backbone resnet18 --strategy simsiam
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import onnx
from torchvision.models import resnet18, mobilenet_v2, efficientnet_b0

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

DEPLOY_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "deploy")
NUM_CLASSES = 10

CLASSIFIER_STRATEGIES = ["baseline", "augmented", "imagenet_transfer"]


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
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return model


def build_encoder_only(backbone_name: str):
    """Same architectures as above, but with the classifier head removed --
    used for SimSiam, whose checkpoint is an encoder only (see
    training/linear_probe.py)."""
    if backbone_name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Identity()
    elif backbone_name == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
        model.classifier = nn.Identity()
    elif backbone_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return model


def load_model(backbone_name: str, strategy: str, device):
    if strategy in CLASSIFIER_STRATEGIES:
        checkpoint_path = os.path.join(
            DEPLOY_CHECKPOINT_DIR, f"{backbone_name}_{strategy}_deploy.pt"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No deployment checkpoint found at {checkpoint_path}. "
                f"Run training/{strategy}.py --backbone {backbone_name} first "
                f"-- it saves this checkpoint automatically at the 100% "
                f"label percentage."
            )
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_classifier_model(backbone_name)
        model.load_state_dict(ckpt["model_state"])

    elif strategy == "simsiam":
        checkpoint_path = os.path.join(
            DEPLOY_CHECKPOINT_DIR, f"{backbone_name}_simsiam_encoder_deploy.pt"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No SimSiam encoder checkpoint found at {checkpoint_path}. "
                f"Run training/linear_probe.py --backbone {backbone_name} "
                f"first -- it saves this checkpoint automatically at the "
                f"100% label percentage. Note: this exports the encoder "
                f"only; the linear classifier weights "
                f"({backbone_name}_simsiam_linear_weights.npz) are applied "
                f"separately with numpy at inference time, not part of "
                f"this ONNX graph."
            )
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_encoder_only(backbone_name)
        model.load_state_dict(ckpt["model_state"])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: backbone={backbone_name}, strategy={strategy}")
    return model


def export_to_onnx(model, backbone_name: str, strategy: str, image_size=96):
    os.makedirs(DEPLOY_CHECKPOINT_DIR, exist_ok=True)
    suffix = "encoder" if strategy == "simsiam" else "classifier"
    onnx_path = os.path.join(
        DEPLOY_CHECKPOINT_DIR, f"{backbone_name}_{strategy}_{suffix}.onnx"
    )

    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )

    print(f"Exported to: {onnx_path}")
    return onnx_path, dummy_input


def verify_structural(onnx_path: str):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("Structural check passed: ONNX graph is well-formed.")


def verify_numerical(model, onnx_path: str, dummy_input, atol=1e-4):
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "\nWARNING: onnxruntime not installed -- skipping numerical "
            "verification. This only confirmed the ONNX file is "
            "structurally valid, NOT that it produces the same predictions "
            "as the original model. Install with: pip install onnxruntime"
        )
        return

    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

    max_diff = np.abs(torch_output - onnx_output).max()
    if max_diff < atol:
        print(f"Numerical check passed: max difference = {max_diff:.2e} "
              f"(within tolerance {atol:.0e})")
    else:
        print(f"WARNING: max difference = {max_diff:.2e} exceeds tolerance "
              f"{atol:.0e} -- the ONNX export may not be equivalent to the "
              f"original model. Do not deploy without investigating this.")


def run_export(backbone_name: str, strategy: str):
    device = torch.device("cpu")  # export on CPU for portability
    model = load_model(backbone_name, strategy, device)

    onnx_path, dummy_input = export_to_onnx(model, backbone_name, strategy)
    verify_structural(onnx_path)
    verify_numerical(model, onnx_path, dummy_input)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nONNX file size: {size_mb:.2f} MB")

    if strategy == "simsiam":
        weights_path = os.path.join(
            DEPLOY_CHECKPOINT_DIR, f"{backbone_name}_simsiam_linear_weights.npz"
        )
        print(f"Remember: the app also needs {weights_path} "
              f"(saved by training/linear_probe.py) to apply the linear "
              f"classifier on top of this encoder's output.")

    print("\nNext: use this .onnx file in the app for serving, instead of "
          "loading the raw PyTorch checkpoint -- decouples the deployed "
          "app from the training environment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet18", "mobilenet_v2", "efficientnet_b0"], default="mobilenet_v2")
    parser.add_argument("--strategy", choices=["baseline", "augmented", "imagenet_transfer", "simsiam"], default="imagenet_transfer")
    args = parser.parse_args()

    run_export(args.backbone, args.strategy)
