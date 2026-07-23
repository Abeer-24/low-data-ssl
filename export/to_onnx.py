"""
Export the deployed model (MobileNetV2 + ImageNet transfer, see
PROJECT_DOCUMENTATION.md Section 11.6) to ONNX -- decouples the served
model from the training code/environment, per the deployment plan
(Section 8).

Verifies the export two ways:
1. Structural validity via onnx.checker (catches malformed graphs).
2. Numerical equivalence -- runs the same input through both the original
   PyTorch model and the exported ONNX model, checks outputs match within
   floating-point tolerance. Requires onnxruntime (pip install onnxruntime
   if not already installed -- it's a separate package from onnx).

Usage:
    python export/to_onnx.py --backbone mobilenet_v2
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import onnx
from torchvision.models import resnet18, mobilenet_v2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

DEPLOY_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "deploy")
NUM_CLASSES = 10


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


def export_to_onnx(model, backbone_name: str, image_size=96):
    os.makedirs(DEPLOY_CHECKPOINT_DIR, exist_ok=True)
    onnx_path = os.path.join(DEPLOY_CHECKPOINT_DIR, f"{backbone_name}_deploy.onnx")

    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},   # allow variable batch size
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


def run_export(backbone_name: str):
    device = torch.device("cpu")  # export on CPU for portability
    model = load_deployed_model(backbone_name, device)

    onnx_path, dummy_input = export_to_onnx(model, backbone_name)
    verify_structural(onnx_path)
    verify_numerical(model, onnx_path, dummy_input)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nONNX file size: {size_mb:.2f} MB")
    print("Next: use this .onnx file in app/gradio_app.py for serving, "
          "instead of loading the raw PyTorch checkpoint -- decouples the "
          "deployed app from the training environment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet18", "mobilenet_v2"], default="mobilenet_v2")
    args = parser.parse_args()

    run_export(args.backbone)
