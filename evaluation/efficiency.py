"""
Efficiency profiling: parameters, model size (MB), inference time, and FPS
for each backbone. This is architecture-dependent only -- ResNet18 has the
same param count/size/speed regardless of which strategy (Baseline,
Augmented, ImageNet Transfer, SimSiam) trained it, since the strategies
differ in *how* the weights were learned, not the network structure.

This directly answers "which model would you actually deploy?" --
Section 6 of PROJECT_DOCUMENTATION.md.

Usage:
    python evaluation/efficiency.py
"""

import os
import time
import json
import torch
import torch.nn as nn
from torchvision.models import resnet18, mobilenet_v2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "downstream")
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    # Save state dict to a temp file and measure actual bytes on disk --
    # more accurate than estimating from param count x dtype size, since it
    # reflects what a real checkpoint file would weigh.
    temp_path = os.path.join(os.path.dirname(__file__), "_temp_size_check.pt")
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_mb


@torch.no_grad()
def measure_inference_time(model, device, image_size=96, num_warmup=10, num_runs=100):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    # Warmup -- first few forward passes on GPU include CUDA kernel
    # compilation/caching overhead that isn't representative of steady-state
    # inference speed, so we discard these.
    for _ in range(num_warmup):
        _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    avg_time_ms = (elapsed / num_runs) * 1000
    fps = num_runs / elapsed
    return avg_time_ms, fps


def run_efficiency_profile():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}\n")

    backbones = ["resnet18", "mobilenet_v2"]
    results = {}

    for backbone_name in backbones:
        model = build_model(backbone_name)

        params = count_params(model)
        size_mb = model_size_mb(model)
        inference_ms, fps = measure_inference_time(model, device)

        results[backbone_name] = {
            "params": params,
            "params_millions": round(params / 1e6, 2),
            "size_mb": round(size_mb, 2),
            "inference_time_ms": round(inference_ms, 2),
            "fps": round(fps, 1),
        }

        print(f"{backbone_name}:")
        print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
        print(f"  Model size: {size_mb:.2f} MB")
        print(f"  Inference time: {inference_ms:.2f} ms/image")
        print(f"  Throughput: {fps:.1f} FPS")
        print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "efficiency.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print("\nSummary table:")
    print(f"{'Model':<15} {'Params':<10} {'Size (MB)':<12} {'Inference (ms)':<16} {'FPS':<8}")
    for name, r in results.items():
        print(f"{name:<15} {r['params_millions']:<10} {r['size_mb']:<12} "
              f"{r['inference_time_ms']:<16} {r['fps']:<8}")

    print(
        "\nNote: these numbers are architecture-dependent only -- identical "
        "regardless of which strategy (Baseline/Augmented/ImageNet/SimSiam) "
        "trained the weights, since the network structure doesn't change."
    )


if __name__ == "__main__":
    run_efficiency_profile()
