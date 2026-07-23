# Self-Supervised Learning in the Low-Data Regime

Comparing self-supervised pretraining (SimSiam) against supervised
baselines, data augmentation, and ImageNet transfer learning — across two
CNN backbones and six labeled-data percentages (1% to 100%) — to find out
when SSL pretraining is actually worth the extra compute.

Full methodology, dataset details, seed policy, and limitations are in
[`PROJECT_DOCUMENTATION.md`](./PROJECT_DOCUMENTATION.md).

## Key finding

SimSiam only beats ImageNet transfer learning in the extreme low-data
regime (≤10% labels) — and even there, the margin narrows fast. Past 10%
labels, transfer learning wins, and its lead grows. The best-performing
combination overall isn't the elaborate SSL pipeline — it's the smaller
backbone (MobileNetV2) fine-tuned from free ImageNet weights.

| Label % | ResNet18 Baseline | ResNet18 SimSiam | MobileNetV2 ImageNet |
|---|---|---|---|
| 1% | 20.7% | **56.7%** | 57.3% |
| 10% | 36.3% | 70.1% | **74.6%** |
| 100% | 62.5% | 76.1% | **87.7%** |

Full 2-backbone × 4-strategy comparison table in
[`PROJECT_DOCUMENTATION.md`, Section 11](./PROJECT_DOCUMENTATION.md#11-results-both-backbones).

## Efficiency

| Model | Params | Size (MB) | Inference Time | FPS |
|---|---|---|---|---|
| ResNet18 | 11.18M | 42.73 MB | 4.04 ms | 247.3 |
| MobileNetV2 | 2.24M | **8.76 MB** | 8.25 ms | 121.2 |

MobileNetV2 is ~5x smaller but roughly **2x slower** on this study's GPU
(RTX 3050) — depthwise separable convolutions parallelize less efficiently
on GPU hardware than ResNet18's standard convolutions at batch size 1,
despite having far fewer parameters. The deployed model (MobileNetV2 +
ImageNet transfer) was chosen for accuracy and size, not GPU speed — see
[Section 11.6](./PROJECT_DOCUMENTATION.md#116-deployment-decision) for the
full reasoning.

## Dataset

[STL-10](https://ai.stanford.edu/~acoates/stl10/) — 100,000 unlabeled
images for pretraining, 5,000 labeled images (subsampled to 1-100%) for
downstream training, 8,000 labeled images for evaluation. 96×96 resolution,
10 classes.

## Strategies compared

- **Baseline** — supervised, trained from scratch
- **Data Augmentation** — baseline + random crop/flip/color jitter
- **ImageNet Transfer Learning** — ImageNet-pretrained backbone, fine-tuned
- **Self-Supervised (SimSiam)** — pretrained on unlabeled STL-10, linear
  probe on labeled subset

## Backbones

ResNet18 (11.18M params), MobileNetV2 (2.24M params) -- see
[Efficiency](#efficiency) above for exact measurements.

## Project structure

```
LowDataSSL/
├── data/stl10_loader.py          # dataset download + label-percentage splits
├── training/
│   ├── simsiam.py                 # SSL pretraining
│   ├── baseline.py                 # supervised from scratch
│   ├── augmented.py                 # supervised + augmentation
│   ├── imagenet_transfer.py          # ImageNet fine-tuning
│   └── linear_probe.py                # frozen SSL embeddings + logistic regression
├── configs/
│   ├── seeds.yaml                      # single source of truth for all seeds
│   └── wandb_config.yaml                # experiment tracking config
├── checkpoints/downstream/                # accuracy results (JSON)
└── PROJECT_DOCUMENTATION.md                # full writeup
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn wandb pyyaml matplotlib grad-cam onnx gradio
```

## Usage

```bash
# 1. Download data and verify label-percentage splits
python data/stl10_loader.py

# 2. Pretrain the SSL encoder (slow -- hours on a single GPU)
python training/simsiam.py --backbone resnet18 --epochs 100

# 3. Evaluate + compare (fast -- minutes each)
python training/linear_probe.py --backbone resnet18
python training/baseline.py --backbone resnet18
python training/augmented.py --backbone resnet18
python training/imagenet_transfer.py --backbone resnet18
```

Repeat steps 2-3 with `--backbone mobilenet_v2` for the second backbone.

## Hardware used

RTX 3050 Laptop GPU (4GB VRAM), Ryzen 5 5000-series, 16GB RAM. SimSiam
pretraining run once per backbone (75-80 epochs) due to compute
constraints; all downstream evaluation repeated across 3 seeds. See
[Limitations](./PROJECT_DOCUMENTATION.md#12-limitations) for details.

## License

MIT
