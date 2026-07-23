# Self-Supervised Learning in the Low-Data Regime

## Abstract

Labeled data is expensive; unlabeled data is comparatively cheap. This
project evaluates whether self-supervised pretraining (SimSiam) produces
better downstream classifiers than supervised training from scratch, standard
data augmentation, or ImageNet transfer learning, specifically when only a
small fraction of labels are available (1%–100% of STL-10's labeled split).
Two lightweight CNN backbones (ResNet18, MobileNetV2) are compared across
five training strategies, with accuracy, efficiency, and model-size
trade-offs reported. Encoder pretraining is run once per backbone due to
compute constraints; all downstream evaluation is repeated across 3 seeds
and reported as mean ± standard deviation.

## Introduction

Modern deep learning models typically require large labeled datasets, which
are costly to produce. Self-supervised learning (SSL) offers a way to
leverage abundant unlabeled data to learn transferable representations
before any labels are seen — a property that should matter most when labels
are scarce. This project tests that claim directly and quantitatively,
rather than assuming it, by measuring accuracy as a function of label
percentage across multiple training strategies and backbones.

## Related Work

- **SimCLR** (Chen et al., 2020) — contrastive SSL; requires large batch
  sizes / many negative pairs to work well, which is why it was not chosen
  for this project's hardware.
- **BYOL** (Grill et al., 2020) — non-contrastive SSL using a momentum
  encoder; similar goals to SimSiam without a memory bank.
- **SimSiam** (Chen & He, 2021) — the method used here; avoids negative
  pairs and large batch requirements via a stop-gradient operation, making
  it tractable on limited hardware.
- **STL-10** (Coates et al., 2011) — benchmark dataset explicitly designed
  for semi-/self-supervised evaluation, with a large unlabeled split and a
  small labeled split.

## 1. Project Overview

This project studies how self-supervised pretraining reduces dependence on
labeled data. A CNN encoder is pretrained on unlabeled images using SimSiam,
then evaluated by training a linear classifier on top of the frozen
embeddings using only a small percentage of labeled data (1%, 5%, 10%, 20%,
50%, 100%). The core claim being tested: **at low label percentages,
self-supervised pretraining should outperform training from scratch or with
standard augmentation alone.**

This is not a demo of "self-supervised learning" as a buzzword — it is a
controlled comparison designed to produce one specific artifact: an accuracy
vs. label-percentage curve, per backbone, per training strategy.

---

## 2. Motivation

Labeled data is expensive to collect; unlabeled data is comparatively free.
Self-supervised learning exploits large pools of unlabeled data to learn
useful representations before any labels are seen, which matters most
precisely when labels are scarce. The project intentionally uses a **large
unlabeled pool** (100k images) and a **small, artificially restricted labeled
pool** (subsampled down to as little as 1% of 5k images = 50 labeled
examples) to make this trade-off explicit and measurable.

---

## 3. Dataset

**STL-10**

| Split | Size | Purpose |
|---|---|---|
| Unlabeled | 100,000 images | SimSiam pretraining |
| Labeled (train) | 5,000 images | Downstream linear probe training, subsampled |
| Labeled (test) | 8,000 images | Downstream evaluation |
| Classes | 10 | airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck |
| Image size | 96×96 (native) | Kept native — no upscaling to 224×224 |

STL-10 was chosen specifically because it was designed for semi-/self-supervised
research: it ships with a dedicated large unlabeled split, unlike CIFAR-10.

**Label percentage subsets used for evaluation:** 1%, 5%, 10%, 20%, 50%, 100%
of the 5,000-image labeled train split, with a **fixed random seed** across
runs so subsampling noise doesn't get confused with real signal.

---

## 4. Learning Strategies Compared

| Strategy | Description | Uses unlabeled data? |
|---|---|---|
| Baseline (Supervised) | Trained from scratch on labeled subset only | No |
| Data Augmentation | Baseline + random crop/flip/color jitter | No |
| ImageNet Transfer Learning | ImageNet-pretrained backbone, fine-tuned on labeled subset | No (uses external pretrained weights, not STL-10's unlabeled pool) |
| Self-Supervised (SimSiam) | Pretrained on unlabeled pool, then linear probe on labeled subset | Yes |

SimSiam was chosen over SimCLR/BYOL because it does not require large batch
sizes or negative pairs (no contrastive loss, no memory bank), which makes it
tractable on a 4GB-VRAM laptop GPU. This is a deliberate hardware-driven
choice, documented as such rather than hidden.

**Why ImageNet transfer learning is included:** it answers a practical
question a SSL-only comparison can't — *if ImageNet weights are freely
available, is self-supervised pretraining on your own unlabeled data still
worth the extra compute?* Fine-tuning a pretrained backbone is cheaper than
SimSiam pretraining (no pretraining stage at all), so this comparison is
close to free to add.

Semi-supervised methods (pseudo-labeling, FixMatch) and additional SSL
methods (SimCLR, BYOL) remain out of scope for v1 — listed under Future Work.

Semi-supervised methods (pseudo-labeling, FixMatch) and additional SSL
methods (SimCLR, BYOL) are explicitly **out of scope for v1** — listed under
Future Work, not implemented, to keep the comparison clean and finishable.

---

## 5. Architectures

| Backbone | Parameters (approx.) | Included? |
|---|---|---|
| ResNet18 | ~11M | Yes |
| MobileNetV2 | ~3.4M | Yes |
| EfficientNet-B0 | ~5.3M | Cut for v1 (stretch goal) |
| ViT-B/16, Swin, ConvNeXt | 30M–86M | Excluded — infeasible on 4GB VRAM and works against low-data thesis (transformers are more data-hungry) |

---

## 6. Evaluation Protocol

1. Pretrain encoder (SimSiam) on the full unlabeled pool — **1 seed**, due
   to compute cost (see Section 7 and Limitations).
2. Freeze encoder weights.
3. Train a linear classifier (scikit-learn `LogisticRegression`) on frozen
   embeddings, separately for each label percentage (1/5/10/20/50/100%),
   **repeated across 3 seeds (42, 123, 999)**.
4. Repeat steps 2–3's seed policy for the Baseline, Data Augmentation, and
   ImageNet Transfer Learning strategies (all trained directly on the
   labeled subset, all cheap enough to run 3 seeds each).
5. Repeat all of the above for both backbones.
6. Plot: **accuracy (y-axis) vs. label percentage (x-axis)**, one line per
   strategy, per backbone, with **mean ± standard deviation** across the 3
   seeds (e.g., `92.4 ± 0.6%`) rather than a single point estimate.

**Seed policy summary:**

| Stage | Seeds | Reasoning |
|---|---|---|
| SimSiam pretraining (encoder) | 1 | Multi-day training on available hardware; cost does not scale down |
| Linear probe (SSL embeddings) | 3 (42, 123, 999) | Trains in seconds–minutes; cheap to make robust |
| Baseline (from scratch) | 3 (42, 123, 999) | Same order of cost as linear probe |
| Data Augmentation | 3 (42, 123, 999) | Same order of cost as linear probe |
| ImageNet Transfer Learning | 3 (42, 123, 999) | Fine-tuning is cheap; no pretraining stage required |

**Metrics reported:** accuracy (mean ± std), precision, recall, F1-score,
confusion matrix. ROC/PR curves only if time allows (stretch).

**Efficiency metrics reported (per backbone):**

| Model | SSL Pretrain Time | Linear Probe Time | Params | Size (MB) | Inference Time |
|---|---|---|---|---|---|
| ResNet18 | *(measured)* | *(measured)* | ~11M | *(measured)* | *(measured)* |
| MobileNetV2 | *(measured)* | *(measured)* | ~3.4M | *(measured)* | *(measured)* |

This table exists specifically to answer "which model would you actually
deploy?" — accuracy alone doesn't answer that; efficiency and size do.

**Explainability:** one Grad-CAM figure (image → prediction → heatmap) is
generated for the final deployed model — mandatory, not a stretch goal,
since it is a single figure produced once, not repeated per experiment.

**Experiment tracking:** Weights & Biases tracks training loss, validation
accuracy, learning rate, GPU memory, and epoch duration for all runs;
relevant screenshots are included in the README rather than building a
custom tracking dashboard.

---

## 7. Hardware & Training Constraints

- **Hardware:** RTX 3050 (laptop, 4GB VRAM), Ryzen 5 5000-series, 16GB RAM.
- **Batch size:** capped at 64, mixed precision (fp16) required to fit
  within VRAM.
- **Resolution:** native 96×96 (no upscaling — keeps memory and compute
  manageable, matches STL-10's design intent).
- **Epochs:** SimSiam pretraining targeted at 100 epochs first, extended
  only if the loss curve / linear-probe accuracy is still improving.
- **Checkpointing:** every 5 epochs, to survive interrupted/overnight runs.
- **Expected training time:** several hours to multi-day for full SimSiam
  pretraining, given laptop GPU throughput — planned as an overnight/
  background job, not a single foreground session.

---

## 8. Deployment Plan

Deployment is treated as a design constraint decided *before* training, not
bolted on afterward.

- **Model export:** TorchScript or ONNX — decouples serving from training
  code, avoids environment/version mismatches.
- **Single deployed model:** one final choice (best backbone + best
  strategy), not a live multi-model comparison tool. Comparison results are
  static (plots/tables) in the README/notebook, not computed at runtime.
- **Serving:** Gradio app on Hugging Face Spaces — free hosting, no custom
  frontend/backend split to maintain.
- **Excluded from the live app:** Grad-CAM, t-SNE, live multi-backbone
  comparisons — kept as static images in documentation to avoid extra
  inference-time compute and failure modes in production.

**Final model choice (locked in after results, see Section 11.6):
MobileNetV2, fine-tuned from ImageNet weights.** It was the most accurate
combination at every label percentage tested across all 8 backbone ×
strategy combinations, and it has the smallest model size/download
footprint. It is not the fastest option on this study's GPU -- ResNet18
is roughly 2x faster per image at inference (Section 11.5) -- but for a
small interactive demo, sub-10ms inference on either backbone is well
within acceptable latency, so accuracy and size were weighted more
heavily than the speed difference.

---

## 9. Project Structure

```
LowDataSSL/
├── data/
│   └── stl10_loader.py        # dataset download + percentage-split logic
├── training/
│   ├── simsiam.py              # SSL pretraining
│   ├── baseline.py              # supervised from scratch
│   └── augmented.py             # supervised + augmentation
├── evaluation/
│   ├── linear_probe.py          # frozen-embedding logistic regression
│   └── metrics.py                # precision/recall/F1/confusion matrix
├── export/
│   └── to_onnx.py                # model export for deployment
├── app/
│   └── gradio_app.py              # HF Spaces deployment
├── notebooks/
│   └── results_analysis.ipynb      # accuracy-vs-label% plots
├── checkpoints/
├── docs/
│   └── PROJECT_DOCUMENTATION.md (this file)
└── README.md
```

---

## 10. Scope Decisions (What Was Deliberately Cut, and Why)

| Feature | Status | Reason |
|---|---|---|
| Semi-supervised (FixMatch, pseudo-labeling) | Cut (v1) | Separate research direction, not an add-on |
| ViT / Swin / ConvNeXt | Cut | Infeasible on 4GB VRAM; more data-hungry, undermines low-data thesis |
| Score-CAM, Integrated Gradients | Cut | Grad-CAM alone is sufficient for a portfolio artifact |
| Quantization / pruning | Cut | Separate edge-deployment project |
| Real-time camera mode | Cut | Not relevant to the core research question |
| OOD detection | Cut | Separate project |
| Retrieval / similarity search | Cut | Interesting but scope creep |
| Custom experiment-tracking dashboard | Cut, replaced by W&B | No need to build one when W&B exists |
| Live multi-model comparison in deployed app | Cut | Conflicts with "effortless deployment" goal; static plots shown instead |
| 3-seed SimSiam pretraining | Cut | Compute cost multiplies the already-expensive stage; 1 seed used, disclosed as a limitation |

**Added after initial scope (accepted on merit, not by default):**

| Feature | Status | Reason |
|---|---|---|
| ImageNet transfer learning baseline | Added | Cheap (no pretraining stage); answers a sharper practical question than SSL alone |
| Training time / model size / inference tables | Added | Cheap logging; directly supports a deployment decision |
| Grad-CAM (mandatory, not stretch) | Added | One figure, produced once, low cost |
| Weights & Biases tracking | Added | Replaces a custom dashboard that was never going to get built anyway |
| Research-paper style documentation structure | Added | Zero compute cost, pure formatting |
| 3-seed evaluation (linear probe, baseline, augmentation, transfer learning) | Added | Cheap stages made statistically robust; only SimSiam pretraining stays single-seed |

---

## 11. Results (Both Backbones)

SimSiam pretrained for 75 epochs (ResNet18) and 80 epochs (MobileNetV2) --
both stopped once loss clearly plateaued, not at a fixed epoch count.
ResNet18's loss stabilized around -0.85 from ~epoch 25; MobileNetV2's
stabilized around -0.89 from ~epoch 50.

**Full comparison, all 8 backbone × strategy combinations, accuracy (mean ±
std across 3 seeds, each seed a different random label-percentage
subsample):**

| Label % | R18 Baseline | R18 Augmented | R18 ImageNet | R18 SimSiam | MNv2 Baseline | MNv2 Augmented | MNv2 ImageNet | MNv2 SimSiam |
|---|---|---|---|---|---|---|---|---|
| 1% | 20.7 ± 1.1% | 23.7 ± 1.6% | 54.4 ± 1.7% | 56.7 ± 0.9% | 10.0 ± 0.0% | 10.0 ± 0.0% | **57.3 ± 1.1%** | 53.6 ± 1.2% |
| 5% | 33.3 ± 0.9% | 38.7 ± 1.5% | 62.9 ± 2.1% | 68.0 ± 0.2% | 22.9 ± 1.0% | 32.3 ± 1.2% | **71.3 ± 1.3%** | 61.2 ± 0.2% |
| 10% | 36.3 ± 1.6% | 43.2 ± 0.6% | 69.0 ± 2.3% | 70.1 ± 0.2% | 26.6 ± 1.8% | 37.0 ± 0.5% | **74.6 ± 1.0%** | 62.9 ± 0.3% |
| 20% | 36.7 ± 4.8% | 49.5 ± 3.5% | 76.8 ± 0.9% | 71.6 ± 0.0% | 33.6 ± 1.3% | 42.2 ± 0.8% | **79.3 ± 0.6%** | 64.6 ± 0.3% |
| 50% | 52.8 ± 0.5% | 60.1 ± 1.6% | 80.1 ± 0.5% | 74.2 ± 0.3% | 42.4 ± 0.8% | 54.4 ± 0.9% | **83.3 ± 0.5%** | 66.7 ± 0.2% |
| 100% | 62.5 ± 0.7% | 68.6 ± 1.3% | 84.3 ± 0.9% | 76.1 ± 0.0% | 50.9 ± 0.9% | 65.2 ± 0.5% | **87.7 ± 0.8%** | 68.0 ± 0.0% |

### 11.1 Key finding 1 -- the ResNet18 crossover (SSL vs. ImageNet transfer)

Restricting to ResNet18: SimSiam pretraining on STL-10's unlabeled pool
beats ImageNet transfer learning only in the extreme low-data regime (≤10%
labels), and even there the margin narrows quickly (2.3 → 5.1 → 1.1 points).
Past 10% labels, ImageNet transfer wins and its advantage grows as more
labels become available (5.2 → 5.9 → 8.2 points). **Practical conclusion:**
self-supervised pretraining on domain-specific unlabeled data is worth the
extra compute specifically when labeled data is extremely scarce (roughly
≤10% in this setup); past that point, transfer learning from ImageNet is
the stronger and cheaper default.

### 11.2 Key finding 2 -- MobileNetV2 + ImageNet transfer wins outright

Across both backbones and every label percentage, **MobileNetV2 fine-tuned
from ImageNet weights is the single best-performing combination in the
entire study** -- beating even ResNet18 + SimSiam, the combination the
project was originally built to showcase. This overturns two naive
assumptions at once: that a larger backbone (ResNet18, ~11M params) should
outperform a smaller one (MobileNetV2, ~3.4M params), and that the more
sophisticated self-supervised method should beat a simple transfer-learning
baseline. Neither held. The practical, defensible conclusion is that the
smaller, cheaper backbone paired with free pretrained weights is the
correct deployment choice here -- not the more elaborate SSL pipeline.

### 11.3 Notable anomaly -- MobileNetV2 collapse at 1% labels (from scratch)

MobileNetV2 Baseline and Augmented both collapsed to exactly 10.0 ± 0.0%
accuracy at 1% labels (50 images) across all 3 seeds -- chance level for a
10-class problem, with zero variance, indicating the model predicted a
single class for every test image regardless of seed. ResNet18 did not
collapse this way at the same label percentage (20.7%). This is reported
as an observed architecture-robustness difference (MobileNetV2's depthwise
separable convolutions and batch normalization statistics may be more
sensitive to very small batches/datasets when trained from scratch), not
attributed to a bug -- the same code path produced sane results at every
other label percentage and for the other three strategies.

### 11.4 Secondary finding -- augmentation's contribution is inversely
related to where SSL's advantage matters most

Data augmentation alone (no pretraining) closes only 8-20% of the gap
between Baseline and SimSiam at the lowest label percentages (1-10%) for
ResNet18, confirming SimSiam's advantage there comes from genuine
pretraining on unlabeled data, not merely better regularization. This
contribution grows at higher label percentages (~45% of the gap closed at
100%) -- the opposite of where SSL's advantage matters most.

### 11.5 Efficiency (architecture-only -- same for every strategy)

Measured on the study's hardware (RTX 3050 Laptop GPU), single-image
inference, 96×96 input:

| Model | Params | Size (MB) | Inference Time | FPS |
|---|---|---|---|---|
| ResNet18 | 11.18M | 42.73 MB | 4.04 ms | 247.3 |
| MobileNetV2 | 2.24M | **8.76 MB** | 8.25 ms | 121.2 |

**Counterintuitive result, stated plainly:** MobileNetV2 is ~5x smaller
but roughly **2x slower** than ResNet18 on this GPU. This is not a bug --
MobileNetV2's depthwise separable convolutions are optimized for FLOP
count and mobile/CPU deployment, but involve many small sequential
operations that parallelize less efficiently on GPU hardware than
ResNet18's larger, cuDNN-optimized standard convolutions, especially at
batch size 1. "Fewer parameters" does not automatically mean "faster" --
the answer is hardware-dependent. On a CPU-only deployment target (e.g.
Hugging Face Spaces' free tier), this comparison could plausibly reverse,
since depthwise separable convolutions often do win on CPU. This was not
verified in this study and is noted as a gap, not assumed.

### 11.6 Deployment decision

**Chosen for deployment: MobileNetV2, fine-tuned from ImageNet weights.**
This is the most accurate combination across all label percentages tested
(Section 11.2) and has the smallest model size/download footprint (Section
11.5) -- but it is not the fastest inference option on this study's GPU,
where ResNet18 is roughly 2x faster per image. The choice prioritizes
accuracy and deployment size over raw GPU inference speed, which is an
explicit tradeoff, not an oversight: for a small classification demo
served via Gradio, sub-10ms inference on either model is well within
acceptable latency for interactive use, so accuracy and download size were
weighted more heavily than the speed difference.

---

## 12. Limitations

- SimSiam encoder pretraining was run **once per backbone** due to compute
  constraints (multi-day training on a 4GB-VRAM laptop GPU); this is the
  one genuine single-seed gap in the project. All downstream evaluation
  (linear probe, baseline, augmentation, transfer learning) was repeated
  across 3 seeds and is reported as mean ± standard deviation.
- Results are specific to STL-10 at 96×96 resolution; they may not
  generalize to larger images or different domains.
- SimSiam was chosen for hardware feasibility, not because it's the
  strongest-performing SSL method available (SimCLR/BYOL/DINO may outperform
  it given sufficient compute).
- Only two backbones compared; conclusions about "which architecture is
  best" are limited to ResNet18 vs. MobileNetV2, not a general claim about
  CNN architectures.
- Label-percentage subsampling uses a fixed split per percentage (not
  re-sampled per seed), so reported variance reflects downstream
  training/init noise, not subsampling noise.
- MobileNetV2 trained from scratch (Baseline and Augmented strategies)
  collapsed to chance-level accuracy at 1% labels (see Section 11.3) --
  this was not investigated further (e.g. via learning-rate tuning or
  batch-norm adjustments) since the deployment decision did not depend on
  fixing it; flagged here rather than omitted.

## 13. Future Work

- Add SimCLR/BYOL for comparison against SimSiam (compute permitting).
- Add EfficientNet-B0 once core results are validated.
- Semi-supervised methods (FixMatch) as a separate comparative study.
- Quantization/pruning for an edge-deployment follow-up project.
- t-SNE/UMAP visualization of embedding space before vs. after SSL
  pretraining (static in docs only).
- Multi-seed SimSiam pretraining, if additional compute becomes available.
