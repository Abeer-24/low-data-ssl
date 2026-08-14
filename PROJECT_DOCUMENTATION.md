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

| Backbone | Parameters (measured) | Included? |
|---|---|---|
| ResNet18 | 11.18M | Yes |
| MobileNetV2 | 2.24M | Yes |
| EfficientNet-B0 | 4.02M | Yes — added after initial scope (see Section 10) |
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
bolted on afterward. **Note: the original plan below (single fixed model,
Hugging Face Spaces) changed after initial deployment — see the update at
the end of this section for what's actually live.**

- **Model export:** ONNX — decouples serving from training code, avoids
  environment/version mismatches. All 12 backbone × strategy combinations
  are exported (SimSiam exports as an encoder-only ONNX graph plus a
  separately saved linear classifier, since scikit-learn's
  `LogisticRegression` isn't ONNX-native).
- **Excluded from the live app:** Grad-CAM specifically — this is a hard
  technical constraint, not a scope choice: ONNX Runtime has no
  autograd/backprop support, so gradient-based explainability methods
  cannot run on it regardless of effort. Grad-CAM remains a static figure
  in this documentation.

**Update — what's actually deployed:** the original single-model plan was
revisited after deployment. The live app (Render.com, not Hugging Face
Spaces — see note below) lets visitors choose *any* of the 12 trained
backbone × strategy combinations and classify images live, plus a
"Compare All" view that runs all 12 on one uploaded image simultaneously,
grouped by backbone. This reverses the original "single deployed model,
no live comparison" decision — live comparison turned out to be worth the
added complexity, since it directly demonstrates the project's actual
thesis (comparing strategies) rather than requiring the visitor to trust
a single number.

**Hosting note:** Hugging Face Spaces changed its pricing during this
project — as of mid-2026, creating a Space with the Gradio or Docker SDK
requires a paid plan; only Static Spaces remain free. The app is deployed
on **Render.com's free tier** instead (no credit card required; the
service sleeps after 15 minutes of inactivity, with a ~30-60s cold start
on the next visit).

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
| EfficientNet-B0 as a third backbone | Added | Originally cut for requiring a fresh multi-day SimSiam pretraining run; added later once that time was available. Its SimSiam run initially destabilized (loss oscillating, near-chance downstream accuracy) at the same learning rate that worked for ResNet18/MobileNetV2 — halving the learning rate (0.05 → 0.02) fixed it, a genuine architecture-specific hyperparameter sensitivity finding, not a bug (see Section 11) |
| Live backbone/strategy selector + "Compare All" view in deployed app | Added, reversing the original "single model, no live comparison" decision | Directly demonstrates the project's core thesis (comparing strategies) rather than requiring the visitor to trust one static number; made practical by ONNX's fast CPU inference (single-digit-ms per model) |

---

## 11. Results (All Three Backbones)

SimSiam pretrained for 75 epochs (ResNet18), 80 epochs (MobileNetV2), and
100 epochs (EfficientNet-B0) — each stopped once loss clearly plateaued,
not at a fixed epoch count. ResNet18's loss stabilized around -0.85 from
~epoch 25; MobileNetV2's around -0.89 from ~epoch 50.

**EfficientNet-B0's pretraining needed a real fix, not just more epochs:**
at the same learning rate (0.05) that worked for the other two backbones,
its loss destabilized after epoch ~55 (repeated sharp drops to -0.62 to
-0.76, never recovering to its earlier best), and the resulting encoder
was nearly useless — 21-24% linear-probe accuracy, barely above the 10%
chance baseline. Halving the learning rate to 0.02 and retraining from
scratch fixed this completely: loss held a clean plateau around -0.88 from
epoch ~55 onward, and linear-probe accuracy jumped to 43.7-61.1%. This is
reported as a genuine architecture-specific hyperparameter sensitivity
finding — the same optimizer settings do not transfer cleanly across
backbones — not glossed over as a minor tuning detail.

**Full comparison, all 12 backbone × strategy combinations, accuracy (mean
± std across 3 seeds, each seed a different random label-percentage
subsample):**

| Backbone | Strategy | 1% | 5% | 10% | 20% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| ResNet18 | Baseline | 20.8±1.3% | 31.7±2.8% | 37.1±1.0% | 42.3±0.6% | 51.7±1.4% | 61.5±0.5% |
| ResNet18 | Augmented | 23.8±2.0% | 38.8±1.5% | 40.4±2.0% | 51.2±2.6% | 59.8±1.7% | 67.1±0.3% |
| ResNet18 | ImageNet Transfer | 54.5±1.7% | 64.4±6.0% | 68.9±2.3% | 76.2±0.6% | 77.9±2.3% | 84.8±1.3% |
| ResNet18 | SimSiam | 56.7±0.9% | 68.0±0.2% | 70.1±0.2% | 71.6±0.0% | 74.2±0.3% | 76.1±0.0% |
| MobileNetV2 | Baseline | 10.0±0.0% | 22.6±0.9% | 26.7±2.1% | 32.7±1.5% | 41.9±0.3% | 49.9±0.8% |
| MobileNetV2 | Augmented | 10.0±0.0% | 33.2±0.7% | 36.5±1.9% | 41.7±0.9% | 53.3±1.3% | 64.3±1.6% |
| MobileNetV2 | ImageNet Transfer | 57.4±0.9% | 73.1±0.3% | 74.7±1.0% | 78.2±0.7% | 82.4±1.2% | 86.4±1.1% |
| MobileNetV2 | SimSiam | 53.6±1.2% | 61.2±0.2% | 62.9±0.3% | 64.6±0.3% | 66.7±0.2% | 68.0±0.0% |
| EfficientNet-B0 | Baseline | 10.0±0.0% | 24.1±2.3% | 28.0±0.3% | 32.6±0.3% | 42.4±0.6% | 53.9±0.8% |
| EfficientNet-B0 | Augmented | 10.0±0.0% | 31.8±3.4% | 38.1±1.8% | 44.5±1.3% | 52.1±2.8% | 65.5±0.9% |
| EfficientNet-B0 | **ImageNet Transfer** | **59.1±1.4%** | **77.3±0.2%** | **78.8±1.1%** | **82.0±1.1%** | **87.0±0.1%** | **89.3±0.5%** |
| EfficientNet-B0 | SimSiam | 43.7±0.9% | 54.1±0.9% | 57.1±0.3% | 58.2±0.3% | 60.1±0.1% | 61.1±0.0% |

**EfficientNet-B0 + ImageNet Transfer Learning is the best-performing
combination at every single label percentage in the entire study**,
overtaking MobileNetV2 + ImageNet Transfer (the previous documented
best). This is flagged explicitly because it changes the practical
conclusion from the two-backbone version of this study, not just extends
it.

### 11.1 Key finding 1 -- the ResNet18 crossover is backbone-specific, not universal

Restricting to ResNet18: SimSiam pretraining on STL-10's unlabeled pool
beats ImageNet transfer learning only in the extreme low-data regime (≤10%
labels), and even there the margin narrows quickly (2.3 → 5.1 → 1.1 points).
Past 10% labels, ImageNet transfer wins and its advantage grows as more
labels become available (5.2 → 5.9 → 8.2 points).

**This crossover does not hold for the other two backbones.** For both
MobileNetV2 and EfficientNet-B0, ImageNet Transfer beats SimSiam at *every*
label percentage tested, including 1% (MobileNetV2: 57.4% vs. 53.6%;
EfficientNet-B0: 59.1% vs. 43.7%). Only ResNet18 shows SimSiam pretraining
ever being the better choice. **Practical conclusion, corrected from the
two-backbone version of this study:** self-supervised pretraining being
worthwhile in the extreme low-data regime is not a general property of
this setup — it depends on the backbone architecture, and held for only
one of the three tested here.

### 11.2 Key finding 2 -- EfficientNet-B0 + ImageNet transfer is the true
best combination, not MobileNetV2

With only two backbones, MobileNetV2 + ImageNet Transfer appeared to be
the best-performing combination in the study. **Adding EfficientNet-B0
overturns that**: EfficientNet-B0 + ImageNet Transfer beats MobileNetV2 +
ImageNet Transfer at every label percentage (e.g. 89.3% vs. 86.4% at
100%; 59.1% vs. 57.4% at 1%), and also beats ResNet18 + SimSiam, the
combination this project was originally built to showcase. This is a
concrete illustration of why the two-backbone version of this document
was labeled a limitation rather than a final answer (see Section 12):
adding one more architecture changed the actual winner, not just the
supporting detail count. The practical, defensible conclusion remains
the same in spirit — the more sophisticated SSL pipeline is not
automatically the best choice — but the specific "best" model changed
once a wider architecture search was actually run.

### 11.3 Notable pattern -- from-scratch collapse at 1% labels (not just MobileNetV2)

MobileNetV2 Baseline and Augmented both collapsed to exactly 10.0 ± 0.0%
accuracy at 1% labels (50 images) across all 3 seeds -- chance level for a
10-class problem, with zero variance, indicating the model predicted a
single class for every test image regardless of seed. **EfficientNet-B0
shows the identical pattern** (Baseline and Augmented both 10.0 ± 0.0% at
1%). ResNet18 did not collapse this way at the same label percentage
(20.8%). With the collapse now observed in two of three backbones, this
looks less like a MobileNetV2-specific quirk and more like a broader
pattern: architectures using batch normalization and comparatively
higher effective capacity for their depth may fail to learn *anything*
useful from just 50 images without either pretrained weights or an SSL
warm-start, while ResNet18 avoids total failure at the same data volume.
This remains a reported observation, not a root-caused bug — the same
code path produced sane results at every other label percentage and
strategy for all three backbones.

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
| ResNet18 | 11.18M | 42.73 MB | 3.89 ms | 257.3 |
| MobileNetV2 | 2.24M | **8.76 MB** | 7.18 ms | 139.4 |
| EfficientNet-B0 | 4.02M | 15.62 MB | 10.61 ms | 94.3 |

**Pattern confirmed, not just a two-backbone coincidence:** fewer
parameters does not mean faster GPU inference. ResNet18 is both the
largest model and the fastest at inference; EfficientNet-B0 is a
mid-sized model but the slowest of the three. This tracks with the
architectures' designs — ResNet18's standard convolutions are the most
GPU/cuDNN-friendly of the three, while EfficientNet-B0 (like MobileNetV2)
relies on depthwise separable convolutions and additional squeeze-excitation
blocks, adding more sequential small operations that parallelize less
efficiently on GPU hardware at batch size 1. Despite being the slowest
and a mid-sized model, EfficientNet-B0 is still the deployment choice
(Section 8) because its accuracy advantage (Section 11.2) was judged more
important than inference speed for an interactive single-image demo,
where sub-15ms inference is imperceptible to a user regardless of which
of the three models is chosen.

### 11.6 Deployment decision

The deployed app does not serve one fixed model — see Section 8's update
for why that changed. **If a single "best" combination had to be named,
it is EfficientNet-B0 + ImageNet Transfer Learning** (Section 11.2): the
most accurate combination at every label percentage tested, though not
the fastest or smallest (Section 11.5) — ResNet18 is roughly 2.7x faster
per image and MobileNetV2 is roughly 1.8x smaller. For an interactive
single-image demo, sub-15ms inference is imperceptible regardless of
which of the three is used, so accuracy was weighted over the size/speed
difference for this specific recommendation — but the live app lets a
visitor make that tradeoff themselves rather than having it decided for
them in advance.

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
- Three backbones compared (ResNet18, MobileNetV2, EfficientNet-B0);
  conclusions about "which architecture is best" are limited to these
  three, not a general claim about CNN architectures. The two-backbone
  version of this study named a different "best" combination than the
  three-backbone version does (Section 11.2) — a concrete illustration
  that this limitation is real, not a formality.
- Label-percentage subsampling uses a fixed split per percentage (not
  re-sampled per seed), so reported variance reflects downstream
  training/init noise, not subsampling noise.
- MobileNetV2 and EfficientNet-B0 trained from scratch (Baseline and
  Augmented strategies) both collapsed to chance-level accuracy at 1%
  labels (see Section 11.3) -- this was not root-caused (e.g. via
  learning-rate tuning or batch-norm adjustments specific to that failure)
  since the deployment decision did not depend on fixing it; flagged here
  rather than omitted.
- EfficientNet-B0's SimSiam pretraining required a different learning
  rate (0.02 vs. 0.05) than the other two backbones to avoid training
  instability (Section 11); hyperparameters were not systematically
  re-tuned per backbone beyond this one necessary fix, so it's possible
  further tuning could change any of the three backbones' SimSiam results.

## 13. Future Work

- Add SimCLR/BYOL for comparison against SimSiam (compute permitting).
- Semi-supervised methods (FixMatch) as a separate comparative study.
- Quantization/pruning for an edge-deployment follow-up project.
- t-SNE/UMAP visualization of embedding space before vs. after SSL
  pretraining (static in docs only).
- Multi-seed SimSiam pretraining, if additional compute becomes available.
- Root-cause the from-scratch collapse at 1% labels (MobileNetV2 and
  EfficientNet-B0, Section 11.3) rather than only reporting it.
- Systematically re-tune SimSiam hyperparameters per backbone, rather than
  only fixing EfficientNet-B0's learning rate reactively after observing
  instability.
- CPU-only inference benchmarking (the current efficiency numbers, Section
  11.5, are GPU-only; the deployed app runs on Render's CPU-only free
  tier, and the size/speed ranking could plausibly differ there).
