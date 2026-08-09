"""
Exports a handful of STL-10 test images as PNG files, one per class, for
use as clickable examples in the Gradio app (better UX than requiring
every visitor to find and upload their own image).

Run this once locally, then copy the output folder into render_deploy/
before redeploying.

Usage:
    python evaluation/export_examples.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from stl10_loader import get_stl10_splits, DATA_ROOT

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


def export_one_per_class():
    _, _, test = get_stl10_splits(root=DATA_ROOT)
    labels = np.array(test.labels)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    found = set()

    for idx in range(len(test)):
        label = labels[idx]
        if label in found:
            continue

        pil_img, _ = test[idx]
        class_name = CLASS_NAMES[label]
        save_path = os.path.join(OUTPUT_DIR, f"{class_name}.png")
        pil_img.save(save_path)
        found.add(label)
        print(f"Saved: {save_path}")

        if len(found) == len(CLASS_NAMES):
            break

    print(f"\nExported {len(found)} example images to: {OUTPUT_DIR}")
    print("Next: copy this examples/ folder into render_deploy/ before redeploying.")


if __name__ == "__main__":
    export_one_per_class()
