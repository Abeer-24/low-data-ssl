"""
Gradio app serving the deployed model (MobileNetV2 + ImageNet transfer,
trained on 100% labels -- see PROJECT_DOCUMENTATION.md, Section 11.6).

Serves the ONNX export (export/to_onnx.py), not the raw PyTorch checkpoint
-- decouples the app from the training environment/dependencies, per the
deployment plan (Section 8). Only the single chosen model is served; the
full backbone x strategy comparison lives as static tables/figures in
PROJECT_DOCUMENTATION.md, not as a live feature here.

Run locally:
    python app/gradio_app.py

To deploy on Hugging Face Spaces:
    1. Create a new Space (SDK: Gradio).
    2. Upload this file as app.py, plus checkpoints/deploy/mobilenet_v2_deploy.onnx.
    3. Add a requirements.txt with: gradio, onnxruntime, numpy, pillow
"""

import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import gradio as gr

ONNX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "deploy", "mobilenet_v2_deploy.onnx"
)

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

IMAGE_SIZE = 96
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_session():
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(
            f"No ONNX model found at {ONNX_PATH}. Run export/to_onnx.py "
            f"--backbone mobilenet_v2 first."
        )
    return ort.InferenceSession(ONNX_PATH)


def preprocess(pil_image: Image.Image) -> np.ndarray:
    # Resize to the model's expected input size, convert to the same
    # normalization used during training (ImageNet mean/std -- see
    # get_eval_transform() in training/imagenet_transfer.py).
    img = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - np.max(x))
    return exp / exp.sum()


def predict(pil_image: Image.Image):
    if pil_image is None:
        return {}

    input_array = preprocess(pil_image)
    outputs = session.run(None, {"input": input_array})[0]
    probs = softmax(outputs[0])

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


session = load_session()

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Prediction"),
    title="STL-10 Classifier (MobileNetV2 + ImageNet Transfer)",
    description=(
        "Trained on 100% of STL-10's labeled data, fine-tuned from "
        "ImageNet weights -- the best-performing combination found across "
        "8 backbone x strategy comparisons in this project. "
        "Full methodology and results: see PROJECT_DOCUMENTATION.md in the "
        "repo. Classes: airplane, bird, car, cat, deer, dog, horse, "
        "monkey, ship, truck."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
