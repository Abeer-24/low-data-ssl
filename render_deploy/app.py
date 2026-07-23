"""
Gradio app for Render.com free-tier deployment -- serves the deployed
model (MobileNetV2 + ImageNet transfer, trained on 100% labels).

Differs from app/gradio_app.py (local) and hf_space/app.py (Hugging Face
Spaces) only in how it binds host/port: Render requires binding to
0.0.0.0 and reading the assigned port from the PORT environment variable,
rather than Gradio's default localhost binding.

See PROJECT_DOCUMENTATION.md in github.com/Abeer-24/low-data-ssl for full
methodology and results.
"""

import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import gradio as gr

ONNX_PATH = os.path.join(os.path.dirname(__file__), "mobilenet_v2_deploy.onnx")

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

IMAGE_SIZE = 96
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
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


session = ort.InferenceSession(ONNX_PATH)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Prediction"),
    title="STL-10 Classifier (MobileNetV2 + ImageNet Transfer)",
    description=(
        "Trained on 100% of STL-10's labeled data, fine-tuned from "
        "ImageNet weights. Classes: airplane, bird, car, cat, deer, dog, "
        "horse, monkey, ship, truck. Note: this model always predicts one "
        "of these 10 classes, even for out-of-distribution images -- no "
        "'none of the above' option (see project README for details). "
        "Full methodology and results: "
        "github.com/Abeer-24/low-data-ssl"
    ),
)

if __name__ == "__main__":
    # Render assigns a port via the PORT env var and requires binding to
    # 0.0.0.0, not localhost -- without this, the deployed service will
    # build successfully but never respond to requests.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
