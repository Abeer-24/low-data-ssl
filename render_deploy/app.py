"""
Gradio app for Render.com deployment.

Features:
- Home tab: project motivation, quick nav to Classify/Compare All, stat
  cards, and a ranked/filterable performance comparison with progress
  bars, color-coded metrics, and sparklines.
- Classify tab: pick backbone + strategy, upload an image, see prediction
  with a model info card, confidence gauge, and thumbnail history.
- Compare All tab: upload once, see all 12 backbone x strategy predictions,
  grouped by backbone.
- History tab: this session's prediction history with image thumbnails.

Grad-CAM is intentionally NOT live here -- ONNX Runtime (used for all
inference in this app) has no autograd/backprop support, so gradient-based
explainability methods can't run on it. Grad-CAM stays a static figure in
the project documentation. See PROJECT_DOCUMENTATION.md, Section 8.
"""

import os
import io
import json
import time
import gc
import base64
import numpy as np
from PIL import Image
import onnxruntime as ort
import gradio as gr
from fpdf import FPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = BASE_DIR
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

BACKBONES = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
BACKBONE_LABELS = {"resnet18": "ResNet18", "mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0"}
BACKBONE_ICONS = {"resnet18": "\U0001F525", "mobilenet_v2": "\u26A1", "efficientnet_b0": "\U0001F9E0"}
STRATEGIES = ["baseline", "augmented", "imagenet_transfer", "simsiam"]
STRATEGY_LABELS = {
    "baseline": "Baseline (from scratch)",
    "augmented": "Data Augmentation",
    "imagenet_transfer": "ImageNet Transfer Learning",
    "simsiam": "Self-Supervised (SimSiam)",
}
STRATEGY_ICONS = {
    "baseline": "\U0001F9EA",
    "augmented": "\U0001F504",
    "imagenet_transfer": "\U0001F680",
    "simsiam": "\U0001F9E0",
}

IMAGE_SIZE = 96
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

from collections import OrderedDict

# Bounded LRU cache -- NOT an unbounded dict. There are 12 possible models
# (9 classifiers + 3 SimSiam encoder/weights pairs); if every combination
# gets requested (e.g. via the Compare All tab), an unbounded cache would
# keep all 12 loaded in memory forever, which is very likely what caused
# a real OOM kill (exit 137) on Render's 512MB free tier. Capping at 4
# entries keeps memory bounded while still caching repeat single-model use
# in the Classify tab.
_SESSION_CACHE_MAX_SIZE = 3
_session_cache = OrderedDict()


def _cache_get(key):
    if key in _session_cache:
        _session_cache.move_to_end(key)  # mark as recently used
        return _session_cache[key]
    return None


def _cache_put(key, value):
    _session_cache[key] = value
    _session_cache.move_to_end(key)
    if len(_session_cache) > _SESSION_CACHE_MAX_SIZE:
        _session_cache.popitem(last=False)  # evict least-recently-used


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

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


def _low_memory_session_options() -> ort.SessionOptions:
    """ONNX Runtime's defaults assume a machine with many cores and
    generous memory -- both its thread pool and its memory arena can
    over-allocate relative to what's actually available on a constrained
    container like Render's free tier. This caps both explicitly, which
    is standard practice for onnxruntime in resource-constrained
    deployments, not a guess specific to this app's symptoms."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    return options


def get_classifier_session(backbone: str, strategy: str):
    key = ("classifier", backbone, strategy)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    path = os.path.join(MODELS_DIR, f"{backbone}_{strategy}_classifier.onnx")
    if not os.path.exists(path):
        return None
    session = ort.InferenceSession(path, sess_options=_low_memory_session_options())
    _cache_put(key, session)
    return session


def get_simsiam_session_and_weights(backbone: str):
    key = ("simsiam", backbone)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    encoder_path = os.path.join(MODELS_DIR, f"{backbone}_simsiam_encoder.onnx")
    weights_path = os.path.join(MODELS_DIR, f"{backbone}_simsiam_linear_weights.npz")
    if not os.path.exists(encoder_path) or not os.path.exists(weights_path):
        return None
    session = ort.InferenceSession(encoder_path, sess_options=_low_memory_session_options())
    weights = np.load(weights_path)
    result = (session, weights["coef"], weights["intercept"])
    _cache_put(key, result)
    return result


def run_single_inference(backbone: str, strategy: str, input_array: np.ndarray):
    """Returns (result_dict, inference_time_ms) or (None, None) if the
    model files aren't available. Uses the shared cache -- appropriate
    for the Classify tab, where the same model is likely to be reused
    across several requests in a row."""
    start = time.perf_counter()
    if strategy == "simsiam":
        loaded = get_simsiam_session_and_weights(backbone)
        if loaded is None:
            return None, None
        session, coef, intercept = loaded
        features = session.run(None, {"input": input_array})[0][0]
        logits = coef @ features + intercept
        probs = softmax(logits)
    else:
        session = get_classifier_session(backbone, strategy)
        if session is None:
            return None, None
        outputs = session.run(None, {"input": input_array})[0]
        probs = softmax(outputs[0])
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return result, elapsed_ms


def run_single_inference_ephemeral(backbone: str, strategy: str, input_array: np.ndarray):
    """Same as run_single_inference, but deliberately bypasses the shared
    cache -- loads the model fresh, runs one inference, then explicitly
    frees it before returning. Used by Compare All, where each of the 12
    models is only ever used once per request, so caching provides no
    benefit and only risks keeping several large models resident in
    memory at once -- the most likely cause of the OOM kills this app hit
    on Render's 512MB free tier. Peak memory here stays roughly bounded
    to one model's footprint at a time instead of several."""
    start = time.perf_counter()
    session = None
    try:
        if strategy == "simsiam":
            encoder_path = os.path.join(MODELS_DIR, f"{backbone}_simsiam_encoder.onnx")
            weights_path = os.path.join(MODELS_DIR, f"{backbone}_simsiam_linear_weights.npz")
            if not os.path.exists(encoder_path) or not os.path.exists(weights_path):
                return None, None
            session = ort.InferenceSession(encoder_path, sess_options=_low_memory_session_options())
            weights = np.load(weights_path)
            features = session.run(None, {"input": input_array})[0][0]
            logits = features @ weights["coef"].T + weights["intercept"]
            probs = softmax(logits)
        else:
            path = os.path.join(MODELS_DIR, f"{backbone}_{strategy}_classifier.onnx")
            if not os.path.exists(path):
                return None, None
            session = ort.InferenceSession(path, sess_options=_low_memory_session_options())
            outputs = session.run(None, {"input": input_array})[0]
            probs = softmax(outputs[0])
    finally:
        # Explicitly drop the reference so CPython's refcounting frees the
        # session's memory immediately, rather than waiting for it to be
        # evicted from a cache it was never added to.
        del session

    elapsed_ms = (time.perf_counter() - start) * 1000
    result = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return result, elapsed_ms


# ---------------------------------------------------------------------------
# Small HTML/visual helpers
# ---------------------------------------------------------------------------

def pil_to_thumbnail_data_uri(pil_image: Image.Image, size=48) -> str:
    thumb = pil_image.convert("RGB").copy()
    thumb.thumbnail((size, size))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def build_confidence_gauge_html(confidence: float) -> str:
    """A simple circular progress ring built from SVG (no JS needed)."""
    pct = confidence * 100
    radius = 46
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - confidence)
    color = "#22c55e" if pct >= 70 else ("#eab308" if pct >= 40 else "#ef4444")

    return f"""
    <div style="display:flex; justify-content:center; align-items:center; padding:8px;">
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="{radius}" stroke="#33415555" stroke-width="10" fill="none"/>
        <circle cx="60" cy="60" r="{radius}" stroke="{color}" stroke-width="10" fill="none"
                stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}"
                stroke-linecap="round" transform="rotate(-90 60 60)"/>
        <text x="60" y="66" text-anchor="middle" font-size="22" font-weight="bold"
              fill="currentColor">{pct:.1f}%</text>
      </svg>
    </div>
    """


def build_model_card_html(backbone: str, strategy: str, all_results: dict, efficiency: dict) -> str:
    acc = all_results.get(backbone, {}).get(strategy, {}).get(100)
    acc_str = f"{acc*100:.1f}%" if acc is not None else "N/A"
    eff = efficiency.get(backbone, {})
    inference = f"{eff['inference_time_ms']} ms" if eff else "N/A"
    size = f"{eff['size_mb']} MB" if eff else "N/A"

    return f"""
    <div class="glass-card">
      <div class="model-card-row"><span>\U0001F9E0 Backbone</span><b>{BACKBONE_LABELS[backbone]}</b></div>
      <div class="model-card-row"><span>{STRATEGY_ICONS[strategy]} Strategy</span><b>{STRATEGY_LABELS[strategy]}</b></div>
      <div class="model-card-row"><span>\u26A1 Inference</span><b>{inference}</b></div>
      <div class="model-card-row"><span>\U0001F4BE Size</span><b>{size}</b></div>
      <div class="model-card-row"><span>\U0001F3AF Accuracy @ 100%</span><b>{acc_str}</b></div>
    </div>
    """


# ---------------------------------------------------------------------------
# Classify tab logic
# ---------------------------------------------------------------------------

def predict(pil_image, backbone_label, strategy_label, history):
    if history is None:
        history = []

    empty_gauge = build_confidence_gauge_html(0.0)

    if pil_image is None:
        return {}, history, _history_to_display(history), None, "", empty_gauge

    backbone = [k for k, v in BACKBONE_LABELS.items() if v == backbone_label][0]
    strategy = [k for k, v in STRATEGY_LABELS.items() if v == strategy_label][0]

    input_array = preprocess(pil_image)
    result, _ = run_single_inference(backbone, strategy, input_array)

    if result is None:
        msg = {"Model not available": 1.0}
        return msg, history, _history_to_display(history), None, "", empty_gauge

    top_class = max(result, key=result.get)
    top_conf = result[top_class]

    thumb_uri = pil_to_thumbnail_data_uri(pil_image)
    history.append({
        "thumb": thumb_uri,
        "time": time.strftime("%H:%M:%S"),
        "backbone": BACKBONE_LABELS[backbone],
        "strategy": STRATEGY_LABELS[strategy],
        "prediction": top_class,
        "confidence": f"{top_conf*100:.1f}%",
    })

    pdf_path = _generate_pdf_report(pil_image, BACKBONE_LABELS[backbone], STRATEGY_LABELS[strategy], result)

    all_results = load_all_results()
    efficiency = load_efficiency_data()
    model_card_html = build_model_card_html(backbone, strategy, all_results, efficiency)
    gauge_html = build_confidence_gauge_html(top_conf)

    return result, history, _history_to_display(history), pdf_path, model_card_html, gauge_html


def _history_to_display(history):
    if not history:
        return [["", "", "", "", ""]]
    return [
        [f'<img src="{h["thumb"]}" width="40" height="40" style="border-radius:6px;">',
         h["backbone"], h["strategy"], h["prediction"], h["confidence"]]
        for h in history
    ]


def _generate_pdf_report(pil_image, backbone_label, strategy_label, result):
    os.makedirs(os.path.join(BASE_DIR, "tmp"), exist_ok=True)
    img_path = os.path.join(BASE_DIR, "tmp", "report_image.png")
    pil_image.convert("RGB").save(img_path)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "STL-10 Classification Report", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Model: {backbone_label} ({strategy_label})", ln=True)
    pdf.ln(4)
    pdf.image(img_path, w=60)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Predictions:", ln=True)
    pdf.set_font("Helvetica", "", 11)

    sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_result:
        pdf.cell(0, 7, f"{cls}: {prob*100:.1f}%", ln=True)

    pdf_path = os.path.join(BASE_DIR, "tmp", "prediction_report.pdf")
    pdf.output(pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# Compare All tab logic
# ---------------------------------------------------------------------------

def compare_all_models(pil_image):
    if pil_image is None:
        return "<p>Upload an image above to compare all 12 models.</p>"

    input_array = preprocess(pil_image)
    sections = []

    for backbone in BACKBONES:
        cards = []
        for strategy in STRATEGIES:
            result, elapsed_ms = run_single_inference_ephemeral(backbone, strategy, input_array)
            gc.collect()  # cheap insurance: force reclaiming the just-freed session now
            if result is None:
                cards.append(f"""
                <div class="glass-card compare-card">
                  <div class="compare-card-sub">{STRATEGY_ICONS[strategy]} {STRATEGY_LABELS[strategy]}</div>
                  <div class="compare-card-pred">Not available</div>
                </div>
                """)
                continue

            top_class = max(result, key=result.get)
            top_conf = result[top_class]

            cards.append(f"""
            <div class="glass-card compare-card">
              <div class="compare-card-sub">{STRATEGY_ICONS[strategy]} {STRATEGY_LABELS[strategy]}</div>
              <div class="compare-card-pred">{top_class}</div>
              <div class="compare-card-conf">{top_conf*100:.1f}%</div>
              <div class="compare-card-time">{elapsed_ms:.1f} ms</div>
            </div>
            """)

        sections.append(f"""
        <div class="backbone-section">
          <div class="backbone-section-title">{BACKBONE_ICONS[backbone]} {BACKBONE_LABELS[backbone]}</div>
          <div class="compare-grid">{"".join(cards)}</div>
        </div>
        """)

    return "".join(sections)


# ---------------------------------------------------------------------------
# Home page data + sparklines
# ---------------------------------------------------------------------------

def load_all_results():
    results = {}
    for backbone in BACKBONES:
        results[backbone] = {}
        for strategy in STRATEGIES:
            filename = (f"{backbone}_simsiam_linearprobe.json" if strategy == "simsiam"
                        else f"{backbone}_{strategy}.json")
            path = os.path.join(RESULTS_DIR, filename)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            results[backbone][strategy] = {
                int(pct): stats["mean"] for pct, stats in data.items()
            }
    return results


def load_efficiency_data():
    path = os.path.join(RESULTS_DIR, "efficiency.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_full_metrics():
    path = os.path.join(RESULTS_DIR, "full_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def compute_summary_stats(all_results, efficiency):
    best_acc, best_combo = -1, None
    for backbone, strategies in all_results.items():
        for strategy, curve in strategies.items():
            acc = curve.get(100)
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_combo = (backbone, strategy)

    smallest = min(efficiency.items(), key=lambda kv: kv[1]["size_mb"]) if efficiency else None
    fastest = max(efficiency.items(), key=lambda kv: kv[1]["fps"]) if efficiency else None
    total_combos = sum(len(v) for v in all_results.values())

    return {
        "best_combo": (f"{BACKBONE_LABELS[best_combo[0]]} + {STRATEGY_LABELS[best_combo[1]]}"
                        if best_combo else "N/A"),
        "best_acc": f"{best_acc*100:.1f}%" if best_acc >= 0 else "N/A",
        "smallest_model": f"{BACKBONE_LABELS.get(smallest[0], smallest[0])} ({smallest[1]['size_mb']} MB)" if smallest else "N/A",
        "fastest_model": f"{BACKBONE_LABELS.get(fastest[0], fastest[0])} ({fastest[1]['fps']} FPS)" if fastest else "N/A",
        "total_combos": total_combos,
    }


def build_sparkline_data_uri(curve: dict) -> str:
    """Tiny trend line (no axes) showing accuracy across label percentages
    for one backbone+strategy, using the real curve data already on disk."""
    if not curve:
        return ""
    pcts = sorted(curve.keys())
    accs = [curve[p] for p in pcts]

    fig, ax = plt.subplots(figsize=(1.4, 0.4))
    ax.plot(pcts, accs, color="#f59e0b", linewidth=1.5)
    ax.fill_between(pcts, accs, min(accs), color="#f59e0b", alpha=0.15)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def build_ranked_table_html(backbone_filter, strategy_filter, sort_by):
    all_results = load_all_results()
    full_metrics = load_full_metrics()

    rows = []
    for backbone in BACKBONES:
        if backbone_filter != "All" and BACKBONE_LABELS[backbone] != backbone_filter:
            continue
        for strategy in STRATEGIES:
            if strategy_filter != "All" and STRATEGY_LABELS[strategy] != strategy_filter:
                continue
            curve = all_results.get(backbone, {}).get(strategy)
            if not curve or 100 not in curve:
                continue
            m = full_metrics.get(backbone, {}).get(strategy, {})
            rows.append({
                "backbone": f"{BACKBONE_ICONS[backbone]} {BACKBONE_LABELS[backbone]}",
                "strategy": f"{STRATEGY_ICONS[strategy]} {STRATEGY_LABELS[strategy]}",
                "accuracy": curve[100],
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "sparkline": build_sparkline_data_uri(curve),
            })

    if not rows:
        return "<p>No data available yet.</p>"

    sort_key = sort_by.lower()
    rows.sort(key=lambda r: r.get(sort_key) or 0, reverse=True)

    # Fixed, absolute thresholds -- not relative to whatever's currently
    # filtered. Relative (e.g. tertile-of-visible-rows) coloring means the
    # same accuracy value could show green in one filter view and red in
    # another, which is genuinely confusing, not just a style choice.
    def color_class(v):
        if v is None:
            return "gray"
        if v >= 0.75:
            return "green"
        if v >= 0.50:
            return "yellow"
        return "red"

    def metric_row(icon, label, value):
        if value is None:
            return f'<div class="metric-row"><span>{icon} {label}</span><span class="chip gray">N/A</span></div>'
        cls = color_class(value)
        pct = value * 100
        return f"""
        <div class="metric-row">
          <span>{icon} {label}</span>
          <div class="bar-track"><div class="bar-fill {cls}" style="width:{pct:.1f}%"></div></div>
          <span class="chip {cls}">{pct:.1f}%</span>
        </div>
        """

    medals = ["\U0001F947 Rank #1", "\U0001F948 Rank #2", "\U0001F949 Rank #3"]

    cards = []
    for i, r in enumerate(rows):
        rank_label = medals[i] if i < 3 else f"\U0001F3C5 Rank #{i+1}"
        best_class = "best-row" if i == 0 else ""
        sparkline_img = (f'<img class="sparkline" src="{r["sparkline"]}">'
                          if r["sparkline"] else "")

        cards.append(f"""
        <div class="glass-card rank-card {best_class}">
          <div class="rank-badge">{rank_label}</div>
          <div class="rank-card-title">{r['backbone']}</div>
          <div class="rank-card-sub">{r['strategy']}</div>
          {sparkline_img}
          {metric_row("\U0001F4C8", "Accuracy", r['accuracy'])}
          {metric_row("\U0001F3AF", "Precision", r['precision'])}
          {metric_row("\U0001F504", "Recall", r['recall'])}
          {metric_row("\U0001F522", "F1 Score", r['f1'])}
        </div>
        """)

    return f'<div class="ranked-cards">{"".join(cards)}</div>'


def load_benchmark_tables():
    all_results = load_all_results()
    full_metrics = load_full_metrics()
    efficiency = load_efficiency_data()

    rows = []
    for backbone in BACKBONES:
        for strategy in STRATEGIES:
            curve = all_results.get(backbone, {}).get(strategy)
            if not curve:
                continue
            acc_100 = curve.get(100)
            acc_1 = curve.get(1)
            m = full_metrics.get(backbone, {}).get(strategy, {})
            rows.append([
                BACKBONE_LABELS[backbone],
                STRATEGY_LABELS[strategy],
                f"{acc_1*100:.1f}%" if acc_1 else "N/A",
                f"{acc_100*100:.1f}%" if acc_100 else "N/A",
                f"{m['precision']*100:.1f}%" if m else "N/A",
                f"{m['recall']*100:.1f}%" if m else "N/A",
                f"{m['f1']*100:.1f}%" if m else "N/A",
            ])

    efficiency_rows = []
    for backbone, stats in efficiency.items():
        efficiency_rows.append([
            BACKBONE_LABELS.get(backbone, backbone),
            f"{stats['params_millions']}M",
            f"{stats['size_mb']} MB",
            f"{stats['inference_time_ms']} ms",
            f"{stats['fps']}",
        ])

    return rows, efficiency_rows


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

toggle_dark_js = """
() => {
    document.body.classList.toggle('dark');
}
"""

# Programmatic tab-switching for the Home page's nav cards.
switch_to_classify_js = None  # handled via gr.Tabs(selected=...) in Python instead

CUSTOM_CSS = """
/* Warm-dark theme applies ONLY when .dark is active on <body> -- this is
   the actual fix for the unreadable light-mode bug: previously the dark
   background was forced with !important regardless of mode, which broke
   Gradio's own light-theme text/background contrast. In light mode, no
   override applies here, so Gradio's normal light colors are used. */
.dark body, .dark .gradio-container {
    background: linear-gradient(160deg, #1c1410 0%, #2b1f17 100%) !important;
}

.glass-card {
    border-radius: 14px;
    padding: 16px;
    margin: 6px 0;
    border: 1px solid rgba(0,0,0,0.08);
    background: rgba(0,0,0,0.02);
}
.dark .glass-card {
    background: rgba(255, 200, 140, 0.06);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 190, 120, 0.18);
}

.stat-card {
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid rgba(0,0,0,0.08);
    background: rgba(0,0,0,0.02);
}
.dark .stat-card {
    border: 1px solid rgba(255, 190, 120, 0.18);
    background: rgba(255, 200, 140, 0.06);
    backdrop-filter: blur(10px);
}
.stat-card h2 { margin: 0; font-size: 1.6em; }
.stat-card p { margin: 4px 0 0 0; opacity: 0.75; font-size: 0.85em; }

.model-card-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 0.9em;
}

/* Small circular dark/light toggle instead of a full-width button.
   Glows (box-shadow) specifically in light mode, per request. */
#dark-toggle-btn {
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    font-size: 1.3em;
    position: fixed;
    top: 14px;
    right: 14px;
    z-index: 1000;
    box-shadow: 0 0 14px 3px rgba(250, 204, 21, 0.55);
}
.dark #dark-toggle-btn {
    box-shadow: none;
}

/* Fixed-size upload boxes -- the outer box never resizes based on
   whether/what image is uploaded. */
.fixed-upload, .fixed-upload > div {
    height: 280px !important;
}
.fixed-upload img {
    object-fit: contain !important;
    max-height: 260px !important;
}

/* Home page nav cards */
.nav-card {
    white-space: pre-line !important;
    height: 150px !important;
    font-size: 1em !important;
    line-height: 1.5 !important;
    border-radius: 16px !important;
}

.backbone-section { margin: 18px 0; }
.backbone-section-title {
    font-size: 1.1em;
    font-weight: bold;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(150,150,150,0.3);
}

/* Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
.gradio-container, .gradio-container * {
    font-family: 'Inter', sans-serif !important;
}
.gradio-container h1 { font-size: 32px !important; }
.gradio-container h2 { font-size: 22px !important; }

/* Filters styled as cards */
.filter-card {
    border-radius: 12px;
    padding: 10px 14px;
    border: 1px solid rgba(0,0,0,0.08);
    background: rgba(0,0,0,0.02);
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.dark .filter-card {
    border: 1px solid rgba(255, 190, 120, 0.18);
    background: rgba(255, 200, 140, 0.06);
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
.filter-row {
    position: sticky;
    top: 70px;
    z-index: 50;
    padding: 6px 0;
}

/* Ranked result cards (replaces the old table) */
.ranked-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 14px;
    margin: 10px 0;
}
.rank-card {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    animation: materialize 0.4s ease-out;
}
.rank-card:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 20px rgba(0,0,0,0.18);
}
.rank-card.best-row {
    border: 2px solid #eab308 !important;
}
.rank-badge { font-weight: bold; font-size: 0.85em; opacity: 0.85; }
.rank-card-title { font-size: 1.15em; font-weight: 700; margin-top: 4px; }
.rank-card-sub { font-size: 0.85em; opacity: 0.7; margin-bottom: 6px; }
.sparkline { width: 100%; height: 24px; margin: 4px 0 8px 0; }

.metric-row {
    display: grid;
    grid-template-columns: 90px 1fr 55px;
    align-items: center;
    gap: 8px;
    font-size: 0.82em;
    margin: 5px 0;
}
.bar-track {
    height: 8px;
    border-radius: 4px;
    background: rgba(150,150,150,0.25);
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease-out;
}
.bar-fill.green { background: #22c55e; }
.bar-fill.yellow { background: #eab308; }
.bar-fill.red { background: #ef4444; }
.bar-fill.gray { background: #9ca3af; }

.chip {
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.8em;
    font-weight: 600;
    text-align: center;
}
.chip.green { background: rgba(34,197,94,0.18); color: #22c55e; }
.chip.yellow { background: rgba(234,179,8,0.18); color: #eab308; }
.chip.red { background: rgba(239,68,68,0.18); color: #ef4444; }
.chip.gray { background: rgba(156,163,175,0.18); color: #9ca3af; }

.app-footer {
    text-align: center;
    padding: 24px 0 10px 0;
    opacity: 0.7;
    font-size: 0.9em;
}

.compare-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
}
.compare-card { text-align: center; padding: 12px; }
.compare-card-sub { font-size: 0.75em; opacity: 0.7; margin-bottom: 6px; }
.compare-card-pred { font-size: 1.1em; font-weight: bold; text-transform: capitalize; }
.compare-card-conf { font-size: 0.85em; opacity: 0.85; }
.compare-card-time { font-size: 0.7em; opacity: 0.6; }

/* "Materializing" fade/scale-in animation for dashboard charts */
@keyframes materialize {
    from { opacity: 0; transform: scale(0.96); }
    to { opacity: 1; transform: scale(1); }
}
.chart-animate {
    animation: materialize 0.5s ease-out;
}
"""

with gr.Blocks(
    title="Self-Supervised Learning in Low-Data Regime",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css=CUSTOM_CSS,
) as demo:
    dark_toggle = gr.Button("\U0001F4A1", elem_id="dark-toggle-btn")
    dark_toggle.click(fn=None, js=toggle_dark_js)

    history_state = gr.State([])

    with gr.Tabs() as main_tabs:
        with gr.Tab("Home", id="home"):
            gr.Markdown(
                "# Self-Supervised Learning in Low-Data Regime\n"
                "Interactive comparison of Baseline, Data Augmentation, "
                "ImageNet Transfer Learning, and SimSiam -- across ResNet18, "
                "MobileNetV2, and EfficientNet-B0. Full methodology: "
                "[github.com/Abeer-24/low-data-ssl](https://github.com/Abeer-24/low-data-ssl)"
            )

            gr.Markdown(
                "### \U0001F4A1 Motivation\n"
                "Labeled data is expensive to collect; unlabeled data "
                "is comparatively cheap. Self-supervised learning "
                "(SimSiam) exploits large pools of unlabeled data to "
                "learn useful representations before any labels are "
                "seen -- a property that should matter most precisely "
                "when labels are scarce. This project tests that claim "
                "directly: 12 backbone x strategy combinations, "
                "trained and evaluated across six label percentages "
                "(1% to 100%), to find out when self-supervised "
                "pretraining is actually worth the extra compute -- "
                "and when it isn't. **Key finding:** SimSiam only beats "
                "ImageNet transfer learning in the extreme low-data "
                "regime (\u226410% labels); past that, transfer "
                "learning wins, and its lead grows."
            )

            with gr.Row():
                classify_card = gr.Button(
                    "\U0001F9EA  Classify\nTest the model by uploading an "
                    "image with a chosen backbone and strategy.",
                    elem_classes="nav-card",
                )
                compare_card = gr.Button(
                    "\U0001F500  Compare All\nUpload one image and see all "
                    "12 backbone x strategy combinations classify it side "
                    "by side.",
                    elem_classes="nav-card",
                )

            classify_card.click(lambda: gr.Tabs(selected="classify"), outputs=main_tabs)
            compare_card.click(lambda: gr.Tabs(selected="compare"), outputs=main_tabs)

            _all_results = load_all_results()
            _efficiency = load_efficiency_data()
            _stats = compute_summary_stats(_all_results, _efficiency)
            with gr.Row():
                gr.HTML(f"""<div class="stat-card"><h2>{_stats['total_combos']}</h2>
                             <p>Backbone x Strategy combinations trained</p></div>""")
                gr.HTML(f"""<div class="stat-card"><h2>{_stats['best_acc']}</h2>
                             <p>Best accuracy @ 100% labels<br>{_stats['best_combo']}</p></div>""")
                gr.HTML(f"""<div class="stat-card"><h2>{_stats['smallest_model']}</h2>
                             <p>Smallest model</p></div>""")
                gr.HTML(f"""<div class="stat-card"><h2>{_stats['fastest_model']}</h2>
                             <p>Fastest inference</p></div>""")

            gr.Markdown(
                "## \U0001F4CA Performance Comparison\n"
                "*A ranked snapshot of every trained combination.*"
            )
            with gr.Row(elem_classes="filter-row"):
                filter_backbone = gr.Dropdown(
                    choices=["All"] + list(BACKBONE_LABELS.values()), value="All",
                    label="\U0001F9E0 Backbone", elem_classes="filter-card",
                )
                filter_strategy = gr.Dropdown(
                    choices=["All"] + list(STRATEGY_LABELS.values()), value="All",
                    label="\U0001F680 Strategy", elem_classes="filter-card",
                )
                filter_sort = gr.Dropdown(
                    choices=["Accuracy", "Precision", "Recall", "F1"], value="Accuracy",
                    label="\U0001F4CA Sort by", elem_classes="filter-card",
                )
            ranked_table_html = gr.HTML(value=build_ranked_table_html("All", "All", "Accuracy"))
            for f in (filter_backbone, filter_strategy, filter_sort):
                f.change(
                    fn=build_ranked_table_html,
                    inputs=[filter_backbone, filter_strategy, filter_sort],
                    outputs=ranked_table_html,
                )

            gr.Markdown("## \U0001F4BE Model Statistics")
            _, efficiency_rows = load_benchmark_tables()
            gr.Dataframe(
                headers=["\U0001F9E0 Backbone", "\U0001F522 Params", "\U0001F4BE Size", "\u26A1 Inference Time", "\U0001F680 FPS"],
                value=efficiency_rows if efficiency_rows else [["No data available yet", "", "", "", ""]],
            )

        with gr.Tab("Classify", id="classify"):
            with gr.Row():
                with gr.Column():
                    backbone_dropdown = gr.Dropdown(
                        choices=list(BACKBONE_LABELS.values()),
                        value="MobileNetV2",
                        label="\U0001F9E0 Backbone",
                    )
                    strategy_dropdown = gr.Dropdown(
                        choices=list(STRATEGY_LABELS.values()),
                        value="ImageNet Transfer Learning",
                        label="\U0001F680 Strategy",
                    )
                    image_input = gr.Image(
                        type="pil", label="Upload an image", height=280,
                        elem_classes="fixed-upload",
                    )

                    if os.path.exists(EXAMPLES_DIR):
                        example_files = [
                            os.path.join(EXAMPLES_DIR, f)
                            for f in sorted(os.listdir(EXAMPLES_DIR))
                            if f.endswith(".png")
                        ]
                        if example_files:
                            with gr.Accordion("Show example images", open=False):
                                gr.Examples(examples=example_files, inputs=image_input)

                    model_card_html = gr.HTML()

                with gr.Column():
                    with gr.Group(elem_classes="glass-card"):
                        with gr.Row():
                            label_output = gr.Label(num_top_classes=5, label="\U0001F4CA Prediction")
                            gauge_html = gr.HTML()
                    pdf_output = gr.File(label="\U0001F4E5 Download Report", elem_classes="glass-card")

            predict_inputs = [image_input, backbone_dropdown, strategy_dropdown, history_state]

        with gr.Tab("Compare All", id="compare"):
            gr.Markdown(
                "Upload one image and see all 12 backbone x strategy "
                "combinations classify it, grouped by backbone."
            )
            compare_image_input = gr.Image(
                type="pil", label="Upload an image", height=280,
                elem_classes="fixed-upload",
            )
            compare_grid_output = gr.HTML()

            compare_image_input.change(
                fn=compare_all_models,
                inputs=compare_image_input,
                outputs=compare_grid_output,
            )

        with gr.Tab("History", id="history"):
            gr.Markdown("### Prediction history (this session)")
            history_table = gr.Dataframe(
                headers=["Image", "Backbone", "Strategy", "Prediction", "Confidence"],
                datatype=["markdown", "str", "str", "str", "str"],
                value=[["", "", "", "", ""]],
            )

    # Wired here (not inside the Classify tab block) because history_table
    # is defined later, in the History tab.
    predict_outputs = [label_output, history_state, history_table, pdf_output, model_card_html, gauge_html]
    image_input.change(fn=predict, inputs=predict_inputs, outputs=predict_outputs)
    backbone_dropdown.change(fn=predict, inputs=predict_inputs, outputs=predict_outputs)
    strategy_dropdown.change(fn=predict, inputs=predict_inputs, outputs=predict_outputs)

    gr.HTML(
        '<div class="app-footer">Made with \u2764\ufe0f using PyTorch, ONNX Runtime &amp; Gradio<br>'
        '<a href="https://github.com/Abeer-24/low-data-ssl" target="_blank">GitHub</a> | '
        '<a href="https://github.com/Abeer-24/low-data-ssl/blob/main/PROJECT_DOCUMENTATION.md" target="_blank">Documentation</a></div>'
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
