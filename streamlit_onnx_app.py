# streamlit_onnx_app.py
"""
Streamlit app for ONNX inference (EfficientNet-B0 example).
Usage:
  streamlit run streamlit_onnx_app.py
Set MODEL_URL to your onnx file URL (GitHub raw, HuggingFace raw, or Google Drive direct).
If HuggingFace private, use HUGGINGFACE_TOKEN in Streamlit secrets (handled below).
"""

import streamlit as st
from pathlib import Path
import requests
import os
from io import BytesIO
from PIL import Image
import numpy as np
import onnxruntime as ort
import pandas as pd

# ------------- SETTINGS -------------
IMG_SIZE = 224

# Change this to where you uploaded the ONNX file:
# Example GitHub raw: "https://raw.githubusercontent.com/username/repo/main/efficientnet_b0.onnx"
# Example HF raw: "https://huggingface.co/username/repo/resolve/main/efficientnet_b0.onnx"
MODEL_URL = "https://huggingface.co/zorigoo-bayar/xray/resolve/main/efficientnet_b0.onnx"

CACHE_DIR = Path("/tmp/onnx_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_ONNX = CACHE_DIR / "efficientnet_b0.onnx"

DEFAULT_LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion",
    "Emphysema","Fibrosis","Hernia","Infiltration","Mass","Nodule",
    "Pleural_Thickening","Pneumonia","Pneumothorax"
]

st.set_page_config(page_title="X-ray ONNX Inference", layout="wide")
st.title("🩻 Chest X-ray — ONNX Inference")

st.sidebar.header("Model source")
st.sidebar.write("Model URL (edit if needed):")
MODEL_URL = st.sidebar.text_input("MODEL_URL", value=MODEL_URL)

# ------------------ download with token support ------------------
def download_file(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 1024:
        return str(dest)
    # token from secrets or env (for private HF)
    token = None
    try:
        token = st.secrets["HUGGINGFACE_TOKEN"]
    except Exception:
        token = os.environ.get("HUGGINGFACE_TOKEN", None)

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(url, headers=headers, stream=True, timeout=120)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed download {url}: {e}")
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(8192):
            if chunk:
                f.write(chunk)
    return str(dest)

# ------------------ load ONNX session (cached) ------------------
@st.cache_resource
def load_session(onnx_path):
    # use CPU execution provider; GPU EP optional if available
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess

# Download model
try:
    onnx_path = download_file(MODEL_URL, LOCAL_ONNX)
    st.sidebar.success(f"ONNX downloaded: {onnx_path}")
except Exception as e:
    st.error(f"Model download failed: {e}")
    st.stop()

# Load session
try:
    sess = load_session(onnx_path)
except Exception as e:
    st.error(f"Failed to load ONNX model: {e}")
    st.stop()

# allow user label override (optional)
labels = st.sidebar.text_area("Labels (comma-separated) — leave blank for default", value=",".join(DEFAULT_LABELS))
labels = [l.strip() for l in labels.split(",") if l.strip() != ""]

# ------------------ preprocessing ------------------
def preprocess_pil(img: Image.Image, size=IMG_SIZE):
    img = img.convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std = np.array([0.229,0.224,0.225], dtype=np.float32)
    arr = (arr - mean) / std
    # HWC -> NCHW
    arr = np.transpose(arr, (2,0,1))[None, :].astype(np.float32)
    return arr

# ------------------ UI ------------------
uploaded = st.file_uploader("Upload chest X-ray (png/jpg)", type=["png","jpg","jpeg"])
if uploaded is None:
    st.info("Upload an image to run inference.")
    st.stop()

img = Image.open(uploaded)
st.image(img, caption="Input image", use_column_width=True)

x = preprocess_pil(img)
# run
input_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name
res = sess.run([out_name], {input_name: x})[0]  # shape (1, num_classes)
probs = 1.0 / (1.0 + np.exp(-res))  # sigmoid
probs = probs[0]

# show
df = pd.DataFrame({"label": labels, "prob": probs})
df = df.sort_values("prob", ascending=False).reset_index(drop=True)
st.subheader("Predictions")
st.table(df.head(20))
st.subheader("Probability chart")
st.bar_chart(df.set_index("label")["prob"])
