# streamlit_predict.py
"""
Streamlit app for inference. Run with:
streamlit run streamlit_predict.py --server.fileWatcherType none
"""

import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import pandas as pd

# SETTINGS
root_dir = r"C:/Users/hitech/OneDrive/Documents/X ray"
MODEL_PATH = os.path.join(root_dir, "efficientnet_b0_best.pth")  # or final
CSV_PATH = os.path.join(root_dir, "Data_Entry_2017.csv")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transforms
transform = T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
                       T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@st.cache_resource
def load_labels_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    def parse_labels(x):
        labs = [p.strip() for p in str(x).split("|") if p.strip()!=""]
        return [] if "No Finding" in labs else labs
    df["labels_list"] = df["Finding Labels"].apply(parse_labels)
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit(df["labels_list"])
    return list(mlb.classes_)

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
        return None, None
    ck = torch.load(model_path, map_location="cpu")
    # labels: try checkpoint else CSV
    labels = ck.get("all_labels") if isinstance(ck, dict) else None
    if labels is None:
        labels = load_labels_from_csv(CSV_PATH)
    num_classes = len(labels)
    # build efficientnet-b0 architecture
    model = models.efficientnet_b0(weights=None)
    in_ft = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_ft, num_classes)
    # load weights
    if isinstance(ck, dict) and "model_state_dict" in ck:
        model.load_state_dict(ck["model_state_dict"])
    else:
        model.load_state_dict(ck)
    model.to(DEVICE); model.eval()
    return model, labels

st.title("EfficientNet-B0 — Chest X-ray inference")
st.write("Upload image (png/jpg). Run Streamlit with: streamlit run streamlit_predict.py --server.fileWatcherType none")

model, labels = load_model(MODEL_PATH)
if model is None:
    st.stop()

uploaded = st.file_uploader("Upload X-ray image", type=["png","jpg","jpeg"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input image", use_column_width=True)
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out).cpu().numpy()[0]
    # show
    dfp = pd.DataFrame({"label": labels, "prob": probs})
    dfp = dfp.sort_values("prob", ascending=False).reset_index(drop=True)
    st.subheader("Predictions")
    for _, row in dfp.iterrows():
        st.write(f"{row['label']}: {row['prob']:.3f}")
    st.bar_chart(dfp.set_index("label")["prob"])
