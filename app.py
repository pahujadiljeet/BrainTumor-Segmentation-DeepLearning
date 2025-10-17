import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO
from skimage import measure
import time

# --- Load Model ---
model = load_model("models/brain_tumor_segmentation_model.h5", compile=False)

# --- Page Setup ---
st.set_page_config(page_title="🧠 Brain Tumor Segmentation", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: white; }
    .stButton>button { background-color: #FF4B4B; color: white; font-weight: bold; border-radius: 10px; }
    .footer { text-align: center; color: gray; margin-top: 30px; font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Brain Tumor Segmentation (4-Modality Deep Learning Model)")
st.markdown("### Upload 4 MRI Modalities Below")

# --- Upload four modalities ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    flair = st.file_uploader("FLAIR", type=["png","jpg","jpeg"])
with col2:
    t1 = st.file_uploader("T1", type=["png","jpg","jpeg"])
with col3:
    t1ce = st.file_uploader("T1CE", type=["png","jpg","jpeg"])
with col4:
    t2 = st.file_uploader("T2", type=["png","jpg","jpeg"])

def read_img(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (240, 240))
    return img

if flair and t1 and t1ce and t2:
    # --- Read all images ---
    flair_img = read_img(flair)
    t1_img = read_img(t1)
    t1ce_img = read_img(t1ce)
    t2_img = read_img(t2)

    # --- Thumbnails Preview ---
    st.markdown("### 🎨 Preview of Uploaded Modalities")
    cols = st.columns(4)
    with cols[0]:
        st.image(flair_img, caption="FLAIR", use_container_width=True, channels="GRAY")
    with cols[1]:
        st.image(t1_img, caption="T1", use_container_width=True, channels="GRAY")
    with cols[2]:
        st.image(t1ce_img, caption="T1CE", use_container_width=True, channels="GRAY")
    with cols[3]:
        st.image(t2_img, caption="T2", use_container_width=True, channels="GRAY")

    # --- Stack 4 channels ---
    input_img = np.stack([flair_img, t1_img, t1ce_img, t2_img], axis=-1)
    input_img = input_img / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # --- Progress Bar ---
    st.markdown("### 📈 Model Prediction Progress")
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    # --- Prediction ---
    preds = model.predict(input_img)
    mask = np.squeeze(preds[0])
    binary_mask = (mask > 0.5).astype(np.uint8)

    # --- Refinement ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_area = 500
    refined_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            refined_mask[labels == i] = 1

    label_img = measure.label(refined_mask)
    regions = measure.regionprops(label_img)
    if len(regions) > 0:
        largest = max(regions, key=lambda r: r.area)
        final_mask = np.zeros_like(refined_mask)
        for coord in largest.coords:
            final_mask[coord[0], coord[1]] = 1
    else:
        final_mask = refined_mask

    # --- Overlay mask on FLAIR ---
    overlay = cv2.cvtColor(flair_img, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    alpha = 0.4
    overlay = np.where(
        final_mask[:, :, None].astype(bool),
        cv2.addWeighted(overlay, 1 - alpha, red, alpha, 0),
        overlay
    )

    # --- Side-by-side comparison ---
    st.markdown("### 🖼️ Original vs Predicted Mask")
    c1, c2 = st.columns(2)
    with c1:
        st.image(flair_img, caption="Original FLAIR (Raw)", use_container_width=True, channels="GRAY")
    with c2:
        st.image(overlay, caption="Predicted Tumor Mask Overlay", use_container_width=True)

    st.success("✅ Segmentation Complete!")

    # --- Footer ---
    st.markdown("<div class='footer'>Developed by <b>Diljeet Khoobchand</b> | Brain Tumor Segmentation Project 🧠</div>", unsafe_allow_html=True)
