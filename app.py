"""
app.py — Streamlit web UI for the Vehicle Type Classifier (v2).

UPGRADE NOTES:
  - Uses MobileNetV2 preprocess_input to match the transfer-learning pipeline.
  - Shows top-3 predictions alongside the bar chart for better interpretability.

Run:
    streamlit run app.py
"""

import os
import tempfile
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

from src.preprocess import IMG_SIZE, CLASS_NAMES
from src.decision import get_decision, get_decision_icon

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Vehicle Classifier v2", page_icon="🚗", layout="centered")

MODEL_PATH = "model/cnn_model.h5"


@st.cache_resource
def load_trained_model():
    """Load the model once and cache it across reruns."""
    if not os.path.isfile(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)


# ── Header ───────────────────────────────────────────────────────────────────
st.title("🚗 Vehicle Type Classification System v2")
st.markdown(
    "Upload an image of a vehicle and the **MobileNetV2** transfer-learning model "
    "will classify it as **Car, Truck, Bike, Bus,** or **Ambulance**."
)

model = load_trained_model()

if model is None:
    st.error(
        f"Model file not found at `{MODEL_PATH}`. "
        "Train the model first:\n```\npython src/train.py\n```"
    )
    st.stop()

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save to a temp file so Keras can load it
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp, format="JPEG")
        tmp_path = tmp.name

    # Preprocess — must match training pipeline (MobileNetV2)
    img = load_img(tmp_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Classifying..."):
        predictions = model.predict(img_batch, verbose=0)[0]

    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx])
    predicted_class = CLASS_NAMES[predicted_idx]
    decision = get_decision(confidence)
    icon = get_decision_icon(confidence)

    # ── Results ──────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Class", predicted_class.capitalize())
    col2.metric("Confidence", f"{confidence:.1%}")
    col3.metric("Decision", f"{decision} {icon}")

    # Ambulance alert
    if predicted_class.lower() == "ambulance":
        st.warning("⚠️ **Emergency Vehicle Detected** 🚑", icon="🚨")

    # Top-3 predictions table
    st.markdown("#### Top Predictions")
    sorted_idx = np.argsort(predictions)[::-1][:3]
    for rank, idx in enumerate(sorted_idx, 1):
        pct = predictions[idx] * 100
        bar = "█" * int(pct / 2)
        st.text(f"  {rank}. {CLASS_NAMES[idx].capitalize():<12s}  {pct:5.1f}%  {bar}")

    # Probability bar chart
    st.markdown("#### Class Probabilities")
    prob_data = {name.capitalize(): float(predictions[i]) for i, name in enumerate(CLASS_NAMES)}
    st.bar_chart(prob_data)

    # Clean up temp file
    os.unlink(tmp_path)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built for Hackathon — Vehicle Type Classification System v2 | MobileNetV2 + TensorFlow/Keras")
