import streamlit as st
import torch
import cv2
import numpy as np
import urllib.request
import os
from PIL import Image
from unet_model import UNet

# === Constants ===
MODEL_PATH = "after_melting_unet_model_2.pth"
MODEL_URL = "https://github.com/k-sanika-20/defect_detection_dashboard/releases/download/v1.0/after_melting_unet_model_2.pth"

colors = [
    (128, 128, 128),  # Background [0] - No Defect
    (255, 0, 0),      # [5] Swelling
    (0, 255, 0),      # [8] Spatter
    (0, 0, 255),      # [9] Misprint
    (255, 255, 0),    # [10] Over Melting
    (255, 0, 255),    # [11] Under Melting
]

# === Download model if not present ===
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("Model downloaded successfully!")

download_model()

# === Load model ===
@st.cache_resource
def load_model():
    model = UNet(in_channels=1, out_channels=6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# === UI Title ===
st.title("🧠 Defect Detection Dashboard")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose an option", ["Upload Image"])

# === Preprocessing ===
def preprocess_image(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (256, 256))
    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    return tensor, gray.shape[::-1]  # (width, height)

# === Postprocessing: Create overlay ===
def create_overlay(pred_mask, original_shape, original_image):
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    masks = pred_mask.squeeze().cpu().numpy()
    masks = np.array([cv2.resize(m, original_shape) for m in masks])  # Resize masks back

    overlay = np.zeros((*original_shape[::-1], 3), dtype=np.uint8)  # (H, W, 3)
    for i, mask in enumerate(masks):
        color_mask = np.stack([mask * c for c in colors[i]], axis=-1)
        overlay = np.where(mask[..., None] > 0, color_mask.astype(np.uint8), overlay)

    blend = cv2.addWeighted(original_image, 0.6, overlay, 0.4, 0)
    return blend

# === Main UI ===
if app_mode == "Upload Image":
    st.header("📷 Upload an Image for Defect Detection")
    uploaded_file = st.file_uploader("Choose a grayscale image (JPG or PNG)...", type=["jpg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("❌ Could not decode the image.")
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

            input_tensor, original_size = preprocess_image(image_rgb)

            with torch.no_grad():
                output = model(input_tensor)

            overlay_img = create_overlay(output, original_size, image_rgb)
            st.image(overlay_img, caption="🩻 Overlayed Defect Mask", use_container_width=True)

# === Sidebar Footer ===
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("This app segments and visualizes defect regions in L-PBF parts using a U-Net model.")

# === Legend Display ===
st.sidebar.markdown("### 🟦 Defect Class Legend")
legend = {
    "No Defect [0]": (128, 128, 128),
    "Swelling [5]": (255, 0, 0),
    "Spatter [8]": (0, 255, 0),
    "Misprint [9]": (0, 0, 255),
    "Over Melting [10]": (255, 255, 0),
    "Under Melting [11]": (255, 0, 255),
}

for label, color in legend.items():
    hex_color = '#%02x%02x%02x' % color
    st.sidebar.markdown(
        f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
        f"<div style='width: 20px; height: 20px; background-color: {hex_color}; "
        f"margin-right: 10px; border: 1px solid #000;'></div>"
        f"{label}</div>",
        unsafe_allow_html=True
    )
