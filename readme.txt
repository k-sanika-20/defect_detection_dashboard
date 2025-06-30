# Defect Detection in Additive Manufacturing

This Streamlit app uses a U-Net model to segment and visualize six types of defects in grayscale LPBF (Laser Powder Bed Fusion) images.

## Defect Classes
- Swelling
- Spatter
- Misprint
- Over Melting
- Under Melting
- No Defect (background)

## How it works
- Upload a grayscale image.
- The app displays a color-coded overlay of predicted defect regions.

## 🔗 Live Demo
https://defect-detection-dashboard.streamlit.app/

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
