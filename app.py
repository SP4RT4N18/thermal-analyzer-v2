import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Thermal Image Analyzer", layout="centered")
st.title("🔥 Thermal Image Analyzer")
st.write("Upload a thermal image to detect hot spots and analyze faults.")

uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and show original image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to grayscale
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Simulate temperature range
    min_temp = int(np.min(img_gray) * 0.7)
    max_temp = int(np.max(img_gray) * 1.3)
    st.write(f"🧊 Estimated Min Temp: {min_temp}°C")
    st.write(f"🔥 Estimated Max Temp: {max_temp}°C")

    # Threshold to find hot areas
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image
    img_contour = img_array.copy()
    cv2.drawContours(img_contour, contours, -1, (255, 0, 0), 2)
    st.image(img_contour, caption='Hot Zones Detected', use_column_width=True)

    # Add heatmap visualization
    heatmap = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    st.image(heatmap, caption="Simulated Heatmap View", use_column_width=True)

    # Fault Report
    st.subheader("📝 Fault Analysis Report")
    st.write(f"- {len(contours)} hot zone(s) detected.")

    if len(contours) == 0:
        st.success("✅ No overheating detected.")
    elif len(contours) < 3:
        st.warning("⚠️ Minor hot zones detected. Check modules.")
    else:
        st.error("🚨 Multiple hot areas detected! Possible fault in the panel.")

    # ✅ Now define heatmap_copy AFTER heatmap is created
    heatmap_copy = heatmap.copy()
    cv2.drawContours(heatmap_copy, contours, -1, (255, 255, 255), 2)

    # Encode image to JPEG for download
    is_success, buffer = cv2.imencode(".jpg", heatmap_copy)
    byte_io = io.BytesIO(buffer)

    # Add download button
    st.download_button(
        label="⬇️ Download Analysis Image",
        data=byte_io,
        file_name="thermal_analysis_result.jpg",
        mime="image/jpeg"
    ) 