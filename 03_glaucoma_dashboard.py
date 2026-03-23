import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

#PAGE CONFIGURATION
st.set_page_config(
    page_title="Glaucoma AI Diagnostic Tool",
    page_icon="👁️",
    layout="centered"
)

#SIDEBAR
with st.sidebar:
    st.header("Technical Specifications")
    st.info("Analysis Engine: **MobileNetV2**")
    st.write("**Environment:** Local PC")
    st.write("**Data Folder:** /results")
    st.markdown("---")
    st.caption("AI focus mapping uses Grad-CAM to visualize neural network attention.")

st.title("👁️ Glaucoma Detection AI")
st.write("Upload a retinal fundus image for precise probability analysis and feature mapping.")
st.markdown("---")

#MODEL LOADING
@st.cache_resource
def load_glaucoma_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "results", "glaucoma_model.h5")
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_glaucoma_model()

if model is None:
    st.error("❌ **Error:** 'glaucoma_model.h5' not found in the /results folder. Please run glaucoma.py first.")
    st.stop()

#GRAD-CAM ALGORITHM
def make_gradcam_heatmap(img_tensor, full_model, last_conv_layer_name='out_relu'):
    base_model = full_model.layers[0]
    conv_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = conv_model(img_tensor)
        tape.watch(conv_outputs)
        x = base_outputs
        for layer in full_model.layers[1:]:
            x = layer(x)
        preds = x
    grads = tape.gradient(preds, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

#INTERFACE: UPLOAD & INFERENCE
uploaded_file = st.file_uploader("Upload Retinal Scan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)

    with st.spinner('🔬 Running diagnostic inference...'):
        prediction_raw = model.predict(img_tensor)[0][0]
        prediction = float(prediction_raw)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Scan**")
        st.image(image, use_container_width=True)

    with col2:
        # VISUAL FEEDBACK
        st.markdown("**AI Focus Map**")
        heatmap = make_gradcam_heatmap(img_tensor, model)
        original_img_array = np.array(image).astype("float32")

        heatmap_resized = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype("float32")

        superimposed_img = np.clip(heatmap_rgb * 0.4 + original_img_array * 0.6, 0, 255).astype("uint8")
        st.image(superimposed_img, use_container_width=True)

    st.markdown("---")

    #DIAGNOSTIC RESULTS
    st.subheader("📊 Diagnostic Summary")

    if prediction > 0.8 or prediction < 0.2:
        status_text, status_color = "Certain", "green"
    elif 0.4 <= prediction <= 0.6:
        status_text, status_color = "Borderline / Review Required", "orange"
    else:
        status_text, status_color = "Analyzing Features", "blue"

    if prediction > 0.5:
        st.error(f"⚠️ **Result: High Risk of Glaucoma (GON+)**")
        m1, m2 = st.columns(2)
        m1.metric("Glaucoma Probability", f"{prediction * 100:.4f}%")
        m2.metric("Confidence Score", f"{prediction:.4f}")
    else:
        st.success(f"✅ **Result: Low Risk (Normal / GON-)**")
        m1, m2 = st.columns(2)
        m1.metric("Healthy Score", f"{(1 - prediction) * 100:.4f}%")
        m2.metric("Confidence Score", f"{prediction:.4f}")

    st.markdown(f"**AI Decision Stability:** :{status_color}[{status_text}]")

    with st.expander("ℹ️ How to interpret these numbers?"):
        st.write(f"""
        - **Probability/Healthy Score:** The percentage likelihood that the scan falls into a specific category.
        - **Confidence Score:** The raw AI output (0.0000 to 1.0000). Values near **1.0000** suggest high glaucoma risk; values near **0.0000** suggest a healthy scan.
        - **Stability:** A 'Certain' status means the AI has a very clear numerical polarization in its decision.
        """)