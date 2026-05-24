import tempfile

import streamlit as st
import torch
from PIL import Image

from src.inference import get_transform, load_model, predict_image
from src.gradcam import GradCAM, create_gradcam_overlay

st.set_page_config(
    page_title="Brain MRI Classifier",
    layout="centered"
)

st.title("Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI image to classify it using the fine-tuned ResNet18 model."
)

model, device, metadata = load_model()

st.caption(
    f"Model: {metadata['model_name']} | Test accuracy: {metadata['test_accuracy']:.2%}"
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.image(
        image,
        caption="Uploaded MRI",
        width="stretch"
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)

        predicted_class, confidence_scores = predict_image(
            temp_file.name,
            model=model,
            device=device
        )
    
    st.subheader(f"Prediction: {predicted_class}")

    st.subheader("Confidence Scores")

    for class_name, score in confidence_scores.items():
        st.write(f"{class_name}: {score:.4f}")
    
    show_gradcam = st.checkbox("Show Grad-CAM explanation")

    if show_gradcam:
        st.subheader("Grad-CAM Explanation")

        gradcam = GradCAM(
            model=model,
            target_layer=model.layer4[-1]
        )

        original_image = Image.open(temp_file.name).convert("L")
        transform = get_transform()
        input_tensor = transform(original_image).unsqueeze(0).to(device)

        target_class = list(confidence_scores.keys()).index(predicted_class)

        heatmap, _ = gradcam.generate(
            input_tensor=input_tensor,
            target_class=target_class
        )

        alpha = st.slider(
            "Heatmap intensity",
            min_value=0.1,
            max_value=0.8,
            value=0.45,
            step=0.05
        )

        overlay = create_gradcam_overlay(
            original_image=original_image,
            heatmap=heatmap,
            alpha=alpha
        )

        st.image(
            overlay,
            caption="Grad-CAM overlay showing regions that influenced the prediction",
            width="stretch"
        )

        gradcam.remove_hooks()