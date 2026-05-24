import tempfile

import streamlit as st
from PIL import Image

from src.inference import load_model, predict_image

st.set_page_config(
    page_title="Brain MRI Classifier",
    layout="centered"
)

st.title("Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI image to classify it using the fine-tuned ResNet18 model."
)

model, device = load_model()

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