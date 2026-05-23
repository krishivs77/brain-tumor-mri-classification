# Brain Tumor MRI Classification with CNNs and Transfer Learning

## Overview

This project builds a deep learning pipline for classifying brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The project begins with exploratory data analysis and a baseline convolutional neural network (CNN), then progresses toward transfer learning and model interpretability techniques such as Grad-CAM.

The goal is not only to achieve strong classification performance, but also to build a reproducible and interpretable neuro-imaging workflow.

---

## Dataset

Dataset source:

- Brain Tumor MRI Dataset (Kaggle)
- Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset contains labeled MRI scans split into training and testing sets.

---

## Project Structure

```text
brain-tumor-mri-classification/
├── data/
├── notebooks/
├── src/
├── figures/
├── models/
├── reports/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Planned Workflow

### 1. Exploratory Data Analysis
- Class distribution
- MRI visualization
- Image dimension analysis

### 2. Dataset Pipeline
- PyTorch Dataset class
- Image preprocessing
- DataLoader setup

### 3. Baseline CNN
- Convolutional neural network trained from scratch
- Training and validation curves
- Baseline performance evaluation

### 4. Transfer Learning
- Comparison against pretrained architectures
- Fine-tuning experiments

### 5. Model Evaluation & Interpretability
- Confusion matrix
- Per-class accuracy
- Grad-CAM visualizations
- Error analysis

---

## Tech Stack

- Python 3.11
- PyTorch
- torchvision
- NumPy
- matplotlib
- scikit-learn
- Jupyter

---

## Status

Project setup complete.
Exploratory data analysis complete.
Dataset pipline complete.
Baseline CNN training complete.

The baseline CNN reached a best validation accuracy of approximately **92.9%** after 5 epochs. FInal test evaluation has not yet been performed.