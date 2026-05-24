import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from src.config import CLASSES, MODELS_DIR
from src.transforms import get_transform

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model(model_path=MODELS_DIR / "resnet18_brain_mri_v1.pth"):
    device = get_device()

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, len(CLASSES))
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    metadata = {
        "class_names": checkpoint.get("class_names", CLASSES),
        "image_size": checkpoint.get("image_size", 224),
        "model_name": checkpoint.get("model_name", "resnet18_finetuned"),
        "validation_accuracy": checkpoint.get("validation_accuracy"),
        "test_accuracy": checkpoint.get("test_accuracy"),
        "notes": checkpoint.get("notes")
    }

    return model, device, metadata

def predict_image(image_path, model=None, device=None):
    if model is None or device is None:
        model, device, _ = load_model()
    
    transform = get_transform()

    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    predicted_index = probabilities.argmax().item()
    predicted_class = CLASSES[predicted_index]

    confidence_scores = {
        class_name: float(probabilities[idx].cpu())
        for idx, class_name in enumerate(CLASSES)
    }

    return predicted_class, confidence_scores
