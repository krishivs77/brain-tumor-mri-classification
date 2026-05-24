import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from config import CLASSES, MODELS_DIR

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_model(model_path=MODELS_DIR / "resnet18_finetuned.pth"):
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

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model, device

def predict_image(image_path, model=None, device=None):
    if model is None or device is None:
        model, device = load_model()
    
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
