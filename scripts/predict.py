import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference import load_model, predict_image


def main():
    parser = argparse.ArgumentParser(
        description="Run brain MRI tumor classification on a single image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the MRI image file."
    )

    parser.add_argument(
        "--show-probabilities",
        action="store_true",
        help="Print confidence scores for all classes."
    )

    args = parser.parse_args()

    model, device, metadata = load_model()

    predicted_class, confidence_scores = predict_image(
        args.image,
        model=model,
        device=device
    )

    print(f"Model: {metadata['model_name']}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence_scores[predicted_class]:.4f}")

    if args.show_probabilities:
        print("\nAll class probabilities:")
        for class_name, score in confidence_scores.items():
            print(f"{class_name}: {score:.4f}")

if __name__ == "__main__":
    main()