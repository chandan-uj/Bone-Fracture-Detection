"""
predict.py
----------
Single-image inference. Used by:
  * app.py (Streamlit UI)
  * command line: python src/predict.py --image path/to/xray.jpg

Handles preprocessing (resize + normalize) exactly the same way as
training/evaluation, so predictions are consistent with the reported
test-set metrics.
"""

import os
import sys
import json
import argparse

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_class_indices, resolve_label_name


def preprocess_image(image: Image.Image, img_size=None):
    """
    Takes a PIL image (any mode/size), converts to RGB, resizes to the
    model's expected input size, and normalizes pixel values to [0, 1] —
    identical preprocessing to what ImageDataGenerator(rescale=1/255) did
    during training.
    """
    img_size = img_size or config.IMG_SIZE
    image = image.convert("RGB")
    image = image.resize(img_size)
    array = np.array(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)  # add batch dimension
    return array


def load_trained_model(model_path=None):
    model_path = model_path or config.MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Train it first with "
            f"`python src/train.py`."
        )
    return load_model(model_path)


def predict_image(model, image: Image.Image, class_indices=None):
    """
    Runs inference on a single PIL image.

    Returns a dict:
        {
          "label": "Fracture Detected" | "No Fracture Detected",
          "raw_class": the raw dataset folder name predicted,
          "confidence": float in [0, 1]  -> confidence in the predicted class
          "probability_fracture": float in [0, 1] -> raw sigmoid-derived
                                    probability of the "fracture" class
        }
    """
    class_indices = class_indices or load_class_indices()
    if sorted(class_indices.values()) != [0, 1]:
        raise ValueError(
            "Binary sigmoid prediction requires exactly two classes indexed 0 and 1; "
            f"received {class_indices}."
        )
    idx_to_class = {v: k for k, v in class_indices.items()}

    array = preprocess_image(image)
    # A single sigmoid output is P(class index 1), not inherently P(fracture).
    probability_index_1 = float(model.predict(array, verbose=0)[0][0])

    predicted_index = (
        1 if probability_index_1 >= config.DECISION_THRESHOLD else 0
    )
    predicted_class_name = idx_to_class[predicted_index]
    label = resolve_label_name(predicted_class_name)

    # Confidence in the predicted class (not just raw P(class 1))
    confidence = (
        probability_index_1 if predicted_index == 1 else 1 - probability_index_1
    )

    # Also report probability specifically for the "fracture" class,
    # regardless of which index it happens to be, for a stable UI display.
    fracture_index = next(
        (idx for name, idx in class_indices.items() if "fracture" in name.lower()
         and "not" not in name.lower()),
        None,
    )
    if fracture_index is not None:
        probability_fracture = (
            probability_index_1 if fracture_index == 1
            else 1 - probability_index_1
        )
    else:
        probability_fracture = None

    return {
        "label": label,
        "raw_class": predicted_class_name,
        "predicted_index": predicted_index,
        "class_index": predicted_index,
        "confidence": confidence,
        # `probability` means confidence in the predicted class; the raw
        # sigmoid probability is included separately to remove ambiguity.
        "probability": confidence,
        "output_probability_index_1": probability_index_1,
        "probability_fracture": probability_fracture,
    }


def main():
    parser = argparse.ArgumentParser(description="Run fracture detection on a single X-ray image.")
    parser.add_argument("--image", required=True, help="Path to an X-ray image (.jpg/.png).")
    parser.add_argument("--model", default=config.MODEL_PATH, help="Path to the trained model file.")
    args = parser.parse_args()

    model = load_trained_model(args.model)
    class_indices = load_class_indices()
    image = Image.open(args.image)

    result = predict_image(model, image, class_indices)

    print(json.dumps(result, indent=2))
    print(f"\nResult: {result['label']}  (confidence: {result['confidence']*100:.2f}%)")


if __name__ == "__main__":
    main()
