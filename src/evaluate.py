"""
evaluate.py
-----------
Loads the trained model and the held-out test set, runs real inference
(no hardcoded/fake numbers), and reports:

  * Accuracy, Precision, Recall, F1-score (via sklearn)
  * Full classification report
  * Confusion matrix (saved as a PNG heatmap)

All numbers are written to outputs/reports/test_metrics.json so the
Streamlit app's "Analytics" tab can display the model's real, measured
performance instead of anything fabricated.

Run from the project root:
    python src/evaluate.py
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import get_test_generator, load_class_indices


def get_ordered_labels(class_indices):
    """
    class_indices looks like {'fractured': 0, 'not fractured': 1}.
    Returns display labels ordered by index: ['fractured', 'not fractured'].
    """
    return [name for name, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]


def main():
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. Run `python src/train.py` first."
        )

    print("Loading model...")
    model = load_model(config.MODEL_PATH)

    class_indices = load_class_indices()
    if sorted(class_indices.values()) != [0, 1]:
        raise ValueError(
            "Binary sigmoid evaluation requires exactly two classes indexed 0 and 1; "
            f"received {class_indices}."
        )
    labels_ordered = get_ordered_labels(class_indices)
    print(f"Class indices: {class_indices}")

    print("Loading test set...")
    test_gen = get_test_generator()
    if test_gen.class_indices != class_indices:
        raise ValueError(
            "Test directory class indices do not match the mapping saved during training. "
            f"Saved mapping: {class_indices}; test mapping: {test_gen.class_indices}."
        )
    if test_gen.samples == 0:
        raise ValueError(
            f"No images found in {config.TEST_DIR}. Add a test split, or point "
            f"evaluate.py at the validation set if your dataset has no separate test set."
        )

    print(f"Running inference on {test_gen.samples} test images...")
    # For a one-unit sigmoid with binary cross-entropy, model output is the
    # probability of class index 1. flow_from_directory uses the saved
    # alphabetically ordered class indices for both train and test.
    sigmoid_probability_index_1 = model.predict(test_gen, verbose=1).ravel()
    positive_index = 1
    y_pred = (
        sigmoid_probability_index_1 >= config.DECISION_THRESHOLD
    ).astype(int)
    y_true = test_gen.classes

    # ---- Metrics (all computed from real predictions, nothing fabricated) ----
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, pos_label=positive_index, zero_division=0
    )
    recall = recall_score(y_true, y_pred, pos_label=positive_index, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_index, zero_division=0)
    try:
        # ROC-AUC's positive class is explicitly index 1, matching the
        # sigmoid probability. Here that means "not fractured".
        y_true_positive = (y_true == positive_index).astype(int)
        auc = roc_auc_score(y_true_positive, sigmoid_probability_index_1)
    except ValueError:
        auc = None  # happens only if the test set has a single class present

    report_text = (
        f"Positive class for aggregate precision/recall/F1 and ROC-AUC: "
        f"{labels_ordered[positive_index]} (index {positive_index})\n\n"
        + classification_report(
        y_true, y_pred, labels=[0, 1], target_names=labels_ordered,
        digits=4, zero_division=0,
        )
    )
    print("\n===== Classification Report =====")
    print(report_text)

    with open(config.CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write(report_text)

    metrics = {
        "num_test_samples": int(test_gen.samples),
        "class_indices": class_indices,
        "positive_class": labels_ordered[positive_index],
        "sigmoid_probability_class": labels_ordered[1],
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "decision_threshold": config.DECISION_THRESHOLD,
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics -> {config.METRICS_PATH}")

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    if auc is not None:
        print(f"ROC-AUC  : {auc:.4f}")

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels_ordered, yticklabels=labels_ordered,
        cbar=False,
    )
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {config.CONFUSION_MATRIX_PLOT_PATH}")


if __name__ == "__main__":
    main()
