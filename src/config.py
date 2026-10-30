"""
config.py
---------
Central configuration for the Bone Fracture Detection project.
Every other script (data_loader, model, train, evaluate, predict, app)
imports paths and hyperparameters from here, so changing a setting in
one place updates the whole pipeline.
"""

import os

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT points to the folder that contains "src", "data", "models", etc.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "bone_fracture_model.keras")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "bone_fracture_model_best.keras")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# Separate YOLO model paths. These never replace the classifier.
DETECTOR_MODEL_DIR = os.path.join(MODEL_DIR, "fracture_detector")
DETECTOR_ONNX_MODEL_PATH = os.path.join(DETECTOR_MODEL_DIR, "yolov7-p6-bonefracture.onnx")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

HISTORY_PATH = os.path.join(REPORTS_DIR, "training_history.json")
METRICS_PATH = os.path.join(REPORTS_DIR, "test_metrics.json")
CLASSIFICATION_REPORT_PATH = os.path.join(REPORTS_DIR, "classification_report.txt")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(PLOTS_DIR, "confusion_matrix.png")
ACCURACY_PLOT_PATH = os.path.join(PLOTS_DIR, "accuracy_curve.png")
LOSS_PLOT_PATH = os.path.join(PLOTS_DIR, "loss_curve.png")

# ---------------------------------------------------------------------------
# Image / model hyperparameters
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)      # required input size for MobileNetV2
IMG_CHANNELS = 3
BATCH_SIZE = 32
SEED = 42

# Training schedule
INITIAL_EPOCHS = 15        # phase 1: train the classification head, base frozen
FINE_TUNE_EPOCHS = 10      # phase 2: unfreeze top of the base and fine-tune
FINE_TUNE_AT_LAYER = 100   # unfreeze layers from this index onward in the base model

INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5

# Decision threshold for turning a sigmoid probability into a class label.
DECISION_THRESHOLD = 0.5

# Optional Grad-CAM visualization settings. This is an attention
# explanation, not a trained fracture-localization model or a boundary.
LOCALIZATION_ENABLED = True
LOCALIZATION_CONFIDENCE_THRESHOLD = 0.5
LOCALIZATION_OVERLAY_ALPHA = 0.38
LOCALIZATION_COLORMAP = "JET"

# YOLOv7 ONNX detector settings.
YOLO_CONFIDENCE_THRESHOLD = 0.25
DETECTOR_IMAGE_SIZE = 640  # matches the upstream inference_onnx.py preprocessing
DETECTOR_MAX_DETECTIONS = 300
DETECTOR_BOX_THICKNESS = 3
DETECTOR_CLASS_ID_FRACTURE = 3  # upstream detector class map

# Human-readable label used by the app / reports for each side of the
# classification. Actual index-to-class mapping is NOT hardcoded here —
# it is learned from the dataset's folder names at training time and
# stored in CLASS_INDICES_PATH, so the pipeline works even if a dataset
# uses different folder naming (e.g. "Fractured" vs "fractured").
POSITIVE_KEYWORDS = ("fracture",)   # any class folder containing this (and not "not") = positive
NEGATIVE_KEYWORDS = ("not", "normal", "healthy")

for _dir in (MODEL_DIR, OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR):
    os.makedirs(_dir, exist_ok=True)
