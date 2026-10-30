# 🦴 Bone Fracture Detection Using X-ray Images

An AI-powered screening application that analyzes a bone X-ray image and predicts
**Fracture Detected** or **No Fracture Detected**, with a confidence score — built with
TensorFlow/Keras (MobileNetV2 transfer learning) and served through a Streamlit web app.

> ⚠️ **This is a screening/research support tool, not a medical device. It does not replace a
> radiologist or physician. See the full disclaimer at the bottom of this file.**

---

## 1. Project Structure

```
bone-fracture-detection/
├── data/                          # Dataset goes here (see Section 2)
│   ├── train/
│   │   ├── fractured/
│   │   └── not_fractured/
│   ├── val/
│   │   ├── fractured/
│   │   └── not_fractured/
│   └── test/
│       ├── fractured/
│       └── not_fractured/
├── models/                        # Trained model + class-index mapping (generated)
│   ├── bone_fracture_model.keras
│   ├── bone_fracture_model_best.keras
│   └── class_indices.json
├── outputs/
│   ├── plots/                     # accuracy_curve.png, loss_curve.png, confusion_matrix.png
│   └── reports/                   # training_history.json, test_metrics.json, classification_report.txt
├── src/
│   ├── detect_fracture.py         # optional trained YOLO box inference
│   ├── config.py                  # all paths & hyperparameters
│   ├── data_loader.py             # dataset validation, generators, augmentation
│   ├── model.py                   # MobileNetV2 transfer-learning model (+ CNN baseline)
│   ├── train.py                   # two-phase training script
│   ├── evaluate.py                # test-set evaluation, metrics, confusion matrix
│   └── predict.py                 # single-image inference (CLI + used by the app)
├── app.py                         # Streamlit application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Optional: Real YOLOv7 Fracture Localization

The MobileNetV2 classifier remains the project's classification pipeline. YOLOv7 localization
is a separate ONNX Runtime pipeline using the pretrained GRAZPEDWRI-DX detector. The supplied
ZIP does **not** contain model weights; the official ONNX model is available from the documented
release:

`https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection/releases/download/trained-models/yolov7-p6-bonefracture.onnx`

The ZIP provides annotated evaluation outputs and example predictions, but not the trained
weights. The checkpoint has been downloaded from the official release to
`models/fracture_detector/yolov7-p6-bonefracture.onnx`. The upstream model has nine classes:
`0 boneanomaly`, `1 bonelesion`, `2 foreignbody`, `3 fracture`, `4 metal`,
`5 periostealreaction`, `6 pronatorsign`, `7 softtissue`, `8 text`. The app displays only class 3.
No detector training is performed in this project.

### Inference setup (with the released ONNX model)

The ONNX detector runs through ONNX Runtime alongside the existing TensorFlow classifier; PyTorch
and the YOLOv7 source checkout are not used by this app path.

```powershell
python -m streamlit run app.py
```

The adapter follows the upstream preprocessing: direct resize to 640 × 640, RGB float32 NCHW,
and division by 255. The ONNX graph includes `NonMaxSuppression`; detections are scaled back to
the uploaded image dimensions. Only class ID 3 (`fracture`) is displayed.

The Detect Fracture tab displays actual class-3 fracture boxes and confidence. If none pass
`YOLO_CONFIDENCE_THRESHOLD` (default 0.25), it reports
“No fracture region detected.” and draws no box. If weights are unavailable, the app provides
the documented release link while classification, Model Analytics, metrics, and medical
disclaimer remain available. Grad-CAM is not used as a substitute for a detector box.

---

## 2. Dataset

**Recommended dataset:** *Bone Fracture Detection Using X-rays* (Kaggle), by
**vuppalaadithyasairam**:
`https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-x-rays`

It is a binary-labeled dataset of X-ray images split into `fractured` and `not fractured`
classes, already organized into `train/`, `val/`, and `test/` folders — which matches the
structure this project expects.

Any other binary bone-fracture X-ray dataset works too (e.g. MURA-derived binary subsets),
as long as it follows the folder-per-class layout below. `data_loader.py` reads whatever class
folder names actually exist (it does not hardcode `"fractured"`/`"not_fractured"`), so small
naming differences like `"Fractured"` vs `"fractured"` or `"not fractured"` vs `"not_fractured"`
are handled automatically — Keras's `class_indices` mapping is learned at training time and
saved to `models/class_indices.json`, and every downstream script (evaluate, predict, the app)
reads that file instead of assuming an index means a fixed thing.

### 2.1 How to download it

Option A — via the Kaggle website:
1. Create a free Kaggle account if you don't have one.
2. Open the dataset link above and click **Download**.
3. Unzip it.

Option B — via the Kaggle API:
```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json (from Kaggle Account settings)
kaggle datasets download -d vuppalaadithyasairam/bone-fracture-detection-using-x-rays
unzip bone-fracture-detection-using-x-rays.zip -d raw_dataset
```

### 2.2 Required folder structure

After downloading, arrange (or copy/rename) the images into this exact layout under `data/`:

```
data/
├── train/
│   ├── fractured/        <- training images showing a fracture
│   └── not_fractured/    <- training images with no fracture
├── val/
│   ├── fractured/
│   └── not_fractured/
└── test/
    ├── fractured/
    └── not_fractured/
```

If the dataset you downloaded already ships with `train/`, `val/`, `test/` folders (as the
recommended Kaggle dataset does), simply move/copy those three folders into `data/`. If class
folder names differ slightly (e.g. `"not fractured"` with a space), you do **not** need to
rename anything — the code adapts automatically, as explained above.

If your dataset has no separate `val/` split, create one by moving ~15% of images out of
`train/` into a new `val/` folder (keeping the same class subfolders), so `train.py` has data to
validate against during training.

### 2.3 How the code loads it

`src/data_loader.py`:
- `validate_dataset()` checks that `data/train` and `data/val` exist, contain at least two class
  subfolders, and contain image files — and raises a clear error message telling you exactly
  what's missing if not.
- `get_train_val_generators()` builds Keras `ImageDataGenerator`s: augmentation + rescaling for
  `train/`, rescaling only for `val/`.
- `get_test_generator()` builds the same kind of generator (no augmentation) for `test/`.
- `save_class_indices()` / `load_class_indices()` persist and reload the label mapping learned
  from your dataset's actual folder names.

---

## 3. Model

- **Architecture:** MobileNetV2 (ImageNet-pretrained) as a frozen feature extractor, topped with
  `GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.2) →
  Dense(1, sigmoid)`. Images are loaded into `[0, 1]` by the data pipeline and rescaled inside
  the model to `[-1, 1]`, the input range expected by MobileNetV2's ImageNet weights.
- **Why transfer learning over a CNN trained from scratch:** bone-fracture datasets are small
  relative to typical CNN training needs; a from-scratch CNN tends to overfit. MobileNetV2's
  ImageNet-pretrained features (edges, textures, gradients) transfer well to X-ray images and
  reach higher validation accuracy, faster, on limited data. It's also small enough (~14MB) to
  serve responsively inside the Streamlit app. A custom CNN baseline (`build_custom_cnn` in
  `src/model.py`) is included for anyone who wants to compare the two approaches directly.
- **Training strategy (two phases, both run automatically by `train.py`):**
  1. Freeze the MobileNetV2 base, train only the new head (`INITIAL_EPOCHS`, default 15).
  2. Unfreeze the top layers of the base (from layer index `FINE_TUNE_AT_LAYER` onward) and
     fine-tune at a much lower learning rate (`FINE_TUNE_EPOCHS`, default 10), so pretrained
     features adapt to X-ray textures without being destroyed.
- **Callbacks:** `ModelCheckpoint` (saves the best-val-accuracy model), `EarlyStopping`,
  `ReduceLROnPlateau`. The checkpoint tracks validation accuracy across both phases, and the
  final inference model is exported from that best checkpoint.

All reported metrics (accuracy, precision, recall, F1, confusion matrix) are computed by
`evaluate.py` by running the actual saved model over the actual held-out test set — nothing in
this project hardcodes or fabricates a result.

### Binary class-index convention

`flow_from_directory(class_mode="binary")` assigns indices according to the sorted class
folder names. In this dataset the saved mapping is `fractured: 0`, `not fractured: 1`.
The model has one sigmoid output, which represents the probability of index 1; therefore
`sigmoid >= 0.5` means **not fractured** for this mapping. Prediction maps that index through
`class_indices.json`, and confidence is the probability of the predicted class. The returned
`probability_fracture` is separately derived from the fracture class's actual index.

Evaluation verifies that the test generator uses the saved training mapping, fixes report and
confusion-matrix order to indices 0 then 1, and explicitly defines index 1 (`not fractured`)
as the positive class for aggregate precision, recall, F1, and ROC-AUC. ROC-AUC uses the raw
sigmoid probability of index 1 and the corresponding binary ground truth. Per-class report
rows remain in the order `fractured`, then `not fractured`.

---

## 4. Setup & Running — exact commands

### 4.1 Create the project and virtual environment
```bash
# If you're starting from this delivered project folder, just cd into it:
cd bone-fracture-detection

# Create and activate a virtual environment
python -m venv venv

# macOS / Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### 4.2 Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.3 Prepare the dataset
Follow Section 2 above, then verify it's recognized correctly:
```bash
python src/data_loader.py
```
This prints an image count per class per split. Fix any errors it reports before continuing.

### 4.4 Train the model
```bash
python src/train.py
```
Optional flags:
```bash
python src/train.py --epochs 20 --fine-tune-epochs 10 --batch-size 32
python src/train.py --no-fine-tune          # phase 1 only, skip fine-tuning
```
This produces:
- `models/bone_fracture_model.keras` (final model)
- `models/bone_fracture_model_best.keras` (best checkpoint by val_accuracy)
- `models/class_indices.json`
- `outputs/reports/training_history.json`
- `outputs/plots/accuracy_curve.png`, `outputs/plots/loss_curve.png`

### 4.5 Evaluate the model
```bash
python src/evaluate.py
```
This produces (from real test-set inference):
- `outputs/reports/test_metrics.json` (accuracy, precision, recall, F1, ROC-AUC)
- `outputs/reports/classification_report.txt`
- `outputs/plots/confusion_matrix.png`

### 4.6 Predict on a single image (optional, CLI)
```bash
python src/predict.py --image path/to/some_xray.jpg
```

### 4.7 Run the Streamlit application
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (typically `http://localhost:8501`).

---

## 5. Using the App

1. Go to the **🩻 Detect Fracture** tab.
2. Upload an X-ray image (JPG/PNG).
3. Preview it, then click **Analyze X-ray**.
4. View the result — **Fracture Detected** / **No Fracture Detected** — with a confidence score.
5. Check the **📊 Model Analytics** tab for dataset counts, accuracy/precision/recall/F1,
   the confusion matrix, and training curves — all read from the files generated in Sections
   4.4–4.5, not hardcoded.
6. See **ℹ️ About & Disclaimer** for the full healthcare disclaimer.

### 5.1 Classifier and fracture-localization panel

The current dataset provides image-level class folders (`fractured` and
`not fractured`) but no fracture bounding boxes, YOLO labels, VOC XML, COCO
annotations, or segmentation masks. Therefore, this project cannot train or
claim a true fracture-localization detector from the supplied labels.

The Detect Fracture tab's **Candidate fracture regions** panel is wired to the
separate YOLOv7 detector. It displays only class ID 3 boxes returned by trained
weights; it reports when weights are unavailable and never substitutes a
Grad-CAM heatmap for a detector box. Classification remains available
independently. The original Grad-CAM implementation remains in
`src/localization.py` as an optional classifier explanation, not a detector.

In the interface, MobileNetV2 produces the image-level class score and YOLOv7
produces independent candidate fracture boxes. The outputs are shown together
for review but their scores are not combined; no detected box does not negate a
classifier result. The classifier's displayed score is not calibrated.
If the classifier and detector disagree, the app explicitly flags the result as
unresolved and advises qualified medical review. Results persist while switching
tabs, and detected-box overlays can be downloaded for review.

The detector threshold is `YOLO_CONFIDENCE_THRESHOLD` in `src/config.py` (default 0.25).
The Grad-CAM implementation remains in `src/localization.py` but is not used by the main UI.

The app uses ONNX Runtime with `models/fracture_detector/yolov7-p6-bonefracture.onnx`
and displays actual class-3 boxes only. If the ONNX file is unavailable,
classification, Model Analytics, and the disclaimer remain available.

#### True localization dataset and training requirements

True localization needs a separate dataset with image-level bounding boxes
or masks for fracture regions, plus a detector/segmentation model and
localization-specific evaluation. Suitable formats include YOLO label files,
Pascal VOC XML, COCO JSON, or masks from which boxes can be safely derived.
Keep such data separate from the current classifier folders. This project
does not currently include a localization dataset, detector architecture,
localization training script, or localization evaluation results. Do not
claim trained localization until those components are added and evaluated.

To run the application on Windows PowerShell:

```powershell
cd "C:\Users\india\OneDrive\Desktop\Bone-Fracture-Detection-main"
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Classification determines whether the model predicts fracture or no fracture.
Localization identifies a region only when trained with location annotations;
the app's detector panel uses true YOLO predictions when weights exist.

---

## 6. Healthcare Disclaimer

This application is provided for **educational, research, and screening-support purposes
only**. It is **not a certified medical device** and has not been evaluated or approved by any
regulatory health authority. Model predictions may be incorrect (false positives and false
negatives are possible). **This tool does not replace a radiologist, orthopedic specialist, or
physician**, and must never be the sole basis for a medical decision. If you or someone else may
have a bone injury, seek evaluation from a qualified healthcare professional or emergency
service.
