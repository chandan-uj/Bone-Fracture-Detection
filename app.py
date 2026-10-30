"""
app.py
------
Streamlit application for the Bone Fracture Detection project.

Run from the project root (after the model has been trained):
    streamlit run app.py

Tabs:
  1. Detect Fracture  - upload an X-ray, preview it, click Analyze, see result.
  2. Model Analytics   - real accuracy/precision/recall/F1, confusion matrix,
                         training curves, and dataset counts (nothing fake —
                         all read from files produced by train.py / evaluate.py).
  3. About & Disclaimer - project info and the healthcare disclaimer.
"""

import os
import sys
import json
import io
import hashlib

import streamlit as st
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import config
from predict import load_trained_model, predict_image
from data_loader import load_class_indices, validate_dataset
from detect_fracture import load_detector, predict_with_detector


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    :root {
        --ink: #e8f1f8;
        --muted-ink: #9db2c4;
        --line: rgba(157, 190, 212, 0.16);
        --surface: #102331;
        --canvas: #08141e;
        --blue: #65d6d0;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 14% 0%, rgba(22, 108, 124, 0.22), transparent 38%),
            radial-gradient(ellipse at 92% 8%, rgba(32, 79, 130, 0.2), transparent 34%),
            var(--canvas);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(8, 20, 30, 0.55); }
    [data-testid="stMainBlockContainer"] { max-width: 1440px; padding-top: 2rem; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stWidgetLabel"] p { color: var(--ink) !important; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp [data-testid="stImage"] figcaption {
        color: var(--muted-ink) !important;
        font-size: 0.88rem !important;
        line-height: 1.5 !important;
    }
    .stApp label, .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricLabel"] p { color: var(--muted-ink) !important; }
    .stApp [data-testid="stMetricValue"] { color: #f2fbff !important; font-weight: 750; }
    .stApp [data-testid="stTab"] {
        color: var(--muted-ink);
        font-weight: 650;
        padding-bottom: 0.65rem;
    }
    .stApp [data-testid="stTab"][aria-selected="true"] { color: var(--blue); }
    .stApp [data-testid="stTabHighlight"] { background: var(--blue); }
    .stApp [data-testid="stExpander"],
    .stApp [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(16, 35, 49, 0.74);
        border: 1px solid var(--line);
        border-radius: 16px;
    }
    .stApp [data-testid="stAlert"] { color: var(--ink); }
    .stApp [data-testid="baseButton-primary"],
    .stApp [data-testid="stBaseButton-primary"] {
        background: linear-gradient(115deg, #27a9a5, #267ab4);
        border: 1px solid rgba(101, 214, 208, 0.65);
        border-radius: 10px;
        color: #f7ffff;
        font-weight: 700;
        min-height: 3rem;
        box-shadow: 0 8px 24px rgba(24, 141, 148, 0.2);
    }
    .stApp [data-testid="baseButton-primary"]:hover,
    .stApp [data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(115deg, #31bbb4, #318cc7);
        border-color: #8ce5df;
    }
    .stApp [data-testid="stFileUploader"] section {
        background: rgba(13, 31, 44, 0.7);
        border: 1px dashed rgba(101, 214, 208, 0.42);
        border-radius: 14px;
    }
    .stApp [data-testid="stProgressBar"] > div > div { background: #65d6d0; }

    .app-header {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(circle at 88% 18%, rgba(101, 214, 208, 0.22), transparent 23%),
            linear-gradient(120deg, #102b40 0%, #104d61 52%, #146277 100%);
        padding: 2.2rem 2.5rem;
        border: 1px solid rgba(125, 218, 218, 0.24);
        border-radius: 22px;
        margin-bottom: 1.35rem;
        box-shadow: 0 20px 55px rgba(0, 0, 0, 0.2);
    }
    .hero-kicker {
        color: #85e3dc;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .stApp [data-testid="stMarkdownContainer"] .app-header h1 {
        margin: 0.55rem 0 0;
        color: #f4fbff !important;
        font-size: clamp(1.8rem, 3.2vw, 2.75rem);
        line-height: 1.13;
    }
    .stApp [data-testid="stMarkdownContainer"] .app-header p {
        max-width: 760px;
        margin: 0.75rem 0 0;
        color: #c6dce8 !important;
        font-size: 1rem;
        line-height: 1.65;
    }
    .hero-badges { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 1.2rem; }
    .hero-badges span {
        border: 1px solid rgba(179, 232, 231, 0.25);
        border-radius: 999px;
        background: rgba(4, 22, 32, 0.24);
        color: #d5f3f1;
        padding: 0.35rem 0.72rem;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.06em;
    }
    .disclaimer-banner {
        background: rgba(73, 53, 22, 0.3);
        border: 1px solid rgba(232, 180, 85, 0.3);
        border-left: 4px solid #e7b34e;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        font-size: 0.9rem;
        line-height: 1.55;
        color: #f3dcae;
        margin-bottom: 1.4rem;
    }
    .result-card {
        padding: 1.45rem 1.4rem;
        border-radius: 17px;
        margin-top: 0.4rem;
        text-align: center;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.16);
    }
    .result-fracture {
        background: linear-gradient(145deg, rgba(111, 36, 50, 0.58), rgba(56, 25, 39, 0.68));
        border: 1px solid rgba(244, 116, 128, 0.58);
        color: #ffe3e4;
    }
    .result-no-fracture {
        background: linear-gradient(145deg, rgba(22, 91, 74, 0.55), rgba(18, 57, 57, 0.68));
        border: 1px solid rgba(87, 212, 166, 0.5);
        color: #dcfff0;
    }
    .result-card h2 { margin: 0 0 0.8rem; color: inherit !important; font-size: 1.55rem; }
    .result-confidence-label {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        opacity: 0.85;
    }
    .result-confidence-value { margin-top: 0.1rem; font-size: 2.1rem; font-weight: 800; line-height: 1.2; }
    .attention-intro {
        background: linear-gradient(120deg, rgba(17, 50, 66, 0.86), rgba(15, 38, 54, 0.8));
        border: 1px solid rgba(101, 214, 208, 0.21);
        border-left: 4px solid #65d6d0;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        margin: 0.4rem 0 0.9rem;
    }
    .attention-intro .eyebrow {
        color: #82e0da;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .attention-intro h3 { color: #eefaff !important; font-size: 1.18rem; margin: 0.2rem 0 0.35rem; }
    .attention-intro p { color: #b5ccd9 !important; margin: 0; line-height: 1.55; }
    .image-panel { background: rgba(16, 35, 49, 0.74); border: 1px solid var(--line); border-radius: 14px; padding: 0.6rem; }
    @media (max-width: 700px) {
        .app-header { padding: 1.55rem 1.25rem; border-radius: 16px; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header + persistent disclaimer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="hero-kicker">Computer vision · screening research</div>
        <h1>Bone fracture analysis</h1>
        <p>Explore an image-level classifier and an independent fracture-localization model in one
        clear workspace. Upload an X-ray to view both outputs side by side.</p>
        <div class="hero-badges">
            <span>MobileNetV2 · CLASSIFICATION</span>
            <span>YOLOv7 · FRACTURE BOXES</span>
            <span>RESEARCH PROTOTYPE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <b>Healthcare Disclaimer:</b> This tool is intended for <b>screening / research support
        only</b> and is <b>not a substitute for professional medical diagnosis</b>. It does not
        replace a radiologist or physician. Always consult a qualified healthcare professional
        for an accurate diagnosis and treatment decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(model_version=None):
    return load_trained_model()


@st.cache_data(show_spinner=False)
def get_class_indices(mapping_version=None):
    return load_class_indices()


@st.cache_resource(show_spinner=False)
def get_detector():
    return load_detector()


def model_is_available():
    return os.path.exists(config.MODEL_PATH) and os.path.exists(config.CLASS_INDICES_PATH)


tab_detect, tab_analytics, tab_about = st.tabs(
    ["🩻 Detect Fracture", "📊 Model Analytics", "ℹ️ About & Disclaimer"]
)


# ---------------------------------------------------------------------------
# Tab 1: Detect Fracture
# ---------------------------------------------------------------------------
with tab_detect:
    if not model_is_available():
        st.error(
            "No trained model found yet. Train the model first:\n\n"
            "```\npython src/train.py\n```\n\n"
            "then restart this app."
        )
    else:
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.subheader("1. Upload an X-ray image")
            st.caption("Image-level classification and region detection run as separate models.")
            model_version = os.path.getmtime(config.MODEL_PATH)
            detector_version = (
                os.path.getmtime(config.DETECTOR_ONNX_MODEL_PATH)
                if os.path.exists(config.DETECTOR_ONNX_MODEL_PATH) else None
            )
            uploaded_file = st.file_uploader(
                "Supported formats: JPG, JPEG, PNG",
                type=["jpg", "jpeg", "png"],
            )

            upload_fingerprint = (
                hashlib.sha256(uploaded_file.getvalue()).hexdigest()
                if uploaded_file is not None else None
            )
            image = None
            image_error = None
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    image.load()
                    st.subheader("2. Preview")
                    st.image(image, caption="Uploaded X-ray", use_column_width=True)
                except (OSError, ValueError, Image.DecompressionBombError):
                    image_error = "This file could not be read as a supported image. Please upload a valid JPG, JPEG, or PNG image."
                    st.error(image_error)

            analyze_clicked = st.button(
                "🔍 Analyze X-ray", type="primary",
                disabled=uploaded_file is None or image is None, use_container_width=True,
            )

        with col_result:
            st.subheader("3. Prediction Result")
            if uploaded_file is None:
                st.info("Upload an X-ray image on the left to get started.")
            elif image_error:
                st.info("No prediction was run because the uploaded file is not a valid image.")
            elif analyze_clicked or (
                st.session_state.get("analysis_result", {}).get("fingerprint") == upload_fingerprint
                and st.session_state["analysis_result"].get("model_version") == model_version
                and st.session_state["analysis_result"].get("detector_version") == detector_version
            ):
                if analyze_clicked:
                    try:
                        with st.spinner("Analyzing X-ray..."):
                            model = get_model(model_version)
                            class_indices = get_class_indices(
                                os.path.getmtime(config.CLASS_INDICES_PATH)
                            )
                            result = predict_image(model, image, class_indices)
                    except Exception as exc:
                        st.error(f"Classification could not be completed: {exc}")
                        result = None
                else:
                    result = st.session_state["analysis_result"]["result"]

                if result is not None:
                    is_fracture = result["label"] == "Fracture Detected"
                    css_class = "result-fracture" if is_fracture else "result-no-fracture"
                    icon = "🚨" if is_fracture else "✅"

                    st.markdown(
                        f"""
                        <div class="result-card {css_class}">
                            <h2>{icon} {result['label']}</h2>
                            <div class="result-confidence-label">Prediction confidence · not calibrated</div>
                            <div class="result-confidence-value">{result['confidence']*100:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.write("")
                    st.progress(result["confidence"], text=f"Predicted-class score: {result['confidence']*100:.1f}%")

                    with st.expander("See detailed probability breakdown"):
                        if result["probability_fracture"] is None:
                            st.warning(
                                "The dataset class names do not identify which output represents "
                                "fracture, so a fracture-specific probability is unavailable."
                            )
                        else:
                            probability_col1, probability_col2 = st.columns(2)
                            probability_col1.metric(
                                "Classifier fracture score",
                                f"{result['probability_fracture']*100:.2f}%",
                            )
                            probability_col2.metric(
                                "Classifier no-fracture score",
                                f"{(1 - result['probability_fracture'])*100:.2f}%",
                            )
                        st.caption(f"Raw predicted class (from dataset labels): `{result['raw_class']}`")

                    st.subheader("Independent fracture localization")
                    st.markdown(
                        """
                        <div class="attention-intro">
                            <div class="eyebrow">YOLOv7 · ONNX DETECTOR</div>
                            <h3>Candidate fracture regions</h3>
                            <p>Boxes are produced by a separate pretrained detector. They are not generated by, or fused with, the image-level classifier.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Showing detector class “fracture” only · score threshold "
                        f"{config.YOLO_CONFIDENCE_THRESHOLD:.2f}. No box does not rule out a fracture."
                    )
                    annotated = None
                    detections = None
                    detector_error = None
                    if not os.path.exists(config.DETECTOR_ONNX_MODEL_PATH):
                        from detect_fracture import YOLOV7_ONNX_RELEASE_URL
                        st.info(
                            "Fracture localization model is unavailable. Classification is still working. "
                            "Download the official YOLOv7 ONNX model and place it at "
                            f"`{config.DETECTOR_ONNX_MODEL_PATH}`."
                        )
                        st.markdown(f"[Open the official YOLOv7 ONNX release]({YOLOV7_ONNX_RELEASE_URL})")
                    elif not analyze_clicked:
                        cached_analysis = st.session_state["analysis_result"]
                        annotated = cached_analysis["annotated"]
                        detections = cached_analysis["detections"]
                        detector_error = cached_analysis["detector_error"]
                    else:
                        try:
                            with st.spinner("Checking for fracture regions..."):
                                detector = get_detector()
                                annotated, detections = predict_with_detector(detector, image)
                        except Exception as exc:
                            detector_error = str(exc)
                    if detections is None:
                        if detector_error:
                            st.warning(f"Fracture localization is unavailable: {detector_error}")
                    elif not detections:
                        st.info(
                            "The detector returned no fracture box above its score threshold. "
                            "This does not override or negate the classifier result."
                        )
                    else:
                        if result["label"] != "Fracture Detected":
                            st.error(
                                "Model outputs disagree: the classifier predicts no fracture, "
                                f"but YOLOv7 localized {len(detections)} fracture region(s). "
                                "Treat this as unresolved and seek qualified medical review."
                            )
                        col_original, col_detection = st.columns(2)
                        with col_original:
                            with st.container(border=True):
                                st.image(image, caption="Original X-ray", use_column_width=True)
                        with col_detection:
                            with st.container(border=True):
                                st.image(annotated, caption="YOLOv7 fracture detections", use_column_width=True)
                        summary_col1, summary_col2 = st.columns(2)
                        summary_col1.metric("Detected fracture regions", len(detections))
                        summary_col2.metric("Highest detector score", f"{max(d['confidence'] for d in detections) * 100:.1f}%")
                        st.markdown("**Detected regions**")
                        for index, detection in enumerate(detections, start=1):
                            st.write(
                                f"Region {index}: {detection['class_name']} — "
                                f"{detection['confidence'] * 100:.1f}% detector score · "
                                f"box ({detection['x1']}, {detection['y1']}, "
                                f"{detection['x2']}, {detection['y2']}) px"
                            )
                        image_buffer = io.BytesIO()
                        annotated.save(image_buffer, format="PNG")
                        st.download_button(
                            "Download annotated X-ray",
                            data=image_buffer.getvalue(),
                            file_name="fracture-localization.png",
                            mime="image/png",
                            use_container_width=True,
                        )

                    st.session_state["analysis_result"] = {
                        "fingerprint": upload_fingerprint,
                        "model_version": model_version,
                        "detector_version": detector_version,
                        "result": result,
                        "annotated": annotated,
                        "detections": detections,
                        "detector_error": detector_error,
                    }
                    st.warning(
                        "Remember: this is an AI screening aid, not a diagnosis. "
                        "Please confirm any finding with a qualified radiologist or physician."
                    )
            else:
                st.info("Click **Analyze X-ray** to run the model on the uploaded image.")


# ---------------------------------------------------------------------------
# Tab 2: Model Analytics
# ---------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Dataset Summary")
    try:
        report = validate_dataset()
        cols = st.columns(len(report))
        for col, (split, info) in zip(cols, report.items()):
            total = sum(info["classes"].values()) if info["classes"] else 0
            with col:
                st.metric(label=f"{split.capitalize()} images", value=total)
                if info["classes"]:
                    st.caption(" · ".join(f"{k}: {v}" for k, v in info["classes"].items()))
    except Exception as e:
        st.warning(f"Dataset not found or incomplete: {e}")

    st.divider()
    st.subheader("Test Set Performance")

    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as f:
            metrics = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        m3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        m4.metric("F1-score", f"{metrics['f1_score']*100:.2f}%")
        st.caption(
            f"Evaluated on {metrics['num_test_samples']} held-out test images "
            f"(decision threshold = {metrics['decision_threshold']})."
            + (
                f" ROC-AUC: {metrics['roc_auc']:.4f}"
                if metrics.get("roc_auc") is not None else ""
            )
        )
        if metrics.get("positive_class"):
            st.caption(
                f"Aggregate precision, recall, F1, and ROC-AUC positive class: "
                f"{metrics['positive_class']}. Per-class rows are shown in the report."
            )

        col_cm, col_curves = st.columns(2)
        with col_cm:
            if os.path.exists(config.CONFUSION_MATRIX_PLOT_PATH):
                st.image(config.CONFUSION_MATRIX_PLOT_PATH, caption="Confusion Matrix (Test Set)",
                         use_column_width=True)
        with col_curves:
            if os.path.exists(config.ACCURACY_PLOT_PATH):
                st.image(config.ACCURACY_PLOT_PATH, caption="Training vs Validation Accuracy",
                         use_column_width=True)
            if os.path.exists(config.LOSS_PLOT_PATH):
                st.image(config.LOSS_PLOT_PATH, caption="Training vs Validation Loss",
                         use_column_width=True)

        if os.path.exists(config.CLASSIFICATION_REPORT_PATH):
            with st.expander("Full scikit-learn classification report"):
                with open(config.CLASSIFICATION_REPORT_PATH) as f:
                    st.code(f.read())
    else:
        st.info(
            "No evaluation results yet. After training, run:\n\n"
            "```\npython src/evaluate.py\n```\n\n"
            "to generate real accuracy/precision/recall/F1 and the confusion matrix from the "
            "trained model on the held-out test set."
        )


# ---------------------------------------------------------------------------
# Tab 3: About & Disclaimer
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        **Bone Fracture Detection Using X-ray Images** is a deep-learning screening tool that
        classifies an uploaded X-ray as either **Fracture Detected** or **No Fracture Detected**,
        along with a confidence score.

        **Model:** MobileNetV2 (pretrained on ImageNet) with a custom classification head,
        fine-tuned in two phases on a labeled bone X-ray dataset. Transfer learning was chosen
        over a from-scratch CNN because it performs better and trains faster on the
        comparatively small datasets typical of medical imaging.

        **Technology stack:** Python, TensorFlow/Keras, OpenCV/PIL, NumPy, Pandas, scikit-learn,
        Matplotlib, Streamlit.

        **Pipeline:** dataset validation → preprocessing (resize + normalize) → augmentation
        (train only) → MobileNetV2 transfer learning → two-phase training (frozen, then
        fine-tuned) → evaluation (accuracy, precision, recall, F1, confusion matrix) → Streamlit
        deployment.
        """
    )

    st.divider()
    st.subheader("⚠️ Healthcare Disclaimer")
    st.markdown(
        """
        - This application is provided **for educational, research, and screening-support
          purposes only**.
        - It is **not a certified medical device** and has **not been evaluated or approved by
          any regulatory health authority**.
        - Predictions are generated by a machine learning model and **may be incorrect**. The
          model can produce false positives and false negatives.
        - This tool **does not replace a radiologist, orthopedic specialist, or physician**, and
          must never be used as the sole basis for a medical decision.
        - If you or someone else may have a bone injury, seek evaluation from a qualified
          healthcare professional or emergency service.
        """
    )
