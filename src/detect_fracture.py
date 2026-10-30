"""ONNX Runtime inference for the project's official YOLOv7 fracture model.

Preprocessing and output interpretation follow the supplied project's
``inference_onnx.py``: direct 640x640 resize, RGB float32 NCHW input scaled to
[0, 1], and model output rows whose first six fields are xyxy, confidence,
and class ID. The exported model includes its detection decoding/NMS graph.
"""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

import config

YOLOV7_ONNX_RELEASE_URL = (
    "https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection/releases/download/"
    "trained-models/yolov7-p6-bonefracture.onnx"
)

CLASS_NAMES = {
    0: "boneanomaly",
    1: "bonelesion",
    2: "foreignbody",
    3: "fracture",
    4: "metal",
    5: "periostealreaction",
    6: "pronatorsign",
    7: "softtissue",
    8: "text",
}


def load_detector():
    """Create a CPU ONNX Runtime session for the separate detector artifact."""
    if not os.path.isfile(config.DETECTOR_ONNX_MODEL_PATH):
        raise FileNotFoundError(
            "Fracture localization model is unavailable. Download the official model from "
            f"{YOLOV7_ONNX_RELEASE_URL} and save it to {config.DETECTOR_ONNX_MODEL_PATH}."
        )
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is not installed in this environment. Install onnx and "
            "onnxruntime in the detector-enabled environment."
        ) from exc

    session = ort.InferenceSession(
        config.DETECTOR_ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError("Unexpected ONNX detector signature; expected one input and one output.")
    return session


def _prepare_input(image: Image.Image, input_size: int):
    """Match the upstream inference_onnx.py RGB resize/scale/NCHW preprocessing."""
    import cv2
    import numpy as np

    rgb = image.convert("RGB")
    rgb_array = np.asarray(rgb)
    resized = cv2.resize(
        rgb_array,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    batch = np.asarray(resized, dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.expand_dims(batch, axis=0).transpose(0, 3, 1, 2))


def predict_with_detector(session: Any, image: Image.Image):
    """Run ONNX inference, retain class ID 3 only, and draw returned boxes."""
    import cv2
    import numpy as np

    original = image.convert("RGB")
    input_size = int(config.DETECTOR_IMAGE_SIZE)
    input_tensor = _prepare_input(original, input_size)
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    outputs = session.run([output_meta.name], {input_meta.name: input_tensor})
    if not outputs:
        raise RuntimeError("The ONNX detector returned no outputs.")

    rows = np.asarray(outputs[0])
    # The supplied ONNX adapter reads output[0][:, :6]. Its export returns
    # postprocessed detections (xyxy, score, class ID, followed by class scores).
    if rows.ndim == 3 and rows.shape[0] == 1:
        rows = rows[0]
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise RuntimeError(f"Unexpected YOLOv7 ONNX output shape: {rows.shape}")

    image_width, image_height = original.size
    x_scale = image_width / float(input_size)
    y_scale = image_height / float(input_size)
    detections = []
    annotated_bgr = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)

    for row in rows[:, :6]:
        x1_model, y1_model, x2_model, y2_model, score, class_value = map(float, row)
        if not np.all(np.isfinite([x1_model, y1_model, x2_model, y2_model, score, class_value])):
            continue
        class_id = int(class_value)
        if class_id != config.DETECTOR_CLASS_ID_FRACTURE:
            continue
        if not np.isfinite(score) or score <= config.YOLO_CONFIDENCE_THRESHOLD:
            continue

        # The original project scales the model's xyxy boxes by W/640 and H/640.
        x1 = max(0, min(image_width - 1, int(round(x1_model * x_scale))))
        y1 = max(0, min(image_height - 1, int(round(y1_model * y_scale))))
        x2 = max(0, min(image_width - 1, int(round(x2_model * x_scale))))
        y2 = max(0, min(image_height - 1, int(round(y2_model * y_scale))))
        if x2 <= x1 or y2 <= y1:
            continue

        detection = {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": float(score),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "bbox": (x1, y1, x2, y2),
        }
        detections.append(detection)

        label = f"fracture {score * 100:.1f}%"
        color = (35, 57, 229)  # visible red in OpenCV's BGR order
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, config.DETECTOR_BOX_THICKNESS)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_top = max(text_height + baseline + 4, y1)
        cv2.rectangle(
            annotated_bgr,
            (x1, label_top - text_height - baseline - 4),
            (min(image_width - 1, x1 + text_width + 8), label_top + 2),
            color,
            -1,
        )
        cv2.putText(
            annotated_bgr,
            label,
            (x1 + 4, label_top - baseline - 1),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), detections
