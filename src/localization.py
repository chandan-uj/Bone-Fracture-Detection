"""Approximate Grad-CAM visualization for the existing classifier.

This module does not detect fracture boundaries or produce bounding boxes.
It visualizes image regions that influence a selected classifier output.
True localization requires separately annotated training data and a detector.
"""

import numpy as np
import tensorflow as tf
from PIL import Image

try:  # Support both `import localization` from app.py and `src.localization`.
    from . import config
    from .predict import preprocess_image
except ImportError:  # pragma: no cover - exercised by the Streamlit entry point
    import config
    from predict import preprocess_image


def _last_spatial_layer(model):
    """Find the last top-level layer with a spatial (rank-4) output."""
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer
        except (AttributeError, TypeError):
            continue
    return None


def make_attention_overlay(model, image, predicted_index):
    """Return a heatmap overlay at original image dimensions, or None.

    The class index is the class selected by the existing prediction flow.
    For a sigmoid model, index 1 uses p and index 0 uses 1-p.
    """
    if not config.LOCALIZATION_ENABLED:
        return None

    target_layer = _last_spatial_layer(model)
    if target_layer is None:
        return None

    try:
        input_array = preprocess_image(image)
        target_position = model.layers.index(target_layer)
        with tf.GradientTape() as tape:
            # Calling the nested backbone and then the existing head layers
            # avoids Keras 3's disconnected nested-output graph issue while
            # reusing the saved model's exact weights and inference behavior.
            feature_maps = target_layer(input_array, training=False)
            tape.watch(feature_maps)
            prediction = feature_maps
            for layer in model.layers[target_position + 1:]:
                if isinstance(layer, tf.keras.layers.Dropout):
                    prediction = layer(prediction, training=False)
                else:
                    prediction = layer(prediction)

            probability_index_1 = prediction[:, 0]
            score = (probability_index_1 if predicted_index == 1
                     else 1.0 - probability_index_1)

        gradients = tape.gradient(score, feature_maps)
        if gradients is None:
            return None
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)
        if float(max_value.numpy()) <= 1e-8:
            return None
        heatmap = heatmap / max_value
        heatmap = tf.image.resize(
            heatmap[..., tf.newaxis], (image.height, image.width),
            method="bilinear",
        )[..., 0]
        heatmap = np.asarray(heatmap.numpy(), dtype=np.float32)

        import matplotlib.cm as cm
        palette = getattr(cm, config.LOCALIZATION_COLORMAP.lower())
        rgb = (palette(heatmap)[..., :3] * 255).astype(np.uint8)
        colored = Image.fromarray(rgb, mode="RGB")
        original = image.convert("RGB")
        return Image.blend(original, colored, config.LOCALIZATION_OVERLAY_ALPHA)
    except (ValueError, TypeError, IndexError, AttributeError, RuntimeError):
        # Explainability must never prevent the original classifier result.
        return None
