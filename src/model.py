"""
model.py
--------
Defines the deep learning architecture used for fracture classification.

Why transfer learning (MobileNetV2) instead of a CNN trained from scratch:

  * Medical imaging datasets are typically small (a few thousand images per
    class at most). A CNN trained from scratch on that little data tends to
    overfit badly.
  * MobileNetV2 is pretrained on ImageNet (1.4M images) and has already
    learned general-purpose visual features (edges, textures, shapes,
    gradients) that transfer well to X-ray images, which are grayscale-like
    and edge/texture heavy.
  * MobileNetV2 is lightweight (~14MB, ~3.4M params) compared to
    ResNet50/EfficientNetB0, so it trains fast even on a CPU and is cheap
    to serve in the Streamlit app — a good fit for a screening tool that
    should run responsively.
  * In practice, on small binary medical-imaging datasets, transfer
    learning consistently outperforms a from-scratch CNN, which is why it
    is used as the default model here. A simple CNN baseline is included
    below for comparison / experimentation.

Two-phase training strategy (implemented in train.py):
  1. Freeze the MobileNetV2 base, train only the new classification head.
  2. Unfreeze the top layers of the base and fine-tune at a very low
     learning rate, so the pretrained features adapt slightly to X-ray
     textures without being destroyed by large gradient updates.
"""

import sys
import os

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def build_transfer_model(img_size=None, channels=None):
    """
    Builds the MobileNetV2-based transfer learning model.
    Returns (model, base_model) — base_model is returned separately so
    train.py can freeze/unfreeze it for the two-phase training schedule.
    """
    img_size = img_size or config.IMG_SIZE
    channels = channels or config.IMG_CHANNELS
    input_shape = (img_size[0], img_size[1], channels)

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # phase 1: frozen

    inputs = layers.Input(shape=input_shape)
    # ImageDataGenerator and predict.py both scale images to [0, 1].
    # MobileNetV2's ImageNet weights expect preprocess_input's [-1, 1] range.
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenetv2_preprocessing")(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="bone_fracture_mobilenetv2")
    return model, base_model


def build_custom_cnn(img_size=None, channels=None):
    """
    A simple CNN trained from scratch, provided as a baseline for
    comparison against the transfer-learning model. Not used by default
    (see module docstring for why transfer learning was chosen), but kept
    here so the choice can be verified experimentally rather than assumed.
    """
    img_size = img_size or config.IMG_SIZE
    channels = channels or config.IMG_CHANNELS
    input_shape = (img_size[0], img_size[1], channels)

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ], name="bone_fracture_custom_cnn")

    return model


def unfreeze_top_layers(base_model, fine_tune_at=None):
    """
    Unfreezes layers of the base model from `fine_tune_at` index onward,
    keeping earlier (more generic) layers frozen. Used for phase-2
    fine-tuning.
    """
    fine_tune_at = fine_tune_at if fine_tune_at is not None else config.FINE_TUNE_AT_LAYER
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    return base_model
