"""
data_loader.py
---------------
Handles everything related to reading the X-ray dataset off disk:

  * validates that the expected folder structure exists
  * builds Keras ImageDataGenerators for train / validation / test splits
  * applies resizing + normalization (rescale=1/255) to every split
  * applies data augmentation ONLY to the training split
  * is tolerant of small naming differences in the dataset
    (e.g. "not fractured" vs "not_fractured" vs "Not Fractured"),
    because it reads whatever class subfolders actually exist instead
    of assuming fixed names.

Expected folder structure (this is what the recommended Kaggle dataset
already ships with):

    data/
        train/
            fractured/
            not fractured/
        val/
            fractured/
            not fractured/
        test/
            fractured/
            not fractured/

Each class folder must contain image files (.jpg / .jpeg / .png).
"""

import os
import sys
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _list_class_folders(split_dir):
    """Return the class subfolder names found under a split directory."""
    if not os.path.isdir(split_dir):
        return []
    return sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith(".")
    )


def _count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(
        1 for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def validate_dataset():
    """
    Checks that data/train, data/val (and optionally data/test) exist and
    contain at least two class subfolders with images in them.

    Raises a clear, actionable error instead of letting Keras fail with a
    cryptic "found 0 images" message.

    Returns a small report dict that train.py / evaluate.py can print.
    """
    report = {}
    required_splits = ["train", "val"]
    optional_splits = ["test"]

    for split in required_splits + optional_splits:
        split_dir = os.path.join(config.DATA_DIR, split)
        classes = _list_class_folders(split_dir)
        counts = {c: _count_images(os.path.join(split_dir, c)) for c in classes}
        report[split] = {"path": split_dir, "classes": counts}

        if split in required_splits:
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(
                    f"Expected dataset folder not found: {split_dir}\n"
                    f"Please download the dataset and place it as described in README.md "
                    f"(data/train/<class>/, data/val/<class>/)."
                )
            if len(classes) < 2:
                raise ValueError(
                    f"Expected at least 2 class subfolders inside {split_dir}, "
                    f"found: {classes}. See README.md for the required dataset layout."
                )
            if sum(counts.values()) == 0:
                raise ValueError(
                    f"No images found inside {split_dir}. Supported extensions: "
                    f"{IMAGE_EXTENSIONS}."
                )
    return report


def print_dataset_report(report):
    print("\n--- Dataset summary ---")
    for split, info in report.items():
        if not info["classes"]:
            print(f"  {split:5s}: (not found / empty) -> {info['path']}")
            continue
        total = sum(info["classes"].values())
        print(f"  {split:5s}: {total} images  ->  {info['classes']}")
    print("-----------------------\n")


def get_train_val_generators(batch_size=None, img_size=None):
    """
    Builds the training and validation generators.

    Training generator: resizing + normalization + augmentation
    (rotation, shifts, zoom, horizontal flip, brightness) to help the
    model generalize from a limited number of medical images.

    Validation generator: resizing + normalization ONLY. We never
    augment validation/test data — augmentation is purely a training-time
    regularization technique, and evaluating on augmented images would
    give a misleading picture of real-world performance.
    """
    batch_size = batch_size or config.BATCH_SIZE
    img_size = img_size or config.IMG_SIZE

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.85, 1.15),
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        color_mode="rgb",
        shuffle=True,
        seed=config.SEED,
    )

    val_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False,
    )

    return train_generator, val_generator


def get_test_generator(batch_size=None, img_size=None):
    """Builds the test generator (resizing + normalization only, no shuffle)."""
    batch_size = batch_size or config.BATCH_SIZE
    img_size = img_size or config.IMG_SIZE

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False,
    )
    return test_generator


def save_class_indices(class_indices):
    """
    Persists the {class_name: index} mapping learned from the training
    folder structure, e.g. {"fractured": 0, "not fractured": 1}.

    This is the mechanism that makes the rest of the pipeline robust to
    dataset naming differences: predict.py and app.py load this file
    instead of assuming which index means what.
    """
    with open(config.CLASS_INDICES_PATH, "w") as f:
        json.dump(class_indices, f, indent=2)


def load_class_indices():
    if not os.path.exists(config.CLASS_INDICES_PATH):
        raise FileNotFoundError(
            f"{config.CLASS_INDICES_PATH} not found. Train the model first "
            f"(python src/train.py) — training saves this mapping automatically."
        )
    with open(config.CLASS_INDICES_PATH) as f:
        return json.load(f)


def resolve_label_name(class_name):
    """
    Maps a raw folder/class name (e.g. 'not fractured', 'Fractured',
    'not_fractured') to a normalized human-readable label used across
    the app: 'Fracture Detected' or 'No Fracture Detected'.
    """
    name = class_name.lower()
    if any(neg in name for neg in config.NEGATIVE_KEYWORDS):
        return "No Fracture Detected"
    if any(pos in name for pos in config.POSITIVE_KEYWORDS):
        return "Fracture Detected"
    # Fallback: treat index 0 as fractured, shouldn't normally hit this
    return class_name


if __name__ == "__main__":
    report = validate_dataset()
    print_dataset_report(report)
