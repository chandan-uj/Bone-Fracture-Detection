"""
train.py
--------
Trains the bone fracture classifier end to end:

  1. Validates the dataset folder structure.
  2. Builds train/validation generators (with augmentation on train only).
  3. Builds the MobileNetV2 transfer-learning model.
  4. Phase 1: trains the classification head with the base frozen.
  5. Phase 2: unfreezes the top of the base and fine-tunes at a low LR.
  6. Saves the final model, the best checkpoint, the class-index mapping,
     the training history, and accuracy/loss plots.

Run from the project root:
    python src/train.py
    python src/train.py --epochs 20 --fine-tune-epochs 10 --batch-size 32
"""

import os
import sys
import json
import argparse

# Windows terminals can default to a non-UTF-8 code page (for example cp1252),
# which breaks Keras/TF progress output when it emits Unicode box-drawing chars.
# Force UTF-8 for the Python process so the training pipeline runs reliably on
# Windows without crashing during fit().
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")  # no display needed, we only save figures
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import (
    validate_dataset,
    print_dataset_report,
    get_train_val_generators,
    save_class_indices,
)
from model import build_transfer_model, unfreeze_top_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Train the bone fracture detection model.")
    parser.add_argument("--epochs", type=int, default=config.INITIAL_EPOCHS,
                         help="Epochs for phase 1 (frozen base).")
    parser.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS,
                         help="Epochs for phase 2 (fine-tuning).")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--no-fine-tune", action="store_true",
                         help="Skip phase 2 fine-tuning and only train the head.")
    return parser.parse_args()


def plot_and_save_curves(history_dict, fine_tune_start_epoch=None):
    """Saves separate accuracy and loss curve PNGs, as required by the spec."""
    acc = history_dict.get("accuracy", [])
    val_acc = history_dict.get("val_accuracy", [])
    loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, label="Training Accuracy", marker="o")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="o")
    if fine_tune_start_epoch:
        plt.axvline(fine_tune_start_epoch, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.ACCURACY_PLOT_PATH, dpi=150)
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, label="Training Loss", marker="o")
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="o")
    if fine_tune_start_epoch:
        plt.axvline(fine_tune_start_epoch, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.LOSS_PLOT_PATH, dpi=150)
    plt.close()

    print(f"Saved accuracy curve -> {config.ACCURACY_PLOT_PATH}")
    print(f"Saved loss curve     -> {config.LOSS_PLOT_PATH}")


def merge_histories(h1, h2):
    """Concatenates phase-1 and phase-2 Keras History.history dicts."""
    if h2 is None:
        return h1
    merged = {}
    for key in h1:
        merged[key] = h1[key] + h2.get(key, [])
    return merged


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(config.SEED)

    print("Validating dataset...")
    report = validate_dataset()
    print_dataset_report(report)

    train_gen, val_gen = get_train_val_generators(batch_size=args.batch_size)
    save_class_indices(train_gen.class_indices)
    print(f"Class indices learned from folder structure: {train_gen.class_indices}")

    model, base_model = build_transfer_model()
    model.compile(
        optimizer=Adam(learning_rate=config.INITIAL_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    model.summary()

    best_checkpoint = ModelCheckpoint(
        config.BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    callbacks_phase1 = [
        best_checkpoint,
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    print("\n===== Phase 1: training classification head (base frozen) =====")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks_phase1,
    )

    fine_tune_start_epoch = None
    history2 = None

    if not args.no_fine_tune:
        print("\n===== Phase 2: fine-tuning top layers of MobileNetV2 =====")
        unfreeze_top_layers(base_model, fine_tune_at=config.FINE_TUNE_AT_LAYER)
        model.compile(
            optimizer=Adam(learning_rate=config.FINE_TUNE_LR),
            loss="binary_crossentropy",
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
        )
        model.summary()

        callbacks_phase2 = [
            best_checkpoint,
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8, verbose=1),
        ]

        fine_tune_start_epoch = len(history1.history["loss"])
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=fine_tune_start_epoch + args.fine_tune_epochs,
            initial_epoch=fine_tune_start_epoch,
            callbacks=callbacks_phase2,
        ).history

    # Publish the best checkpoint across both phases, not simply the final
    # early-stopping epoch, so inference uses the model with best validation accuracy.
    model = load_model(config.BEST_MODEL_PATH)
    model.save(config.MODEL_PATH)
    print(f"\nSaved final model -> {config.MODEL_PATH}")
    print(f"Best checkpoint saved during training -> {config.BEST_MODEL_PATH}")

    # Merge and persist training history
    full_history = merge_histories(history1.history, history2)
    with open(config.HISTORY_PATH, "w") as f:
        json.dump(full_history, f, indent=2)
    print(f"Saved training history -> {config.HISTORY_PATH}")

    plot_and_save_curves(full_history, fine_tune_start_epoch)

    print("\nTraining complete. Next step: python src/evaluate.py")


if __name__ == "__main__":
    main()
