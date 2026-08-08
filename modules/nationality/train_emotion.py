"""
train_emotion.py

Transfer-learning training script for EMOTION classification via MobileNetV2.
Matches the same pattern as train_age_ethnicity.py, so if you already have a
trained emotion model from your RAF-DB Streamlit project, you can SKIP this
file entirely and just point inference_utils.py at your existing .h5 file
(see EMOTION_LABELS below - update to match your model's class order).

Expects a directory layout like:
    data/emotion/
        train/
            angry/
            disgust/
            fear/
            happy/
            neutral/
            sad/
            surprise/
        val/
            angry/
            ... (same subfolders)

This works directly with RAF-DB (re-split into class folders) or FER2013
(converted to images) - whichever you already used.

Usage:
    python train_emotion.py --data_dir data/emotion --epochs 15
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, optimizers

IMG_SIZE = 224
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def build_datasets(data_dir, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="categorical",
        class_names=EMOTION_LABELS,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="categorical",
        class_names=EMOTION_LABELS,
    )

    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.15),
        layers.RandomRotation(0.05),
    ])

    def prep(x, y, training):
        if training:
            x = aug(x)
        x = preprocess_input(x)
        return x, y

    train_ds = train_ds.map(lambda x, y: prep(x, y, True)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: prep(x, y, False)).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def build_model(num_classes):
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = models.Model(inputs, outputs)
    return model, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fine_tune_epochs", type=int, default=8)
    ap.add_argument("--fine_tune_lr", type=float, default=1e-5)
    ap.add_argument("--out_path", default="models/emotion_model.h5")
    args = ap.parse_args()

    train_ds, val_ds = build_datasets(args.data_dir, args.batch_size)
    model, base = build_model(len(EMOTION_LABELS))

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(args.out_path, save_best_only=True, monitor="val_loss")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

    print("=== Phase 1: frozen backbone ===")
    model.compile(optimizer=optimizers.Adam(args.lr), loss="categorical_crossentropy", metrics=["accuracy"])
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[ckpt, early_stop])
    phase1_epochs = len(history1.history["loss"])

    print("=== Phase 2: fine-tuning ===")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(args.fine_tune_lr), loss="categorical_crossentropy", metrics=["accuracy"])

    early_stop_ft = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(
        train_ds, validation_data=val_ds,
        initial_epoch=phase1_epochs,
        epochs=phase1_epochs + args.fine_tune_epochs,
        callbacks=[ckpt, early_stop_ft],
    )

    model.save(args.out_path)
    print(f"Saved final model to {args.out_path}")


if __name__ == "__main__":
    main()
