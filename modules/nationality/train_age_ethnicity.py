"""
train_age_ethnicity.py

Transfer-learning training script for a MULTI-TASK model:
    Input:  face image
    Output: (1) age  -> regression
            (2) ethnicity-proxy -> classification (5 classes, UTKFace's own
                labels: White, Black, Asian, Indian, Others)

Dataset: UTKFace (https://susanqq.github.io/UTKFace/)
  - Filenames encode labels: [age]_[gender]_[race]_[date&time].jpg
  - race: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
  - Download and unzip so all images sit in one folder, e.g. data/UTKFace/

Why one model for both age and ethnicity: they share the same dataset and
the same useful low/mid-level face features, so a shared MobileNetV2 trunk
with two heads is more data-efficient than training two separate backbones.

Usage:
    python train_age_ethnicity.py --data_dir data/UTKFace --epochs 15
"""

import os
import re
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
NUM_RACE_CLASSES = 5
RACE_NAMES = ["white", "black", "asian", "indian", "others"]  # index order fixed by UTKFace


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_utkface_filename(filename):
    """
    UTKFace filenames look like: 25_0_1_20170116174525125.jpg.chip.jpg
    Returns (age:int, race:int) or None if the filename doesn't match.
    """
    base = os.path.basename(filename)
    m = re.match(r"^(\d+)_(\d+)_(\d+)_", base)
    if not m:
        return None
    age = int(m.group(1))
    race = int(m.group(3))
    if race < 0 or race >= NUM_RACE_CLASSES:
        return None
    if age < 0 or age > 120:
        return None
    return age, race


def build_dataframe(data_dir):
    paths, ages, races = [], [], []
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parsed = parse_utkface_filename(fname)
        if parsed is None:
            continue
        age, race = parsed
        paths.append(os.path.join(data_dir, fname))
        ages.append(age)
        races.append(race)
    return paths, np.array(ages, dtype=np.float32), np.array(races, dtype=np.int32)


def make_tf_dataset(paths, ages, races, batch_size, shuffle=True, augment=False):
    def _load(path, age, race):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.15)
        img = preprocess_input(img)  # MobileNetV2's expected [-1, 1] scaling
        race_onehot = tf.one_hot(race, NUM_RACE_CLASSES)
        return img, {"age_output": age, "race_output": race_onehot}

    ds = tf.data.Dataset.from_tensor_slices((paths, ages, races))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(fine_tune_last_n=30):
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False  # phase 1: frozen backbone

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(0.3)(x)

    # Age head (regression)
    age_branch = layers.Dense(128, activation="relu")(x)
    age_branch = layers.Dropout(0.2)(age_branch)
    age_output = layers.Dense(1, activation="linear", name="age_output")(age_branch)

    # Ethnicity-proxy head (classification)
    race_branch = layers.Dense(128, activation="relu")(x)
    race_branch = layers.Dropout(0.2)(race_branch)
    race_output = layers.Dense(NUM_RACE_CLASSES, activation="softmax", name="race_output")(race_branch)

    model = models.Model(inputs=inputs, outputs=[age_output, race_output])
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss={"age_output": "mse", "race_output": "categorical_crossentropy"},
        loss_weights={"age_output": 0.01, "race_output": 1.0},
        metrics={"age_output": "mae", "race_output": "accuracy"},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to folder of UTKFace images")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fine_tune_epochs", type=int, default=8)
    ap.add_argument("--fine_tune_lr", type=float, default=1e-5)
    ap.add_argument("--out_path", default="models/age_ethnicity_model.h5")
    args = ap.parse_args()

    print("Indexing dataset...")
    paths, ages, races = build_dataframe(args.data_dir)
    print(f"Found {len(paths)} labeled images.")

    train_p, val_p, train_a, val_a, train_r, val_r = train_test_split(
        paths, ages, races, test_size=0.15, random_state=42, stratify=races
    )

    train_ds = make_tf_dataset(train_p, train_a, train_r, args.batch_size, shuffle=True, augment=True)
    val_ds = make_tf_dataset(val_p, val_a, val_r, args.batch_size, shuffle=False, augment=False)

    model, base = build_model()
    compile_model(model, args.lr)
    model.summary()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        args.out_path, save_best_only=True, monitor="val_loss"
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    print("\n=== Phase 1: training heads with frozen MobileNetV2 backbone ===")
    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=[ckpt, early_stop],
    )
    phase1_epochs = len(history1.history["loss"])

    print("\n=== Phase 2: fine-tuning top layers of the backbone ===")
    base.trainable = True
    # Freeze all but the last N layers of the backbone for gentle fine-tuning
    for layer in base.layers[:-30]:
        layer.trainable = False
    compile_model(model, args.fine_tune_lr)  # lower LR for fine-tuning

    # Reset early stopping so it doesn't carry stale patience from Phase 1
    early_stop_ft = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )
    model.fit(
        train_ds, validation_data=val_ds,
        initial_epoch=phase1_epochs,
        epochs=phase1_epochs + args.fine_tune_epochs,
        callbacks=[ckpt, early_stop_ft],
    )

    model.save(args.out_path)
    print(f"\nSaved final model to {args.out_path}")


if __name__ == "__main__":
    main()
