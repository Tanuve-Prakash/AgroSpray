#!/usr/bin/env python3
"""
Train the AgroSpray VGG16 classifier (transfer learning) and export for the Raspberry Pi.

Dataset layout (one folder per class, folder names become labels):
  dataset/
    betel_healthy/  betel_diseased/
    malabar_spinach_healthy/  malabar_spinach_diseased/
    potato_healthy/  potato_diseased/
Keep "healthy" in the folder name of every healthy class - the Pi code relies on it.

Run:  python3 train_vgg16.py --data dataset --out models
Outputs: models/agrospray_vgg16.keras, models/agrospray_vgg16.tflite, models/class_names.json
Note: the saved models expect VGG16-preprocessed input (RGB->BGR, minus ImageNet mean);
the Pi script does exactly that.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

IMG = (224, 224)


def build_model(n_classes: int):
    base = VGG16(weights="imagenet", include_top=False, input_shape=IMG + (3,))
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(base.input, out), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="models")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--head-epochs", type=int, default=10)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    common = dict(validation_split=0.2, seed=42, image_size=IMG,
                  batch_size=args.batch, label_mode="categorical")
    train_ds = tf.keras.utils.image_dataset_from_directory(args.data, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(args.data, subset="validation", **common)
    class_names = train_ds.class_names
    (out / "class_names.json").write_text(json.dumps(class_names, indent=2))
    print("Classes:", class_names)

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ])
    autotune = tf.data.AUTOTUNE
    train_ds = (train_ds.map(lambda x, y: (preprocess_input(augment(x, training=True)), y),
                             num_parallel_calls=autotune).prefetch(autotune))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y),
                        num_parallel_calls=autotune).prefetch(autotune)

    model, base = build_model(len(class_names))
    ckpt = tf.keras.callbacks.ModelCheckpoint(str(out / "agrospray_vgg16.keras"),
                                              monitor="val_accuracy", save_best_only=True)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                             restore_best_weights=True)

    # Phase 1: train the new head only
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, callbacks=[ckpt, early])

    # Phase 2: fine-tune the last conv block at a low learning rate
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith("block5")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs, callbacks=[ckpt, early])

    model = tf.keras.models.load_model(out / "agrospray_vgg16.keras")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {acc * 100:.2f}%")

    # Per-class report
    try:
        from sklearn.metrics import classification_report
        y_true, y_pred = [], []
        for x, y in val_ds:
            y_true += list(np.argmax(y.numpy(), axis=1))
            y_pred += list(np.argmax(model.predict(x, verbose=0), axis=1))
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    except ImportError:
        print("pip install scikit-learn for the per-class report")

    # Export for the Pi (dynamic-range quantisation shrinks the file ~4x)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    (out / "agrospray_vgg16.tflite").write_bytes(conv.convert())
    print("Saved:", *(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
