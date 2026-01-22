# src/train.py
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import make_dataset
from src.model import build_model

BATCH_SIZE = 32
EPOCHS_WARMUP = 5
EPOCHS_FINETUNE = 8
LR_WARMUP = 1e-3
LR_FINETUNE = 1e-5


def main():
   
    # Load label maps
   
    cond_map = json.loads(open("condition_map.json", "r").read())
    age_map = json.loads(open("age_map.json", "r").read())

    num_conditions = len(cond_map)
    num_ages = len(age_map)

   
    # Load splits (for weights + step counts)
   
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")

    train_steps = math.ceil(len(train_df) / BATCH_SIZE)
    val_steps = math.ceil(len(val_df) / BATCH_SIZE)

   
    # Build tf.data datasets
   
    train_ds = make_dataset("tfrecords/train.tfrecord", BATCH_SIZE, training=True)
    val_ds = make_dataset("tfrecords/val.tfrecord", BATCH_SIZE, training=False)

   
    # Condition class weights from TRAIN split
   
    y = train_df["condition"].map(cond_map).values
    weights = compute_class_weight("balanced", classes=np.arange(num_conditions), y=y)
    cond_class_weight = tf.constant(weights, dtype=tf.float32)
    print("Condition class weights:", {i: float(weights[i]) for i in range(num_conditions)})

    # Apply sample weights to condition head only
    def add_sample_weights(img, labels):
        cond = labels["condition"]
        w = tf.gather(cond_class_weight, cond)
        sample_weights = {
            "condition": w,
            "age_group": tf.ones_like(w),
        }
        return img, labels, sample_weights

    train_w_ds = train_ds.map(add_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)

   
    # Build model
   
    model = build_model(num_conditions, num_ages)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/best.keras",
            monitor="val_condition_acc",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_condition_acc",
            patience=3,
            restore_best_weights=True,
        ),
    ]

   
    # Warm-up (frozen backbone)
   
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_WARMUP),
        loss={
            "condition": tf.keras.losses.SparseCategoricalCrossentropy(),
            "age_group": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        metrics={
            "condition": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
            "age_group": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        },
    )

    print("\n=== Warm-up (frozen backbone) ===")
    model.fit(
        train_w_ds,
        validation_data=val_ds,
        epochs=EPOCHS_WARMUP,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

   
    # Fine-tune (unfreeze top of backbone)
   
    print("\n=== Fine-tune (unfreeze top of backbone) ===")

    # NOTE: EfficientNetB0 base is named "efficientnetb0" by Keras
    base = model.get_layer("efficientnetb0")
    base.trainable = True

    # Freeze early layers to reduce overfitting (tweak cutoff as needed)
    for layer in base.layers[:200]:
        layer.trainable = False

    # IMPORTANT: re-declare loss/metrics fresh (avoid sample_weight metric bug)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
        loss={
            "condition": tf.keras.losses.SparseCategoricalCrossentropy(),
            "age_group": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        metrics={
            "condition": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
            "age_group": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        },
    )

    model.fit(
        train_w_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

   
    # Save final model
   
    model.save("models/final.keras")
    print("Saved -> models/final.keras")


if __name__ == "__main__":
    main()
