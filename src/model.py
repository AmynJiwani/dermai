import tensorflow as tf

def build_model(num_conditions: int, num_ages: int):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(224, 224, 3),
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False  # warm-up

    inp = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inp * 255.0)
    x = base(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    cond_out = tf.keras.layers.Dense(num_conditions, activation="softmax", name="condition")(x)
    age_out  = tf.keras.layers.Dense(num_ages, activation="softmax", name="age_group")(x)

    return tf.keras.Model(inp, {"condition": cond_out, "age_group": age_out})
