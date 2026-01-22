import tensorflow as tf

IMG_SIZE = 224

FEATURE_SPEC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "condition": tf.io.FixedLenFeature([], tf.int64),
    "age_group": tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_SPEC)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    y_cond = tf.cast(ex["condition"], tf.int32)
    y_age = tf.cast(ex["age_group"], tf.int32)
    return img, {"condition": y_cond, "age_group": y_age}

def augment(img, labels):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.08)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, labels

def make_dataset(tfrecord_path, batch_size, training):
    ds = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2000)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
