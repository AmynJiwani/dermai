import tensorflow as tf
import pandas as pd
from pathlib import Path
from tqdm import tqdm

IMG_SIZE = 224

def _bytes_feature(v): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
def _int64_feature(v): return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))

def encode_example(image_bytes, cond_id, age_id):
    feat = {
        "image": _bytes_feature(image_bytes),
        "condition": _int64_feature(cond_id),
        "age_group": _int64_feature(age_id),
    }
    return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()

def read_and_encode(path: str) -> bytes:
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.uint8)
    return tf.io.encode_jpeg(img, quality=95).numpy()

def write_tfrecord(csv_path: str, out_path: str, cond_map: dict, age_map: dict):
    df = pd.read_csv(csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tf.io.TFRecordWriter(str(out_path)) as w:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {out_path.name}"):
            img_bytes = read_and_encode(row["path"])
            cond_id = cond_map[row["condition"]]
            age_id = age_map[row["age_group"]]
            w.write(encode_example(img_bytes, cond_id, age_id))

def main():
    train_df = pd.read_csv("train.csv")

    conditions = sorted(train_df["condition"].unique().tolist())
    ages = sorted(train_df["age_group"].unique().tolist())

    cond_map = {c:i for i,c in enumerate(conditions)}
    age_map = {a:i for i,a in enumerate(ages)}

    pd.Series(cond_map).to_json("condition_map.json")
    pd.Series(age_map).to_json("age_map.json")

    print("Condition map:", cond_map)
    print("Age map:", age_map)

    write_tfrecord("train.csv", "tfrecords/train.tfrecord", cond_map, age_map)
    write_tfrecord("val.csv",   "tfrecords/val.tfrecord",   cond_map, age_map)
    write_tfrecord("test.csv",  "tfrecords/test.tfrecord",  cond_map, age_map)

if __name__ == "__main__":
    main()
