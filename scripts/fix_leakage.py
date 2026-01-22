import pandas as pd
from pathlib import Path
from collections import defaultdict
import hashlib
import numpy as np
from PIL import Image
from tqdm import tqdm

def sha1_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def dhash(image_path: str, hash_size: int = 8) -> int:
    with Image.open(image_path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(img, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def load_splits():
    train = pd.read_csv("train.csv")
    val   = pd.read_csv("val.csv")
    test  = pd.read_csv("test.csv")
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    df = pd.concat([train, val, test], ignore_index=True)
    return df

def save_splits(df):
    df[df["split"]=="train"].drop(columns=["split"]).to_csv("train.csv", index=False)
    df[df["split"]=="val"].drop(columns=["split"]).to_csv("val.csv", index=False)
    df[df["split"]=="test"].drop(columns=["split"]).to_csv("test.csv", index=False)

def move_group_to_train(df, indices):
    # If group spans multiple splits, move all members to train (strict eval)
    splits = set(df.loc[indices, "split"].tolist())
    if len(splits) > 1:
        df.loc[indices, "split"] = "train"
        return 1
    return 0

def main():
    df = load_splits()

    # --- Fix 1: exact duplicates via SHA1 ---
    print("\nFixing SHA1 exact duplicates...")
    sha_map = defaultdict(list)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        p = row["path"]
        if not Path(p).exists():
            continue
        try:
            sha = sha1_file(p)
            sha_map[sha].append(i)
        except Exception:
            continue

    exact_groups = [idxs for idxs in sha_map.values() if len(idxs) > 1]
    moved_exact = 0
    for idxs in exact_groups:
        moved_exact += move_group_to_train(df, idxs)

    print(f"Exact duplicate groups: {len(exact_groups)}")
    print(f"Exact duplicate groups moved to train (because they crossed splits): {moved_exact}")

    # --- Fix 2: identical perceptual duplicates (dHash exact match) ---
    print("\nFixing identical dHash duplicates...")
    dh_map = defaultdict(list)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        p = row["path"]
        if not Path(p).exists():
            continue
        try:
            h = dhash(p, hash_size=8)
            dh_map[h].append(i)
        except Exception:
            continue

    dh_groups = [idxs for idxs in dh_map.values() if len(idxs) > 1]
    moved_dh = 0
    for idxs in dh_groups:
        moved_dh += move_group_to_train(df, idxs)

    print(f"Identical dHash groups: {len(dh_groups)}")
    print(f"Identical dHash groups moved to train (because they crossed splits): {moved_dh}")

    # Save updated splits
    save_splits(df)
    print("\nUpdated train/val/test CSVs written.")

    # Report new sizes
    print("\nNew split sizes:")
    print("  train:", (df["split"]=="train").sum())
    print("  val:  ", (df["split"]=="val").sum())
    print("  test: ", (df["split"]=="test").sum())

if __name__ == "__main__":
    main()
