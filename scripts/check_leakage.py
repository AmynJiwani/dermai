import hashlib
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

def dhash(image_path: str, hash_size: int = 8) -> int:
    """Perceptual hash (dHash). Returns 64-bit int for hash_size=8."""
    with Image.open(image_path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(img, dtype=np.int16)

    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()

    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def sha1_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def load_split(csv_path: str, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["split"] = split_name
    return df

def main():
    train = load_split("train.csv", "train")
    val   = load_split("val.csv", "val")
    test  = load_split("test.csv", "test")
    df = pd.concat([train, val, test], ignore_index=True)

    print(f"Total images in CSVs: {len(df)}")

    # --- Exact duplicates (SHA1) ---
    print("\n[1/3] Computing SHA1 (exact duplicates)...")
    sha_map = defaultdict(list)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        p = row["path"]
        if not Path(p).exists():
            continue
        try:
            sha_map[sha1_file(p)].append(i)
        except Exception:
            continue

    exact_groups = [idxs for idxs in sha_map.values() if len(idxs) > 1]
    print(f"Exact-duplicate groups found: {len(exact_groups)}")

    cross_exact = 0
    for idxs in exact_groups:
        splits = set(df.loc[idxs, "split"].tolist())
        if len(splits) > 1:
            cross_exact += 1
    print(f"Exact-duplicate groups crossing splits: {cross_exact}")

    # --- Near duplicates (dHash) ---
    print("\n[2/3] Computing dHash (near duplicates)...")
    hashes = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        p = row["path"]
        if not Path(p).exists():
            continue
        try:
            hashes[i] = dhash(p, hash_size=8)
        except Exception:
            continue

    # Identical dHash groups (fast)
    inv = defaultdict(list)
    for idx, h in hashes.items():
        inv[h].append(idx)

    ident_phash_groups = [idxs for idxs in inv.values() if len(idxs) > 1]
    print(f"Identical dHash groups found: {len(ident_phash_groups)}")

    cross_ident = 0
    for idxs in ident_phash_groups:
        splits = set(df.loc[idxs, "split"].tolist())
        if len(splits) > 1:
            cross_ident += 1
    print(f"Identical dHash groups crossing splits: {cross_ident}")

    # Near duplicates via bucket + hamming threshold
    print("\n[3/3] Scanning for near duplicates (Hamming <= 5)...")
    threshold = 5
    buckets = defaultdict(list)
    for idx, h in hashes.items():
        bucket_key = h >> 48  # top 16 bits
        buckets[bucket_key].append((idx, h))

    near_cross = 0
    near_total = 0
    examples = []

    for _, items in tqdm(buckets.items(), total=len(buckets)):
        for a in range(len(items)):
            ia, ha = items[a]
            for b in range(a + 1, len(items)):
                ib, hb = items[b]
                d = hamming(ha, hb)
                if d <= threshold:
                    near_total += 1
                    sa = df.loc[ia, "split"]
                    sb = df.loc[ib, "split"]
                    if sa != sb:
                        near_cross += 1
                        if len(examples) < 8:
                            examples.append((ia, ib, d))

    print(f"Near-duplicate pairs found: {near_total}")
    print(f"Near-duplicate pairs crossing splits: {near_cross}")

    if examples:
        print("\nExamples (up to 8):")
        for ia, ib, d in examples:
            print(f"\nHamming={d}")
            print(df.loc[ia, ["split", "path", "condition", "age_group"]].to_string())
            print(df.loc[ib, ["split", "path", "condition", "age_group"]].to_string())

    # Write summary report
    out_rows = []
    for sha, idxs in sha_map.items():
        if len(idxs) > 1:
            splits = ",".join(sorted(set(df.loc[idxs, "split"])))
            if len(set(df.loc[idxs, "split"])) > 1:
                kind = "exact_sha1_cross_split"
            else:
                kind = "exact_sha1_same_split"
            for idx in idxs:
                out_rows.append({
                    "type": kind,
                    "group": sha,
                    "split": df.loc[idx, "split"],
                    "path": df.loc[idx, "path"],
                    "condition": df.loc[idx, "condition"],
                    "age_group": df.loc[idx, "age_group"],
                    "splits_in_group": splits
                })

    if out_rows:
        pd.DataFrame(out_rows).to_csv("leakage_report.csv", index=False)
        print("\nWrote leakage_report.csv (exact-duplicate groups)")
    else:
        print("\nNo exact-duplicate groups report written (none found).")

if __name__ == "__main__":
    main()
