import os
import csv
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def main():
    root = Path("Derma AI Images")
    out_csv = Path("data_index.csv")

    if not root.exists():
        raise FileNotFoundError(
            f"Can't find dataset folder: {root.resolve()}\n"
            "Make sure 'Derma AI Images' is in the project root."
        )

    rows = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            # Expected path:
            # Derma AI Images / <Condition> / <AgeGroup> / <Source> / <image>
            parts = p.parts
            try:
                idx = parts.index("Derma AI Images")
                condition = parts[idx + 1]
                age_group = parts[idx + 2]
                source = parts[idx + 3]
            except Exception:
                # Skip anything that doesn't match expected structure
                continue

            rows.append({
                "path": str(p.as_posix()),
                "condition": condition,
                "age_group": age_group,
                "source": source,
            })

    if not rows:
        raise RuntimeError(
            "No images found. Check your folder structure and file extensions."
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "condition", "age_group", "source"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_csv}")

if __name__ == "__main__":
    main()
