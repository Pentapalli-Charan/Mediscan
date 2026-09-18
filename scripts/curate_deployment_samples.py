import json
import shutil
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

train_df = pd.read_csv(PROJECT_ROOT / "reports/data/split_train.csv")
dirs = [
    PROJECT_ROOT / "data/raw/HAM10000_images_part_1",
    PROJECT_ROOT / "data/raw/HAM10000_images_part_2",
]
classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
descriptions = {
    "akiec": "Actinic keratoses / Intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

dest_dir = PROJECT_ROOT / "app/assets/samples"
dest_dir.mkdir(parents=True, exist_ok=True)

manifest = []
for c in classes:
    sub = train_df[train_df["dx"] == c]
    for _, row in sub.iterrows():
        img_id = row["image_id"]
        found = None
        for d in dirs:
            p = d / f"{img_id}.jpg"
            if p.exists():
                found = p
                break
        if found:
            dest_file = dest_dir / f"{img_id}.jpg"
            shutil.copy2(found, dest_file)
            manifest.append({
                "image_id": img_id,
                "dx": c,
                "description": descriptions[c],
                "split": "train",
                "filename": f"{img_id}.jpg",
                "file_size_bytes": dest_file.stat().st_size,
                "source": "HAM10000 (Tschandl et al., 2018; CC BY-NC-SA 4.0)",
                "lesion_id": str(row.get("lesion_id", "")),
                "dx_type": str(row.get("dx_type", "")),
                "age": float(row["age"]) if pd.notna(row.get("age")) else None,
                "sex": str(row.get("sex", "")),
                "localization": str(row.get("localization", "")),
            })
            break

with open(dest_dir / "samples_metadata.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Curated {len(manifest)} samples in {dest_dir}")
for m in manifest:
    print(f"  {m['dx']}: {m['image_id']} ({m['file_size_bytes']} bytes)")
