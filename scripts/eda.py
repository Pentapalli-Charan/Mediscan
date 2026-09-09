"""
MediScan -- Exploratory Data Analysis Script (Day 1)

Performs comprehensive dataset inspection and generates reports.

Usage:
    python scripts/eda.py

Outputs:
    - Console summary of all findings
    - reports/figures/class_distribution.png
    - reports/figures/images_per_lesion.png
    - reports/figures/sample_images.png
    - reports/figures/metadata_distributions.png
    - reports/dataset_summary.json
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    load_metadata, analyze_class_distribution, check_image_integrity,
    analyze_duplicates, analyze_metadata_quality, find_metadata_csv,
    find_image_directories,
)
from src.utils.seed import set_seed

# Full class name mapping
CLASS_NAMES = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}


def setup_dirs():
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def locate_data():
    """Locate metadata CSV and image directories."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    metadata_dir = PROJECT_ROOT / "data" / "metadata"

    csv_path = find_metadata_csv(str(metadata_dir))
    if csv_path is None:
        csv_path = find_metadata_csv(str(raw_dir))
    if csv_path is None:
        csv_path = find_metadata_csv(str(PROJECT_ROOT / "data"))

    image_dirs = find_image_directories(str(raw_dir))

    return csv_path, image_dirs


def plot_class_distribution(dist, figures_dir):
    """Generate class distribution bar chart."""
    counts = dist["class_counts"]
    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]
    full_names = [f"{l}\n({CLASS_NAMES.get(l, l)})" for l in labels]
    pcts = [dist["class_percentages"][l] for l in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(labels))
    bars = ax.bar(full_names, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, pct, val in zip(bars, pcts, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val}\n({pct}%)", ha="center", va="bottom", fontsize=9)

    ax.set_title("HAM10000 -- Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Diagnosis Class", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = figures_dir / "class_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_lesion_distribution(dup_stats, figures_dir):
    """Generate images-per-lesion distribution chart."""
    dist = dup_stats.get("distribution", {})
    if not dist:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    keys = sorted(dist.keys())
    vals = [dist[k] for k in keys]

    ax.bar([str(k) for k in keys], vals, color=sns.color_palette("muted")[0],
           edgecolor="black", linewidth=0.5)
    ax.set_title("Distribution of Images per Lesion", fontsize=14, fontweight="bold")
    ax.set_xlabel("Images per Lesion", fontsize=11)
    ax.set_ylabel("Number of Lesions", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for i, (k, v) in enumerate(zip(keys, vals)):
        ax.text(i, v + max(vals) * 0.01, str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = figures_dir / "images_per_lesion.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_sample_images(df, image_dirs, figures_dir, samples_per_class=2):
    """Show representative sample images from each class."""
    if not image_dirs:
        print("  No image directories found, skipping sample images.")
        return

    classes = sorted(df["dx"].unique())
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, samples_per_class,
                             figsize=(4 * samples_per_class, 3.5 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for i, cls in enumerate(classes):
        cls_df = df[df["dx"] == cls].sample(n=min(samples_per_class, len(df[df["dx"] == cls])),
                                             random_state=42)
        for j, (_, row) in enumerate(cls_df.iterrows()):
            img_id = row["image_id"]
            img_path = None
            for d in image_dirs:
                for ext in [".jpg", ".jpeg", ".png"]:
                    candidate = Path(d) / f"{img_id}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
                if img_path:
                    break

            ax = axes[i, j] if samples_per_class > 1 else axes[i]
            if img_path and img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f"{cls} ({CLASS_NAMES.get(cls, cls)})", fontsize=9)
            else:
                ax.text(0.5, 0.5, f"Not found:\n{img_id}", ha="center", va="center")
            ax.axis("off")

    plt.suptitle("Sample Images per Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = figures_dir / "sample_images.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_metadata_distributions(df, figures_dir):
    """Plot distributions of age, sex, localization."""
    plots_made = 0
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Age distribution
    if "age" in df.columns and df["age"].notna().sum() > 0:
        df["age"].dropna().hist(bins=20, ax=axes[0], color=sns.color_palette("muted")[0],
                                 edgecolor="black", linewidth=0.5)
        axes[0].set_title("Age Distribution", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Age")
        axes[0].set_ylabel("Count")
        plots_made += 1
    else:
        axes[0].text(0.5, 0.5, "Age data not available", ha="center", va="center")
        axes[0].set_title("Age Distribution")

    # Sex distribution
    if "sex" in df.columns and df["sex"].notna().sum() > 0:
        sex_counts = df["sex"].value_counts()
        axes[1].bar(sex_counts.index.astype(str), sex_counts.values,
                    color=sns.color_palette("muted")[1:len(sex_counts)+1],
                    edgecolor="black", linewidth=0.5)
        axes[1].set_title("Sex Distribution", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Count")
        plots_made += 1
    else:
        axes[1].text(0.5, 0.5, "Sex data not available", ha="center", va="center")
        axes[1].set_title("Sex Distribution")

    # Localization distribution
    if "localization" in df.columns and df["localization"].notna().sum() > 0:
        loc_counts = df["localization"].value_counts().head(10)
        axes[2].barh(loc_counts.index[::-1], loc_counts.values[::-1],
                     color=sns.color_palette("muted")[2], edgecolor="black", linewidth=0.5)
        axes[2].set_title("Top 10 Localizations", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Count")
        plots_made += 1
    else:
        axes[2].text(0.5, 0.5, "Localization data not available", ha="center", va="center")
        axes[2].set_title("Localization Distribution")

    plt.tight_layout()
    path = figures_dir / "metadata_distributions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    if plots_made > 0:
        print(f"  Saved: {path}")


def build_summary(dist, dup_stats, integrity, meta_quality, image_dirs, csv_path):
    """Build the dataset summary dictionary."""
    summary = {
        "dataset_name": "HAM10000",
        "inspection_timestamp": datetime.now().isoformat(),
        "metadata_csv_path": csv_path,
        "image_directories": image_dirs,
        "metadata_row_count": dist["total"],
        "image_count": integrity.get("total_files", "NOT VERIFIED"),
        "inspected_image_count": integrity.get("inspected_count", "NOT VERIFIED"),
        "class_count": dist["num_classes"],
        "class_labels": dist["class_labels"],
        "class_distribution": dist["class_counts"],
        "class_percentages": dist["class_percentages"],
        "unique_lesions": dup_stats.get("unique_lesions", "NOT VERIFIED"),
        "images_per_lesion_statistics": dup_stats.get("images_per_lesion", {}),
        "multi_image_lesions": dup_stats.get("multi_image_lesions", "NOT VERIFIED"),
        "single_image_lesions": dup_stats.get("single_image_lesions", "NOT VERIFIED"),
        "image_formats": integrity.get("formats", {}),
        "image_modes": integrity.get("modes", {}),
        "sampled_dimensions": integrity.get("dimensions", {}),
        "corrupted_image_count": len(integrity.get("corrupted", [])),
        "corrupted_images": integrity.get("corrupted", []),
        "missing_image_count": len(integrity.get("missing_from_dir", [])),
        "orphan_image_count": len(integrity.get("orphan_files", [])),
        "readable_images": integrity.get("readable", "NOT VERIFIED"),
        "duplicate_image_id_count": "checked in metadata analysis",
        "missing_metadata_summary": meta_quality.get("missing_values", {}),
        "metadata_columns": meta_quality.get("columns", []),
        "leakage_considerations": dup_stats.get("leakage_risk", "NOT VERIFIED"),
        "notes": [
            "Class distribution is heavily imbalanced — 'nv' dominates.",
            "Train/val/test split MUST use lesion_id grouping to prevent data leakage.",
            "This is an educational project — not for clinical diagnosis.",
        ],
    }
    return summary


def main():
    set_seed(42)
    figures_dir = setup_dirs()

    print("=" * 60)
    print("MediScan -- Exploratory Data Analysis (Day 1)")
    print("=" * 60)
    print()

    # ── 1. Locate Data ──
    print("1. Locating dataset files...")
    csv_path, image_dirs = locate_data()

    if csv_path is None:
        print("\n  ERROR: Metadata CSV not found!")
        print("  Please run 'python scripts/download_dataset.py' first,")
        print("  or manually place the dataset in data/raw/")
        print("  Expected: HAM10000_metadata.csv")
        sys.exit(1)

    print(f"  Metadata CSV: {csv_path}")
    print(f"  Image directories: {image_dirs}")
    print()

    # ── 2. Load Metadata ──
    print("2. Loading metadata...")
    df = load_metadata(csv_path)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print()

    # ── 3. Class Distribution ──
    print("3. Analyzing class distribution...")
    dist = analyze_class_distribution(df)
    print(f"  Number of classes: {dist['num_classes']}")
    print(f"  Class labels: {dist['class_labels']}")
    print(f"  Distribution:")
    for label in sorted(dist["class_counts"].keys()):
        name = CLASS_NAMES.get(label, label)
        count = dist["class_counts"][label]
        pct = dist["class_percentages"][label]
        print(f"    {label:>6} ({name:>25}): {count:>6}  ({pct:>5}%)")
    print()

    # ── 4. Lesion / Duplicate Analysis ──
    print("4. Analyzing lesion IDs (data leakage check)...")
    dup_stats = analyze_duplicates(df)
    if "error" not in dup_stats:
        print(f"  Unique lesions: {dup_stats['unique_lesions']}")
        print(f"  Total images: {dup_stats['total_images']}")
        print(f"  Multi-image lesions: {dup_stats['multi_image_lesions']}")
        print(f"  Single-image lesions: {dup_stats['single_image_lesions']}")
        ipl = dup_stats["images_per_lesion"]
        print(f"  Images per lesion -- min: {ipl['min']}, max: {ipl['max']}, "
              f"mean: {ipl['mean']}, median: {ipl['median']}")
        print(f"  [WARNING] LEAKAGE RISK: {dup_stats['leakage_risk']}")
    else:
        print(f"  {dup_stats['error']}")
    print()

    # ── 5. Metadata Quality ──
    print("5. Inspecting metadata quality...")
    meta_quality = analyze_metadata_quality(df)
    print(f"  Columns: {meta_quality['columns']}")
    print(f"  Missing values:")
    for col, count in meta_quality["missing_values"].items():
        pct = meta_quality["missing_percentages"][col]
        status = "[OK]" if count == 0 else f"[WARNING] {count} missing ({pct}%)"
        print(f"    {col:>15}: {status}")

    # Check for duplicate image IDs
    if "image_id" in df.columns:
        dup_ids = df["image_id"].duplicated().sum()
        print(f"  Duplicate image_id entries: {dup_ids}")
    print()

    # ── 6. Image Integrity ──
    print("6. Checking image integrity...")
    if image_dirs:
        # Collect all images from all directories
        all_image_ids = df["image_id"].tolist() if "image_id" in df.columns else None

        # For full check we inspect all images; for speed use sample
        total_expected = len(df)
        # Check all images for a thorough Day 1 inspection
        integrity = {"total_files": 0, "readable": 0, "corrupted": [],
                     "dimensions": {}, "formats": {}, "modes": {},
                     "missing_from_dir": [], "orphan_files": [], "inspected_count": 0}

        for img_dir in image_dirs:
            result = check_image_integrity(img_dir, image_ids=all_image_ids)
            integrity["total_files"] += result["total_files"]
            integrity["readable"] += result["readable"]
            integrity["corrupted"].extend(result["corrupted"])
            integrity["inspected_count"] += result["inspected_count"]
            for k, v in result["dimensions"].items():
                integrity["dimensions"][k] = integrity["dimensions"].get(k, 0) + v
            for k, v in result["formats"].items():
                integrity["formats"][k] = integrity["formats"].get(k, 0) + v
            for k, v in result["modes"].items():
                integrity["modes"][k] = integrity["modes"].get(k, 0) + v

        # Recalculate missing/orphan across all dirs
        if all_image_ids is not None:
            from src.data.preprocessing import find_image_files
            all_file_stems = set()
            for img_dir in image_dirs:
                files = find_image_files(img_dir)
                all_file_stems.update(f.stem for f in files)
            metadata_ids = set(str(img_id) for img_id in all_image_ids)
            integrity["missing_from_dir"] = sorted(metadata_ids - all_file_stems)
            integrity["orphan_files"] = sorted(all_file_stems - metadata_ids)

        print(f"  Total image files found: {integrity['total_files']}")
        print(f"  Images inspected: {integrity['inspected_count']}")
        print(f"  Readable: {integrity['readable']}")
        print(f"  Corrupted: {len(integrity['corrupted'])}")
        print(f"  Formats: {integrity['formats']}")
        print(f"  Modes: {integrity['modes']}")
        print(f"  Dimensions: {integrity['dimensions']}")
        print(f"  Missing (in metadata, not on disk): {len(integrity['missing_from_dir'])}")
        print(f"  Orphan (on disk, not in metadata): {len(integrity['orphan_files'])}")
        if integrity["corrupted"]:
            print(f"  Corrupted files:")
            for c in integrity["corrupted"][:10]:
                print(f"    {c['file']}: {c['error']}")
    else:
        print("  No image directories found! Cannot verify images.")
        integrity = {"total_files": 0, "readable": 0, "corrupted": [],
                     "dimensions": {}, "formats": {}, "modes": {},
                     "missing_from_dir": [], "orphan_files": [], "inspected_count": 0}
    print()

    # ── 7. Generate Visualizations ──
    print("7. Generating visualizations...")
    plot_class_distribution(dist, figures_dir)
    plot_lesion_distribution(dup_stats, figures_dir)
    if image_dirs:
        plot_sample_images(df, image_dirs, figures_dir)
    plot_metadata_distributions(df, figures_dir)
    print()

    # ── 8. Save Summary ──
    print("8. Saving dataset summary...")
    summary = build_summary(dist, dup_stats, integrity, meta_quality, image_dirs, csv_path)
    summary_path = PROJECT_ROOT / "reports" / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")
    print()

    # ── 9. Final Summary ──
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Dataset: HAM10000")
    print(f"  Metadata rows: {dist['total']}")
    print(f"  Image files found: {integrity['total_files']}")
    print(f"  Classes: {dist['num_classes']} -- {dist['class_labels']}")
    print(f"  Unique lesions: {dup_stats.get('unique_lesions', 'N/A')}")
    print(f"  Corrupted images: {len(integrity['corrupted'])}")
    print(f"  Missing images: {len(integrity['missing_from_dir'])}")
    print()
    print("DATA LEAKAGE WARNING:")
    print(f"  {dup_stats.get('leakage_risk', 'Cannot assess -- lesion_id not found')}")
    print()
    print("NEXT STEPS (Day 2):")
    print("  1. Implement lesion_id-aware stratified train/val/test split (70/15/15)")
    print("  2. Resize images to 224x224")
    print("  3. Apply ImageNet normalization")
    print("  4. Verify split integrity (no leakage)")
    print("=" * 60)


if __name__ == "__main__":
    main()
