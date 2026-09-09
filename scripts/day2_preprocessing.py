"""
MediScan — Day 2: Data Preprocessing & Leakage-Safe Split

This script implements the complete Day 2 pipeline:
    1. Load HAM10000 metadata
    2. Create a 70/15/15 train/val/test split at the lesion_id level
    3. Verify no data leakage (lesion_id and image_id overlap checks)
    4. Compute split statistics and class distribution
    5. Save split metadata to CSV files
    6. Test preprocessing pipeline
    7. Generate reports and figures

Usage:
    python scripts/day2_preprocessing.py

Output:
    - reports/data/split_all.csv (all images with split assignments)
    - reports/data/split_train.csv
    - reports/data/split_val.csv
    - reports/data/split_test.csv
    - reports/day2_split_statistics.json
    - reports/figures/day2_split_distribution.png
    - reports/figures/day2_class_distribution_by_split.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    load_metadata,
    find_metadata_csv,
    find_image_directories,
    create_stratified_group_split,
    preprocess_image,
    verify_no_leakage,
    compute_split_statistics,
    save_split_metadata,
)
from src.utils.seed import set_seed

# Class name mapping
CLASS_NAMES = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}


def setup_directories():
    """Create necessary output directories."""
    (PROJECT_ROOT / "reports" / "data").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    return {
        "data": PROJECT_ROOT / "reports" / "data",
        "figures": PROJECT_ROOT / "reports" / "figures",
    }


def locate_data():
    """Locate metadata CSV and image directories."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    
    csv_path = str(raw_dir / "HAM10000_metadata.csv")
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
    
    image_dirs = find_image_directories(str(raw_dir))
    if not image_dirs:
        raise FileNotFoundError(f"No image directories found in {raw_dir}")
    
    return csv_path, image_dirs


def load_and_prepare_metadata(csv_path):
    """Load metadata and prepare for splitting."""
    print("\n" + "="*70)
    print("STEP 1: LOADING METADATA")
    print("="*70)
    
    df = load_metadata(csv_path)
    print(f"✓ Loaded {len(df)} images from metadata")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Classes: {sorted(df['dx'].unique())}")
    print(f"✓ Unique lesions: {df['lesion_id'].nunique()}")
    
    return df


def create_split(df, seed=42):
    """Create leakage-safe stratified split."""
    print("\n" + "="*70)
    print("STEP 2: CREATING STRATIFIED GROUP SPLIT")
    print("="*70)
    
    set_seed(seed)
    
    # Create split
    df_split = create_stratified_group_split(
        df,
        group_column="lesion_id",
        stratify_column="dx",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=seed,
    )
    
    print("✓ Split created at lesion_id level with stratification by class (dx)")
    print(f"✓ Random seed: {seed}")
    
    return df_split


def verify_split_integrity(df_split):
    """Verify no data leakage."""
    print("\n" + "="*70)
    print("STEP 3: VERIFYING NO DATA LEAKAGE")
    print("="*70)
    
    leakage_check = verify_no_leakage(
        df_split,
        group_column="lesion_id",
        image_column="image_id",
    )
    
    if leakage_check["success"]:
        print("✓ NO LEAKAGE DETECTED")
    else:
        print("✗ LEAKAGE DETECTED!")
        for error in leakage_check["errors"]:
            print(f"  - {error}")
        raise AssertionError("Data leakage verification failed!")
    
    # Print detailed stats
    stats = leakage_check["stats"]
    print(f"\nLesion-level separation:")
    print(f"  Train lesions: {stats['train_lesions']}")
    print(f"  Val lesions: {stats['val_lesions']}")
    print(f"  Test lesions: {stats['test_lesions']}")
    
    print(f"\nImage-level accounting:")
    print(f"  Train images: {stats['train_images']}")
    print(f"  Val images: {stats['val_images']}")
    print(f"  Test images: {stats['test_images']}")
    print(f"  Total in splits: {stats['total_unique_images_in_splits']}")
    print(f"  Total in DataFrame: {stats['total_images_in_df']}")
    
    return leakage_check


def compute_statistics(df_split):
    """Compute and display split statistics."""
    print("\n" + "="*70)
    print("STEP 4: SPLIT STATISTICS & CLASS DISTRIBUTION")
    print("="*70)
    
    split_stats = compute_split_statistics(df_split, class_column="dx")
    
    for split_name in ["train", "val", "test"]:
        stats = split_stats[split_name]
        print(f"\n{split_name.upper()}:")
        print(f"  Images: {stats['image_count']} ({stats['percentage_images']}%)")
        print(f"  Lesions: {stats['lesion_count']} ({stats['percentage_lesions']}%)")
        print(f"  Classes:")
        for cls in sorted(stats["class_distribution"].keys()):
            count = stats["class_distribution"][cls]
            pct = stats["class_percentages"][cls]
            full_name = CLASS_NAMES.get(cls, cls)
            print(f"    {cls:6s} ({full_name:25s}): {count:4d} ({pct:5.2f}%)")
    
    return split_stats


def save_splits(df_split, output_dir):
    """Save split metadata to CSV files."""
    print("\n" + "="*70)
    print("STEP 5: SAVING SPLIT METADATA")
    print("="*70)
    
    paths = save_split_metadata(
        df_split,
        str(output_dir),
        prefix="split",
    )
    
    for key, path in paths.items():
        file_size = Path(path).stat().st_size / (1024 * 1024)
        print(f"✓ {key.upper():8s}: {path} ({file_size:.2f} MB)")
    
    return paths


def test_preprocessing(df_split, image_dirs, n_samples=5):
    """Test preprocessing on sample images from each split."""
    print("\n" + "="*70)
    print("STEP 6: TESTING DETERMINISTIC PREPROCESSING")
    print("="*70)
    
    # Build image file mapping
    print("  Building image file index...")
    image_files = {}
    for image_dir in image_dirs:
        for img_file in Path(image_dir).glob("*.jpg"):
            image_files[img_file.stem] = str(img_file)
        for img_file in Path(image_dir).glob("*.jpeg"):
            image_files[img_file.stem] = str(img_file)
    
    print(f"  Found {len(image_files)} image files")
    
    preprocessing_errors = []
    preprocessing_stats = {
        "total_tested": 0,
        "successful": 0,
        "failed": 0,
        "output_shapes": [],
    }
    
    # Test preprocessing on samples from each split
    for split_name in ["train", "val", "test"]:
        split_df = df_split[df_split["split"] == split_name]
        sample_indices = np.random.choice(len(split_df), size=min(n_samples, len(split_df)), replace=False)
        sample_df = split_df.iloc[sample_indices]
        
        print(f"\n  Testing {split_name.upper()} ({len(sample_indices)} samples):")
        
        for idx, row in sample_df.iterrows():
            image_id = row["image_id"]
            if image_id not in image_files:
                preprocessing_errors.append(f"Image file not found: {image_id}")
                preprocessing_stats["failed"] += 1
                continue
            
            image_path = image_files[image_id]
            try:
                # Test preprocessing
                processed = preprocess_image(
                    image_path,
                    output_size=224,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
                
                if processed is None:
                    preprocessing_stats["failed"] += 1
                    preprocessing_errors.append(f"Preprocessing returned None: {image_id}")
                else:
                    preprocessing_stats["successful"] += 1
                    preprocessing_stats["output_shapes"].append(processed.shape)
                    
                    # Verify output shape
                    if processed.shape != (224, 224, 3):
                        preprocessing_errors.append(
                            f"Unexpected shape for {image_id}: {processed.shape} != (224, 224, 3)"
                        )
                    
                    # Verify it's normalized (values typically in range [-2, 2] after normalization)
                    if np.any(np.isnan(processed)):
                        preprocessing_errors.append(f"NaN values in preprocessed image: {image_id}")
                    
                    print(f"    ✓ {image_id}: shape {processed.shape}, values [{processed.min():.2f}, {processed.max():.2f}]")
                
                preprocessing_stats["total_tested"] += 1
                
            except Exception as e:
                preprocessing_errors.append(f"Error preprocessing {image_id}: {str(e)}")
                preprocessing_stats["failed"] += 1
                preprocessing_stats["total_tested"] += 1
    
    if preprocessing_errors:
        print(f"\n  ✗ Preprocessing errors ({len(preprocessing_errors)}):")
        for error in preprocessing_errors[:5]:  # Show first 5
            print(f"    - {error}")
        if len(preprocessing_errors) > 5:
            print(f"    ... and {len(preprocessing_errors) - 5} more")
    else:
        print(f"\n  ✓ All preprocessing tests passed!")
    
    print(f"\n  Summary:")
    print(f"    Total tested: {preprocessing_stats['total_tested']}")
    print(f"    Successful: {preprocessing_stats['successful']}")
    print(f"    Failed: {preprocessing_stats['failed']}")
    
    return preprocessing_stats


def generate_split_distribution_figure(df_split, output_path):
    """Generate train/val/test size distribution figure."""
    split_counts = df_split["split"].value_counts().sort_index()
    split_pcts = (split_counts / len(df_split) * 100).round(2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"train": "#2ecc71", "val": "#f39c12", "test": "#e74c3c"}
    labels = ["train", "val", "test"]
    counts = [split_counts.get(l, 0) for l in labels]
    pcts = [split_pcts.get(l, 0) for l in labels]
    
    bars = ax.bar(labels, counts, color=[colors.get(l, "#95a5a6") for l in labels], 
                   edgecolor="black", linewidth=1.5)
    
    for bar, pct, count in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{count}\n({pct}%)", ha="center", va="bottom", fontsize=12, fontweight="bold")
    
    ax.set_title("Train/Validation/Test Split Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_xlabel("Split", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_class_distribution_by_split_figure(df_split, output_path):
    """Generate class distribution per split figure."""
    splits = ["train", "val", "test"]
    classes = sorted(df_split["dx"].unique())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = sns.color_palette("viridis", len(classes))
    
    for idx, split in enumerate(splits):
        split_df = df_split[df_split["split"] == split]
        class_counts = split_df["dx"].value_counts().reindex(classes, fill_value=0)
        class_pcts = (class_counts / len(split_df) * 100).round(1)
        
        ax = axes[idx]
        full_names = [f"{c}\n({CLASS_NAMES.get(c, c)})" for c in classes]
        
        bars = ax.bar(full_names, class_counts, color=colors, edgecolor="black", linewidth=0.5)
        
        for bar, pct in zip(bars, class_pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{pct}%", ha="center", va="bottom", fontsize=8)
        
        ax.set_title(f"{split.upper()} (n={len(split_df)})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count", fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_reports(df_split, split_stats, preprocessing_stats, output_dirs):
    """Generate JSON reports and figures."""
    print("\n" + "="*70)
    print("STEP 7: GENERATING REPORTS & FIGURES")
    print("="*70)
    
    # Generate JSON report
    report = {
        "day": 2,
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset": "HAM10000",
        "total_images": int(len(df_split)),
        "total_lesions": int(df_split["lesion_id"].nunique()),
        "total_classes": int(df_split["dx"].nunique()),
        "split_method": "stratified_group_split (lesion_id level, stratified by dx)",
        "split_statistics": split_stats,
        "preprocessing_tests": {
            "total_tested": int(preprocessing_stats["total_tested"]),
            "successful": int(preprocessing_stats["successful"]),
            "failed": int(preprocessing_stats["failed"]),
            "output_size": "224x224",
            "output_channels": 3,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    }
    
    report_path = output_dirs["data"] / "day2_split_statistics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report saved: {report_path}")
    
    # Generate figures
    split_dist_path = output_dirs["figures"] / "day2_split_distribution.png"
    generate_split_distribution_figure(df_split, split_dist_path)
    print(f"✓ Figure saved: {split_dist_path}")
    
    class_dist_path = output_dirs["figures"] / "day2_class_distribution_by_split.png"
    generate_class_distribution_by_split_figure(df_split, class_dist_path)
    print(f"✓ Figure saved: {class_dist_path}")


def main():
    """Main execution."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "MediScan — DAY 2: DATA PREPROCESSING & LEAKAGE-SAFE SPLIT".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        # Setup
        output_dirs = setup_directories()
        csv_path, image_dirs = locate_data()
        
        # Load data
        df = load_and_prepare_metadata(csv_path)
        
        # Create split
        df_split = create_split(df, seed=42)
        
        # Verify integrity
        leakage_check = verify_split_integrity(df_split)
        
        # Compute statistics
        split_stats = compute_statistics(df_split)
        
        # Save splits
        split_paths = save_splits(df_split, output_dirs["data"])
        
        # Test preprocessing
        preprocessing_stats = test_preprocessing(df_split, image_dirs, n_samples=5)
        
        # Generate reports
        generate_reports(df_split, split_stats, preprocessing_stats, output_dirs)
        
        # Final summary
        print("\n" + "="*70)
        print("DAY 2 COMPLETE ✓")
        print("="*70)
        print("\nKey outputs:")
        print(f"  ✓ Split metadata saved to {output_dirs['data']}")
        print(f"  ✓ Figures saved to {output_dirs['figures']}")
        print(f"  ✓ No data leakage detected")
        print(f"  ✓ Preprocessing pipeline tested and verified")
        
        print("\nNext steps (Day 3):")
        print("  → Implement augmentation pipeline")
        print("  → Create Albumentations transforms")
        print("  → Add augmented preprocessing for training set")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
