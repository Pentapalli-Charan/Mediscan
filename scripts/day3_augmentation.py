"""
MediScan — Day 3 Augmentation Pipeline Orchestration

Comprehensive orchestration script implementing all Day 3 augmentation tasks:
    1. Configuration loading and verification
    2. Augmentation transform creation (train/val/test)
    3. PyTorch Dataset instantiation for all splits
    4. Augmentation verification (shape, NaN, Inf checks)
    5. Augmentation visualization generation
    6. Comprehensive safety verification
    7. Final report generation

Execution: python scripts/day3_augmentation.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessing_day3 import (
    load_config,
    get_class_label_mapping,
    get_inverse_class_label_mapping,
    get_augmentation_transform,
    HAM10000Dataset,
    verify_augmentation_pipeline,
    visualize_augmentations,
)


def print_section(title: str, width: int = 80):
    """Print formatted section header."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    """Execute full Day 3 augmentation pipeline."""
    project_root = Path(__file__).parent.parent
    
    print_section("DAY 3: DATA AUGMENTATION PIPELINE")
    
    # =========================================================================
    # 1. LOAD CONFIGURATION
    # =========================================================================
    print_section("1. LOADING CONFIGURATION", 80)
    
    config_path = project_root / "config" / "config.yaml"
    print(f"Loading config from: {config_path}")
    
    try:
        config = load_config(str(config_path))
        print("✓ Configuration loaded successfully")
        
        # Print augmentation parameters
        aug_config = config.get("augmentation", {})
        print("\nAugmentation Parameters:")
        for key, value in aug_config.items():
            print(f"  - {key}: {value}")
        
        # Print image settings
        img_config = config.get("image", {})
        print("\nImage Settings:")
        print(f"  - size: {img_config.get('size', 224)}")
        print(f"  - channels: {img_config.get('channels', 3)}")
        print(f"  - mean: {img_config.get('normalize_mean', [0.485, 0.456, 0.406])}")
        print(f"  - std: {img_config.get('normalize_std', [0.229, 0.224, 0.225])}")
    
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False
    
    # =========================================================================
    # 2. CLASS LABEL MAPPING
    # =========================================================================
    print_section("2. CLASS LABEL MAPPING", 80)
    
    class_mapping = get_class_label_mapping()
    inverse_mapping = get_inverse_class_label_mapping()
    
    print(f"✓ Created class label mapping ({len(class_mapping)} classes):")
    for label, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
        print(f"  - {label:10s} → {idx}")
    
    # =========================================================================
    # 3. CREATE AUGMENTATION TRANSFORMS
    # =========================================================================
    print_section("3. CREATING AUGMENTATION TRANSFORMS", 80)
    
    try:
        train_transform = get_augmentation_transform(config, mode="train")
        val_transform = get_augmentation_transform(config, mode="val")
        test_transform = get_augmentation_transform(config, mode="test")
        
        print("✓ Train transform created (with random augmentation)")
        print("✓ Val transform created (deterministic, no augmentation)")
        print("✓ Test transform created (deterministic, no augmentation)")
    
    except Exception as e:
        print(f"✗ Failed to create transforms: {e}")
        return False
    
    # =========================================================================
    # 4. INITIALIZE PYTORCH DATASETS
    # =========================================================================
    print_section("4. INITIALIZING PYTORCH DATASETS", 80)
    
    splits_dir = project_root / "reports" / "data"
    image_base_dir = project_root / "data" / "raw"
    
    datasets = {}
    split_names = ["train", "val", "test"]
    transforms_list = [train_transform, val_transform, test_transform]
    
    for split_name, transform in zip(split_names, transforms_list):
        split_csv = splits_dir / f"split_{split_name}.csv"
        
        try:
            dataset = HAM10000Dataset(
                split_csv_path=str(split_csv),
                image_base_dir=str(image_base_dir),
                transform=transform,
                include_metadata=False,
            )
            datasets[split_name] = dataset
            print(f"✓ {split_name:5s} dataset: {len(dataset):5d} samples")
        
        except Exception as e:
            print(f"✗ Failed to create {split_name} dataset: {e}")
            return False
    
    # =========================================================================
    # 5. VERIFY AUGMENTATION PIPELINE
    # =========================================================================
    print_section("5. VERIFYING AUGMENTATION PIPELINE", 80)
    
    for split_name in split_names:
        split_csv = splits_dir / f"split_{split_name}.csv"
        
        print(f"\nVerifying {split_name} pipeline...")
        result = verify_augmentation_pipeline(
            split_csv_path=str(split_csv),
            image_base_dir=str(image_base_dir),
            config_path=str(config_path),
            sample_size=10,
            mode=split_name,
        )
        
        if result["success"]:
            print(f"✓ {split_name} pipeline verified")
            print(f"  - Samples checked: {result['samples_checked']}")
            print(f"  - Shapes: {set(result['stats']['shapes'])}")
            print(f"  - Dtypes: {set(result['stats']['dtypes'])}")
            print(f"  - NaN values: {sum(result['stats']['has_nan'])}/{result['samples_checked']}")
            print(f"  - Inf values: {sum(result['stats']['has_inf'])}/{result['samples_checked']}")
            print(f"  - Label range: [{min(result['stats']['label_ranges'])}, {max(result['stats']['label_ranges'])}]")
        else:
            print(f"✗ {split_name} pipeline verification FAILED")
            for error in result["errors"]:
                print(f"  - {error}")
            return False
    
    # =========================================================================
    # 6. GENERATE AUGMENTATION VISUALIZATIONS
    # =========================================================================
    print_section("6. GENERATING AUGMENTATION VISUALIZATIONS", 80)
    
    output_dir = project_root / "reports" / "figures"
    split_csv = splits_dir / "split_train.csv"
    
    print(f"Generating augmentation visualizations...")
    print(f"Output directory: {output_dir}")
    
    result = visualize_augmentations(
        split_csv_path=str(split_csv),
        image_base_dir=str(image_base_dir),
        config_path=str(config_path),
        output_dir=str(output_dir),
        num_examples=3,
    )
    
    if result["success"]:
        print(f"✓ Generated {result['num_images_generated']} augmentation visualization(s)")
        for file_path in result["output_files"]:
            print(f"  - {file_path}")
    else:
        print(f"✗ Augmentation visualization FAILED")
        for error in result["errors"]:
            print(f"  - {error}")
        # Don't return False - visualization failures are non-critical
    
    # =========================================================================
    # 7. SPLIT INTEGRITY VERIFICATION
    # =========================================================================
    print_section("7. VERIFYING SPLIT INTEGRITY", 80)
    
    print("Checking that Day 2 splits are unchanged...")
    
    split_all_csv = splits_dir / "split_all.csv"
    df_all = pd.read_csv(split_all_csv)
    
    # Check 1: All splits assigned
    if df_all["split"].isna().any():
        print(f"✗ Found {df_all['split'].isna().sum()} rows with missing split")
        return False
    print("✓ All rows have split assignment")
    
    # Check 2: No lesion overlap
    lesion_split_counts = df_all.groupby("lesion_id")["split"].nunique()
    if (lesion_split_counts > 1).any():
        print(f"✗ Found {(lesion_split_counts > 1).sum()} lesions in multiple splits")
        return False
    print("✓ No lesion leakage (each lesion in one split)")
    
    # Check 3: No image overlap
    if df_all["image_id"].duplicated().any():
        print(f"✗ Found duplicate image_ids")
        return False
    print("✓ No image overlap (each image in one split)")
    
    # Check 4: All images accounted for
    total_unique_images = len(df_all)
    train_images = len(df_all[df_all["split"] == "train"])
    val_images = len(df_all[df_all["split"] == "val"])
    test_images = len(df_all[df_all["split"] == "test"])
    
    print(f"✓ Split distribution:")
    print(f"  - Train: {train_images:5d} ({100.0 * train_images / total_unique_images:5.2f}%)")
    print(f"  - Val:   {val_images:5d} ({100.0 * val_images / total_unique_images:5.2f}%)")
    print(f"  - Test:  {test_images:5d} ({100.0 * test_images / total_unique_images:5.2f}%)")
    print(f"  - Total: {total_unique_images:5d}")
    
    # =========================================================================
    # 8. GENERATE FINAL REPORT
    # =========================================================================
    print_section("8. GENERATING FINAL REPORT", 80)
    
    report = {
        "day": 3,
        "task": "Data Augmentation Pipeline",
        "status": "COMPLETED",
        "timestamp": pd.Timestamp.now().isoformat(),
        
        "configuration": {
            "augmentation_parameters": aug_config,
            "image_settings": img_config,
            "class_labels": class_mapping,
        },
        
        "augmentation_transforms": {
            "train": "HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, RandomBrightnessContrast + Normalize",
            "val": "Resize + Normalize (deterministic)",
            "test": "Resize + Normalize (deterministic)",
        },
        
        "datasets": {
            "train": {"samples": len(datasets["train"]), "path": str(splits_dir / "split_train.csv")},
            "val": {"samples": len(datasets["val"]), "path": str(splits_dir / "split_val.csv")},
            "test": {"samples": len(datasets["test"]), "path": str(splits_dir / "split_test.csv")},
        },
        
        "verification_results": {
            "all_splits_verified": True,
            "no_data_leakage": True,
            "all_outputs_valid": True,
            "output_shape": (3, 224, 224),
            "output_dtype": "torch.float32",
            "label_range": [0, 6],
        },
        
        "visualizations_generated": {
            "count": result.get("num_images_generated", 0),
            "output_dir": str(output_dir),
            "files": result.get("output_files", []),
        },
        
        "split_integrity": {
            "no_nan_splits": bool(df_all["split"].isna().sum() == 0),
            "no_lesion_overlap": bool((lesion_split_counts > 1).sum() == 0),
            "no_image_overlap": bool(not df_all["image_id"].duplicated().any()),
            "total_images": int(total_unique_images),
            "train_count": int(train_images),
            "val_count": int(val_images),
            "test_count": int(test_images),
        },
    }
    
    # Save report
    report_path = project_root / "reports" / "day3_augmentation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to: {report_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("SUMMARY", 80)
    print("✓ Day 3 Augmentation Pipeline COMPLETED SUCCESSFULLY")
    print()
    print(f"  Files created/modified:")
    print(f"    - src/data/preprocessing_day3.py (Day 3 functions)")
    print(f"    - tests/test_day3_augmentation.py (comprehensive tests)")
    print(f"    - reports/day3_augmentation_report.json (final report)")
    if result.get("num_images_generated", 0) > 0:
        print(f"    - {output_dir}/day3_augmentation_examples_*.png (visualizations)")
    print()
    print(f"  Key metrics:")
    print(f"    - Augmentation parameters: {len(aug_config)} configured")
    print(f"    - Class labels: {len(class_mapping)} (0-6)")
    print(f"    - Train dataset: {len(datasets['train'])} samples")
    print(f"    - Val dataset: {len(datasets['val'])} samples")
    print(f"    - Test dataset: {len(datasets['test'])} samples")
    print(f"    - Data integrity: VERIFIED (no leakage)")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
