"""
MediScan — Dataset Inspection & Preprocessing Utilities (Day 1)

This module provides functions for inspecting and verifying the HAM10000
dataset. It does NOT implement the training preprocessing pipeline
(augmentation, normalization, DataLoader) — those belong to Day 2+.

Functions:
    load_metadata        — Load and return the HAM10000 metadata CSV
    analyze_class_distribution — Compute class counts and percentages
    check_image_integrity — Verify images are readable and inspect properties
    analyze_duplicates   — Analyze lesion_id groupings for leakage prevention
    analyze_metadata_quality — Inspect missing values and field distributions
    find_image_files     — Locate all image files in a directory
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load HAM10000 metadata CSV.

    Args:
        csv_path: Path to the metadata CSV file.

    Returns:
        pd.DataFrame: Loaded metadata.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def analyze_class_distribution(df: pd.DataFrame, class_column: str = "dx") -> dict:
    """
    Compute class distribution from metadata.

    Args:
        df: Metadata DataFrame.
        class_column: Column name containing class labels.

    Returns:
        dict with:
            - class_counts: dict of {class_label: count}
            - class_percentages: dict of {class_label: percentage}
            - total: total number of samples
            - num_classes: number of unique classes
            - class_labels: sorted list of class labels
    """
    if class_column not in df.columns:
        raise ValueError(f"Column '{class_column}' not found in DataFrame. "
                         f"Available columns: {list(df.columns)}")

    counts = df[class_column].value_counts()
    total = len(df)

    return {
        "class_counts": counts.to_dict(),
        "class_percentages": {k: round(v / total * 100, 2) for k, v in counts.items()},
        "total": total,
        "num_classes": len(counts),
        "class_labels": sorted(counts.index.tolist()),
    }


def find_image_files(image_dir: str, extensions: tuple = (".jpg", ".jpeg", ".png")) -> list:
    """
    Find all image files in a directory (non-recursive).

    Args:
        image_dir: Path to directory containing images.
        extensions: Tuple of valid file extensions (case-insensitive).

    Returns:
        List of image file paths (as Path objects).
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    files = []
    for f in image_dir.iterdir():
        if f.is_file() and f.suffix.lower() in extensions:
            files.append(f)
    return sorted(files)


def check_image_integrity(
    image_dir: str,
    image_ids: Optional[list] = None,
    sample_size: Optional[int] = None,
    extensions: tuple = (".jpg", ".jpeg", ".png"),
) -> dict:
    """
    Verify image files are readable and inspect their properties.

    Args:
        image_dir: Path to directory containing images.
        image_ids: Optional list of expected image IDs (without extension).
                   If provided, checks for missing/orphan images.
        sample_size: If set, only inspect this many images (randomly sampled).
                     If None, inspect all images.
        extensions: Valid file extensions.

    Returns:
        dict with:
            - total_files: number of image files found
            - readable: number of successfully readable images
            - corrupted: list of unreadable file paths
            - dimensions: dict of {(width, height): count} for sampled images
            - formats: dict of {format_string: count}
            - modes: dict of {mode_string: count}
            - missing_from_dir: image IDs in metadata but not found as files
            - orphan_files: files in directory but not in metadata
    """
    image_dir = Path(image_dir)
    all_files = find_image_files(image_dir, extensions)

    # Build a mapping: stem -> path
    file_stems = {f.stem: f for f in all_files}

    result = {
        "total_files": len(all_files),
        "readable": 0,
        "corrupted": [],
        "dimensions": {},
        "formats": {},
        "modes": {},
        "missing_from_dir": [],
        "orphan_files": [],
    }

    # Check for missing / orphan images
    if image_ids is not None:
        metadata_ids = set(str(img_id) for img_id in image_ids)
        file_ids = set(file_stems.keys())
        result["missing_from_dir"] = sorted(metadata_ids - file_ids)
        result["orphan_files"] = sorted(file_ids - metadata_ids)

    # Determine which files to inspect
    files_to_check = all_files
    if sample_size is not None and sample_size < len(all_files):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_files), size=sample_size, replace=False)
        files_to_check = [all_files[i] for i in sorted(indices)]


    # Inspect each file
    for filepath in tqdm(files_to_check, desc="Checking image integrity", disable=len(files_to_check) < 50):
        try:
            with Image.open(filepath) as img:
                img.verify()  # Verify image is not corrupted

            # Re-open after verify (verify closes the file)
            with Image.open(filepath) as img:
                w, h = img.size
                fmt = img.format or "UNKNOWN"
                mode = img.mode

                dim_key = f"{w}x{h}"
                result["dimensions"][dim_key] = result["dimensions"].get(dim_key, 0) + 1
                result["formats"][fmt] = result["formats"].get(fmt, 0) + 1
                result["modes"][mode] = result["modes"].get(mode, 0) + 1
                result["readable"] += 1

        except Exception as e:
            result["corrupted"].append({
                "file": str(filepath),
                "error": str(e),
            })

    result["inspected_count"] = len(files_to_check)
    return result


def analyze_duplicates(df: pd.DataFrame, lesion_col: str = "lesion_id",
                       image_col: str = "image_id") -> dict:
    """
    Analyze lesion_id groupings to understand data leakage risk.

    Multiple images may belong to the same lesion. If these end up in
    different splits (train/val/test), the model could memorize lesion-specific
    features, leading to inflated test metrics.

    Args:
        df: Metadata DataFrame.
        lesion_col: Column name for lesion identifiers.
        image_col: Column name for image identifiers.

    Returns:
        dict with:
            - unique_lesions: number of unique lesion IDs
            - total_images: total number of images
            - images_per_lesion: descriptive statistics
            - distribution: counts of how many lesions have N images
            - leakage_risk: summary of the data leakage concern
    """
    if lesion_col not in df.columns:
        return {
            "error": f"Column '{lesion_col}' not found in DataFrame",
            "available_columns": list(df.columns),
        }

    # Count images per lesion
    lesion_counts = df.groupby(lesion_col)[image_col].count()

    # Distribution: how many lesions have 1 image, 2 images, etc.
    count_distribution = lesion_counts.value_counts().sort_index()

    stats = {
        "unique_lesions": int(lesion_counts.shape[0]),
        "total_images": int(len(df)),
        "images_per_lesion": {
            "mean": round(float(lesion_counts.mean()), 2),
            "median": float(lesion_counts.median()),
            "min": int(lesion_counts.min()),
            "max": int(lesion_counts.max()),
            "std": round(float(lesion_counts.std()), 2),
        },
        "distribution": {int(k): int(v) for k, v in count_distribution.items()},
        "multi_image_lesions": int((lesion_counts > 1).sum()),
        "single_image_lesions": int((lesion_counts == 1).sum()),
        "leakage_risk": (
            f"There are {int((lesion_counts > 1).sum())} lesions with multiple images. "
            f"The train/val/test split MUST be performed at the lesion_id level "
            f"(not image_id level) to prevent data leakage."
        ),
    }

    return stats


def analyze_metadata_quality(df: pd.DataFrame) -> dict:
    """
    Inspect metadata completeness and field distributions.

    Args:
        df: Metadata DataFrame.

    Returns:
        dict with:
            - columns: list of column names
            - dtypes: dict of column data types
            - missing_values: dict of {column: missing_count}
            - missing_percentages: dict of {column: missing_percentage}
            - field_summaries: dict of distributions for categorical/numeric fields
    """
    result = {
        "columns": list(df.columns),
        "row_count": len(df),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": {},
        "missing_percentages": {},
        "field_summaries": {},
    }

    for col in df.columns:
        missing = int(df[col].isna().sum())
        result["missing_values"][col] = missing
        result["missing_percentages"][col] = round(missing / len(df) * 100, 2)

    # Summarize individual fields
    for col in df.columns:
        if df[col].dtype == "object" or df[col].nunique() < 30:
            # Categorical or low-cardinality: value counts
            vc = df[col].value_counts(dropna=False)
            result["field_summaries"][col] = {
                "unique_values": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in vc.head(15).items()},
            }
        elif np.issubdtype(df[col].dtype, np.number):
            # Numeric: descriptive statistics
            desc = df[col].describe()
            result["field_summaries"][col] = {
                "mean": round(float(desc["mean"]), 2),
                "std": round(float(desc["std"]), 2),
                "min": float(desc["min"]),
                "25%": float(desc["25%"]),
                "50%": float(desc["50%"]),
                "75%": float(desc["75%"]),
                "max": float(desc["max"]),
                "unique_values": int(df[col].nunique()),
            }

    return result


def find_metadata_csv(data_dir: str) -> Optional[str]:
    """
    Search for the HAM10000 metadata CSV in the data directory.

    Searches recursively for files matching common metadata filenames.

    Args:
        data_dir: Root data directory to search.

    Returns:
        Path to the metadata CSV file, or None if not found.
    """
    data_dir = Path(data_dir)
    candidates = [
        "HAM10000_metadata.csv",
        "HAM10000_metadata",
        "hmnist_28_28_RGB.csv",
    ]

    # Search data_dir and subdirectories
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename in candidates or filename.lower() == "ham10000_metadata.csv":
                return str(Path(root) / filename)
            # Also check for CSV files containing "metadata" in name
            if "metadata" in filename.lower() and filename.endswith(".csv"):
                return str(Path(root) / filename)

    return None


def find_image_directories(data_dir: str) -> list:
    """
    Search for directories containing HAM10000 images.

    HAM10000 images may be split across multiple subdirectories
    (e.g., HAM10000_images_part_1, HAM10000_images_part_2).

    Args:
        data_dir: Root data directory to search.

    Returns:
        List of directory paths containing image files.
    """
    data_dir = Path(data_dir)
    image_dirs = []

    for root, dirs, files in os.walk(data_dir):
        # Check if this directory contains image files
        image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(image_files) > 10:  # Likely an image directory
            image_dirs.append(str(Path(root)))

    return sorted(image_dirs)


# ═════════════════════════════════════════════════════════════════════
# DAY 2: DATA SPLITTING & PREPROCESSING
# ═════════════════════════════════════════════════════════════════════


def create_stratified_group_split(
    df: pd.DataFrame,
    group_column: str = "lesion_id",
    stratify_column: str = "dx",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a leakage-safe train/validation/test split grouped by lesion_id.

    All images belonging to the same lesion_id remain in exactly one split.
    Stratification attempts to preserve class distribution across splits.

    Args:
        df: Metadata DataFrame with at least group_column and stratify_column.
        group_column: Column name for grouping (e.g., "lesion_id").
        stratify_column: Column name for stratification (e.g., "dx" for diagnosis).
        train_ratio: Proportion for train set (0-1).
        val_ratio: Proportion for validation set (0-1).
        test_ratio: Proportion for test set (0-1).
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with original data plus a "split" column ("train", "val", or "test").

    Process:
        1. Group by lesion_id
        2. Assign the most common class label to each group
        3. Perform stratified split at the group level (if possible)
        4. Fall back to random split for very small datasets
        5. Assign all images in each group to the same split
    """
    rng = np.random.RandomState(random_state)

    # Normalize ratios to ensure they sum to 1
    total = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / total
    val_ratio = val_ratio / total
    test_ratio = test_ratio / total

    # Group by lesion_id and get the most common class for stratification
    group_df = df.groupby(group_column).agg({
        stratify_column: lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'image_id': 'count',  # Count images per lesion
    }).reset_index()
    group_df.rename(columns={'image_id': 'image_count'}, inplace=True)

    # Stratified split at group level
    from sklearn.model_selection import train_test_split

    # Check if stratification is possible
    # (need at least 2 groups per class)
    class_counts = group_df[stratify_column].value_counts()
    min_class_count = class_counts.min()
    can_stratify = min_class_count >= 2

    if can_stratify:
        # Stratified split
        try:
            # First split: train + (val+test)
            train_groups, temp_groups = train_test_split(
                group_df,
                test_size=(1.0 - train_ratio),
                stratify=group_df[stratify_column],
                random_state=random_state,
            )

            # Second split: val and test
            # Adjust val_ratio for the remaining data
            adjusted_val_ratio = val_ratio / (val_ratio + test_ratio)
            val_groups, test_groups = train_test_split(
                temp_groups,
                test_size=(1.0 - adjusted_val_ratio),
                stratify=temp_groups[stratify_column],
                random_state=random_state,
            )
        except ValueError:
            # If stratification fails, fall back to random split
            can_stratify = False

    if not can_stratify:
        # Random split (not stratified)
        n_groups = len(group_df)
        n_train = int(n_groups * train_ratio)
        n_val = int(n_groups * val_ratio)
        
        indices = rng.permutation(n_groups)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_groups = group_df.iloc[train_indices]
        val_groups = group_df.iloc[val_indices]
        test_groups = group_df.iloc[test_indices]

    # Create a mapping from group_column value to split
    split_map = {}
    for group_id in train_groups[group_column].values:
        split_map[group_id] = "train"
    for group_id in val_groups[group_column].values:
        split_map[group_id] = "val"
    for group_id in test_groups[group_column].values:
        split_map[group_id] = "test"

    # Assign split to all images
    df_split = df.copy()
    df_split["split"] = df_split[group_column].map(split_map)

    # Verify all rows got a split
    if df_split["split"].isna().any():
        raise ValueError("Some rows did not get assigned to a split!")

    return df_split


def preprocess_image(
    image_path: str,
    output_size: int = 224,
    mean: list = None,
    std: list = None,
) -> Optional[np.ndarray]:
    """
    Apply deterministic preprocessing to an image.

    Steps:
        1. Load image as PIL Image
        2. Convert to RGB (if needed)
        3. Resize to output_size x output_size
        4. Normalize using provided mean and std

    Args:
        image_path: Path to image file.
        output_size: Target size (e.g., 224 for 224x224).
        mean: List of mean values for normalization (default: ImageNet).
        std: List of std values for normalization (default: ImageNet).

    Returns:
        Normalized image as numpy array (float32, shape [output_size, output_size, 3]).
        Returns None if the image cannot be loaded.

    Note:
        Output shape is (output_size, output_size, 3) with values in range [0, 1] before
        normalization, then shifted by mean and scaled by 1/std.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
    if std is None:
        std = [0.229, 0.224, 0.225]  # ImageNet std

    try:
        # Load image
        img = Image.open(image_path)

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize
        img = img.resize((output_size, output_size), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Normalize using mean and std
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        img_array = (img_array - mean) / std

        return img_array

    except Exception as e:
        return None


def verify_no_leakage(
    df_split: pd.DataFrame,
    group_column: str = "lesion_id",
    image_column: str = "image_id",
) -> dict:
    """
    Verify that no data leakage exists in the split.

    Checks:
        1. No lesion_id appears in multiple splits
        2. No image_id appears in multiple splits
        3. All images are accounted for
        4. No NaN values in split column

    Args:
        df_split: DataFrame with "split" column (train, val, test).
        group_column: Column for group-level overlap check (e.g., lesion_id).
        image_column: Column for image-level overlap check (e.g., image_id).

    Returns:
        dict with:
            - success: bool indicating no leakage detected
            - errors: list of error messages (empty if success)
            - stats: dict with verification statistics
    """
    errors = []
    stats = {}

    # Check for NaN splits
    nan_count = df_split["split"].isna().sum()
    if nan_count > 0:
        errors.append(f"Found {nan_count} rows with missing split assignment")

    # Check split column values
    valid_splits = {"train", "val", "test"}
    invalid_splits = set(df_split["split"].unique()) - valid_splits
    if invalid_splits:
        errors.append(f"Found invalid split values: {invalid_splits}")

    # Get unique values per split
    train_groups = set(df_split[df_split["split"] == "train"][group_column].unique())
    val_groups = set(df_split[df_split["split"] == "val"][group_column].unique())
    test_groups = set(df_split[df_split["split"] == "test"][group_column].unique())

    train_images = set(df_split[df_split["split"] == "train"][image_column].unique())
    val_images = set(df_split[df_split["split"] == "val"][image_column].unique())
    test_images = set(df_split[df_split["split"] == "test"][image_column].unique())

    # Check for lesion_id leakage
    train_val_overlap = train_groups & val_groups
    train_test_overlap = train_groups & test_groups
    val_test_overlap = val_groups & test_groups

    if train_val_overlap:
        errors.append(f"Lesion leakage between train and val: {len(train_val_overlap)} lesions overlap")
    if train_test_overlap:
        errors.append(f"Lesion leakage between train and test: {len(train_test_overlap)} lesions overlap")
    if val_test_overlap:
        errors.append(f"Lesion leakage between val and test: {len(val_test_overlap)} lesions overlap")

    # Check for image_id leakage
    train_val_img_overlap = train_images & val_images
    train_test_img_overlap = train_images & test_images
    val_test_img_overlap = val_images & test_images

    if train_val_img_overlap:
        errors.append(f"Image leakage between train and val: {len(train_val_img_overlap)} images overlap")
    if train_test_img_overlap:
        errors.append(f"Image leakage between train and test: {len(train_test_img_overlap)} images overlap")
    if val_test_img_overlap:
        errors.append(f"Image leakage between val and test: {len(val_test_img_overlap)} images overlap")

    # Check total image count
    total_images_unique = len(train_images | val_images | test_images)
    total_images_in_df = len(df_split[image_column].unique())
    stats["total_images_in_df"] = int(total_images_in_df)
    stats["total_unique_images_in_splits"] = int(total_images_unique)
    if total_images_unique != total_images_in_df:
        errors.append(
            f"Image count mismatch: {total_images_in_df} in DataFrame, "
            f"{total_images_unique} in splits"
        )

    # Populate stats
    stats["train_lesions"] = int(len(train_groups))
    stats["val_lesions"] = int(len(val_groups))
    stats["test_lesions"] = int(len(test_groups))
    stats["train_images"] = int(len(train_images))
    stats["val_images"] = int(len(val_images))
    stats["test_images"] = int(len(test_images))

    return {
        "success": len(errors) == 0,
        "errors": errors,
        "stats": stats,
    }


def compute_split_statistics(
    df_split: pd.DataFrame,
    class_column: str = "dx",
) -> dict:
    """
    Compute detailed statistics for each split.

    Args:
        df_split: DataFrame with "split" column.
        class_column: Column containing class labels.

    Returns:
        dict with statistics for train, val, test including:
            - image_count
            - lesion_count
            - percentage of total
            - class distribution
    """
    total_images = len(df_split)
    total_lesions = df_split["lesion_id"].nunique()

    stats = {}

    for split_name in ["train", "val", "test"]:
        split_df = df_split[df_split["split"] == split_name]

        image_count = len(split_df)
        lesion_count = split_df["lesion_id"].nunique()
        pct_images = round(image_count / total_images * 100, 2)
        pct_lesions = round(lesion_count / total_lesions * 100, 2)

        # Class distribution in this split
        class_dist = split_df[class_column].value_counts().to_dict()
        class_pct = {}
        for cls, count in class_dist.items():
            class_pct[cls] = round(count / image_count * 100, 2)

        stats[split_name] = {
            "image_count": int(image_count),
            "lesion_count": int(lesion_count),
            "percentage_images": pct_images,
            "percentage_lesions": pct_lesions,
            "class_distribution": {str(k): int(v) for k, v in sorted(class_dist.items())},
            "class_percentages": {str(k): pct for k, pct in sorted(class_pct.items())},
        }

    return stats


def save_split_metadata(
    df_split: pd.DataFrame,
    output_dir: str,
    prefix: str = "split",
) -> dict:
    """
    Save split metadata to CSV files.

    Creates:
        - {prefix}_all.csv (all images with split assignments)
        - {prefix}_train.csv (train split only)
        - {prefix}_val.csv (validation split only)
        - {prefix}_test.csv (test split only)

    Args:
        df_split: DataFrame with "split" column.
        output_dir: Directory to save CSV files.
        prefix: Prefix for output filenames.

    Returns:
        dict with paths to created files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save all splits in one file
    all_path = output_dir / f"{prefix}_all.csv"
    df_split.to_csv(all_path, index=False)
    paths["all"] = str(all_path)

    # Save individual splits
    for split_name in ["train", "val", "test"]:
        split_df = df_split[df_split["split"] == split_name].copy()
        split_path = output_dir / f"{prefix}_{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        paths[split_name] = str(split_path)

    return paths

    # ═════════════════════════════════════════════════════════════════════
    # DAY 3: DATA AUGMENTATION & PYTORCH DATASET
    # ═════════════════════════════════════════════════════════════════════


    def load_config(config_path: str = "config/config.yaml") -> dict:
        """Load YAML configuration file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config


    def get_class_label_mapping() -> dict:
        """Create deterministic class label mapping."""
        class_labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        return {label: idx for idx, label in enumerate(class_labels)}


    def get_inverse_class_label_mapping() -> dict:
        """Create inverse mapping from integer index to class label."""
        mapping = get_class_label_mapping()
        return {v: k for k, v in mapping.items()}


    def get_augmentation_transform(config: dict, mode: str = "train") -> A.Compose:
        """Create Albumentations transform pipeline."""
        aug_config = config.get("augmentation", {})
        img_config = config.get("image", {})
    
        image_size = img_config.get("size", 224)
        mean = img_config.get("normalize_mean", [0.485, 0.456, 0.406])
        std = img_config.get("normalize_std", [0.229, 0.224, 0.225])
    
        if mode == "train":
            transforms = A.Compose([
                A.Resize(height=image_size, width=image_size, interpolation=1),
                A.HorizontalFlip(p=aug_config.get("horizontal_flip_p", 0.5)),
                A.VerticalFlip(p=aug_config.get("vertical_flip_p", 0.5)),
                A.Rotate(limit=aug_config.get("rotate_limit", 30), p=aug_config.get("rotate90_p", 0.5), border_mode=1),
                A.ShiftScaleRotate(shift_limit=aug_config.get("shift_limit", 0.1), scale_limit=aug_config.get("scale_limit", 0.15), rotate_limit=aug_config.get("rotate_limit", 30), p=0.5, border_mode=1),
                A.RandomBrightnessContrast(brightness_limit=aug_config.get("brightness_limit", 0.2), contrast_limit=aug_config.get("contrast_limit", 0.2), p=0.5),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ])
        else:
            transforms = A.Compose([
                A.Resize(height=image_size, width=image_size, interpolation=1),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ])
    
        return transforms


    class HAM10000Dataset(Dataset):
        """PyTorch Dataset for HAM10000 skin cancer classification."""
    
        def __init__(
            self,
            split_csv_path: str,
            image_base_dir: str = "data/raw",
            transform = None,
            include_metadata: bool = False,
        ):
            self.split_csv_path = Path(split_csv_path)
            self.image_base_dir = Path(image_base_dir)
            self.transform = transform
            self.include_metadata = include_metadata
        
            if not self.split_csv_path.exists():
                raise FileNotFoundError(f"Split CSV not found: {self.split_csv_path}")
        
            self.df = pd.read_csv(self.split_csv_path)
            self.class_label_mapping = get_class_label_mapping()
        
            if not self.image_base_dir.exists():
                raise FileNotFoundError(f"Image base directory not found: {self.image_base_dir}")
        
            self.image_dirs = sorted([d for d in self.image_base_dir.iterdir() if d.is_dir() and "HAM10000_images" in d.name])
        
            if not self.image_dirs:
                raise FileNotFoundError(f"No HAM10000_images* directories found in {self.image_base_dir}")
    
        def __len__(self) -> int:
            return len(self.df)
    
        def _find_image_file(self, image_id: str) -> Path:
            for img_dir in self.image_dirs:
                for ext in [".jpg", ".jpeg", ".png"]:
                    img_path = img_dir / f"{image_id}{ext}"
                    if img_path.exists():
                        return img_path
            raise FileNotFoundError(f"Image not found for ID: {image_id}")
    
        def __getitem__(self, idx: int):
            row = self.df.iloc[idx]
            image_id = row["image_id"]
        
            try:
                img_path = self._find_image_file(image_id)
                image = Image.open(img_path).convert("RGB")
                image = np.array(image)
            except Exception as e:
                raise RuntimeError(f"Failed to load image {image_id}: {str(e)}")
        
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
        
            class_label = row["dx"]
            label_int = self.class_label_mapping[class_label]
        
            if self.include_metadata:
                metadata = {"image_id": str(row["image_id"]), "lesion_id": str(row["lesion_id"]), "dx": str(row["dx"]), "index": int(idx)}
                return image, label_int, metadata
            else:
                return image, label_int


    def verify_augmentation_pipeline(
        split_csv_path: str,
        image_base_dir: str = "data/raw",
        config_path: str = "config/config.yaml",
        sample_size: int = 5,
        mode: str = "train",
    ) -> dict:
        """Verify augmentation pipeline works correctly."""
        errors = []
        stats = {"shapes": [], "dtypes": [], "has_nan": [], "has_inf": [], "label_ranges": []}
    
        try:
            config = load_config(config_path)
            transform = get_augmentation_transform(config, mode=mode)
            dataset = HAM10000Dataset(split_csv_path=split_csv_path, image_base_dir=image_base_dir, transform=transform, include_metadata=False)
        
            n_samples = min(sample_size, len(dataset))
            rng = np.random.RandomState(42)
            indices = rng.choice(len(dataset), size=n_samples, replace=False)
        
            for idx in indices:
                try:
                    image, label = dataset[idx]
                    if not isinstance(image, torch.Tensor):
                        errors.append(f"Sample {idx}: image is not torch.Tensor")
                    if image.shape != (3, 224, 224):
                        errors.append(f"Sample {idx}: shape is {image.shape}, expected (3, 224, 224)")
                    stats["shapes"].append(tuple(image.shape))
                    stats["dtypes"].append(str(image.dtype))
                    if torch.isnan(image).any():
                        errors.append(f"Sample {idx}: contains NaN values")
                        stats["has_nan"].append(True)
                    else:
                        stats["has_nan"].append(False)
                    if torch.isinf(image).any():
                        errors.append(f"Sample {idx}: contains Inf values")
                        stats["has_inf"].append(True)
                    else:
                        stats["has_inf"].append(False)
                    if not (0 <= label <= 6):
                        errors.append(f"Sample {idx}: label {label} outside range [0, 6]")
                    stats["label_ranges"].append(int(label))
                except Exception as e:
                    errors.append(f"Sample {idx}: {str(e)}")
    
        except Exception as e:
            errors.append(f"Pipeline setup failed: {str(e)}")
    
        return {"success": len(errors) == 0, "errors": errors, "stats": stats, "samples_checked": len(stats["shapes"])}


    def visualize_augmentations(
        split_csv_path: str,
        image_base_dir: str = "data/raw",
        config_path: str = "config/config.yaml",
        output_dir: str = "reports/figures",
        num_examples: int = 3,
    ) -> dict:
        """Generate augmentation visualization showing original + 6 variations."""
        import matplotlib.pyplot as plt
    
        errors = []
        output_files = []
    
        try:
            config = load_config(config_path)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
            dataset = HAM10000Dataset(split_csv_path=split_csv_path, image_base_dir=image_base_dir, transform=None, include_metadata=True)
        
            n_samples = min(num_examples, len(dataset))
            rng = np.random.RandomState(42)
            indices = rng.choice(len(dataset), size=n_samples, replace=False)
        
            for idx in indices:
                try:
                    image, label, metadata = dataset[idx]
                    image_id = metadata["image_id"]
                    dx = metadata["dx"]
                
                    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                    fig.suptitle(f"Augmentation Examples: {image_id} ({dx})", fontsize=14, fontweight="bold")
                
                    axes[0, 0].imshow(image)
                    axes[0, 0].set_title("Original")
                    axes[0, 0].axis("off")
                
                    aug = A.Compose([A.HorizontalFlip(p=1.0)])
                    aug_img = aug(image=image)["image"]
                    axes[0, 1].imshow(aug_img)
                    axes[0, 1].set_title("HorizontalFlip")
                    axes[0, 1].axis("off")
                
                    aug = A.Compose([A.VerticalFlip(p=1.0)])
                    aug_img = aug(image=image)["image"]
                    axes[0, 2].imshow(aug_img)
                    axes[0, 2].set_title("VerticalFlip")
                    axes[0, 2].axis("off")
                
                    aug = A.Compose([A.Rotate(limit=30, p=1.0, border_mode=1)])
                    aug_img = aug(image=image)["image"]
                    axes[0, 3].imshow(aug_img)
                    axes[0, 3].set_title("Rotate(±30°)")
                    axes[0, 3].axis("off")
                
                    aug = A.Compose([A.RandomBrightnessContrast(p=1.0)])
                    aug_img = aug(image=image)["image"]
                    axes[1, 0].imshow(aug_img)
                    axes[1, 0].set_title("Brightness/Contrast")
                    axes[1, 0].axis("off")
                
                    aug = A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(limit=15, p=0.5, border_mode=1), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)])
                    aug_img = aug(image=image)["image"]
                    axes[1, 1].imshow(aug_img)
                    axes[1, 1].set_title("Mild Combined")
                    axes[1, 1].axis("off")
                
                    aug = A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=30, p=1.0, border_mode=1), A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)])
                    aug_img = aug(image=image)["image"]
                    axes[1, 2].imshow(aug_img)
                    axes[1, 2].set_title("Aggressive Combined")
                    axes[1, 2].axis("off")
                
                    axes[1, 3].axis("off")
                
                    plt.tight_layout()
                
                    output_path = output_dir / f"day3_augmentation_examples_{image_id}.png"
                    plt.savefig(output_path, dpi=100, bbox_inches="tight")
                    output_files.append(str(output_path))
                    plt.close()
                
                except Exception as e:
                    errors.append(f"Failed to visualize {idx}: {str(e)}")
        
        except Exception as e:
            errors.append(f"Visualization setup failed: {str(e)}")
    
        return {"success": len(errors) == 0, "errors": errors, "num_images_generated": len(output_files), "output_files": output_files}
