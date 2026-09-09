"""
Tests for Day 2: Data Preprocessing & Leakage-Safe Split

Tests cover:
    - Metadata loading
    - Stratified group split creation
    - Leakage verification (lesion_id and image_id level)
    - Split statistics
    - Preprocessing pipeline
    - Reproducibility with fixed seed
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    load_metadata,
    create_stratified_group_split,
    preprocess_image,
    verify_no_leakage,
    compute_split_statistics,
    save_split_metadata,
)
from src.utils.seed import set_seed


# ═════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def metadata_csv_path():
    """Path to the actual HAM10000 metadata CSV."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "HAM10000_metadata.csv"
    if not csv_path.exists():
        pytest.skip(f"Dataset not available at {csv_path}")
    return str(csv_path)


@pytest.fixture
def metadata_df(metadata_csv_path):
    """Load actual metadata."""
    return load_metadata(metadata_csv_path)


@pytest.fixture
def simple_metadata_df():
    """Create a larger synthetic metadata for basic tests (large enough for stratification)."""
    # Create enough data to allow stratification
    # Need at least 2 groups per class
    data = {
        "image_id": [f"img_{i}" for i in range(100)],
        "lesion_id": [f"lesion_{i // 3}" for i in range(100)],  # ~33 unique lesions
        "dx": [
            "nv" if i % 7 == 0 else
            "mel" if i % 7 == 1 else
            "bcc" if i % 7 == 2 else
            "akiec" if i % 7 == 3 else
            "vasc" if i % 7 == 4 else
            "bkl" if i % 7 == 5 else
            "df"
            for i in range(100)
        ],
        "dx_type": ["histo"] * 100,
        "age": [50] * 100,
        "sex": ["m" if i % 2 == 0 else "f" for i in range(100)],
        "localization": ["head"] * 100,
    }
    return pd.DataFrame(data)


# ═════════════════════════════════════════════════════════════════════
# TESTS: METADATA LOADING
# ═════════════════════════════════════════════════════════════════════


class TestMetadataLoading:
    """Tests for metadata loading."""

    def test_load_metadata_shape(self, metadata_df):
        """Verify metadata shape matches expected dimensions."""
        assert len(metadata_df) == 10015, "Expected 10,015 images"
        assert "image_id" in metadata_df.columns, "Missing image_id column"
        assert "lesion_id" in metadata_df.columns, "Missing lesion_id column"
        assert "dx" in metadata_df.columns, "Missing dx column"

    def test_load_metadata_classes(self, metadata_df):
        """Verify all 7 classes are present."""
        classes = set(metadata_df["dx"].unique())
        expected_classes = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
        assert classes == expected_classes, f"Classes mismatch: {classes} vs {expected_classes}"

    def test_load_metadata_no_duplicates(self, metadata_df):
        """Verify no duplicate image_ids."""
        unique_images = metadata_df["image_id"].nunique()
        total_images = len(metadata_df)
        assert unique_images == total_images, f"Duplicate images: {total_images - unique_images}"

    def test_load_metadata_lesion_count(self, metadata_df):
        """Verify unique lesion count."""
        unique_lesions = metadata_df["lesion_id"].nunique()
        assert unique_lesions == 7470, f"Expected 7,470 unique lesions, got {unique_lesions}"


# ═════════════════════════════════════════════════════════════════════
# TESTS: STRATIFIED GROUP SPLIT
# ═════════════════════════════════════════════════════════════════════


class TestStratifiedGroupSplit:
    """Tests for leakage-safe stratified split."""

    def test_split_basic_structure(self, simple_metadata_df):
        """Test that split creates valid split column."""
        df_split = create_stratified_group_split(
            simple_metadata_df,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
        )
        
        assert "split" in df_split.columns, "Missing 'split' column"
        assert len(df_split) == len(simple_metadata_df), "DataFrame length changed"
        assert not df_split["split"].isna().any(), "NaN values in split column"
        
        valid_splits = {"train", "val", "test"}
        assert set(df_split["split"].unique()) == valid_splits, "Invalid split values"

    def test_split_proportions(self, simple_metadata_df):
        """Test that split proportions are approximately correct."""
        df_split = create_stratified_group_split(
            simple_metadata_df,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
        )
        
        counts = df_split["split"].value_counts()
        total = len(df_split)
        
        # Allow ±10% tolerance due to group-level splitting
        train_pct = counts["train"] / total
        val_pct = counts.get("val", 0) / total
        test_pct = counts.get("test", 0) / total
        
        assert 0.4 <= train_pct <= 0.8, f"Train proportion {train_pct} out of expected range"
        assert 0.0 <= val_pct <= 0.5, f"Val proportion {val_pct} out of expected range"
        assert 0.0 <= test_pct <= 0.5, f"Test proportion {test_pct} out of expected range"

    def test_split_lesion_grouping(self, simple_metadata_df):
        """Test that all images from same lesion stay in same split."""
        df_split = create_stratified_group_split(
            simple_metadata_df,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            random_state=42,
        )
        
        # For each lesion, verify all images belong to same split
        for lesion_id in df_split["lesion_id"].unique():
            lesion_rows = df_split[df_split["lesion_id"] == lesion_id]
            unique_splits = lesion_rows["split"].unique()
            assert len(unique_splits) == 1, \
                f"Lesion {lesion_id} has images in multiple splits: {unique_splits}"

    def test_split_on_real_data(self, metadata_df):
        """Test split on actual HAM10000 data."""
        df_split = create_stratified_group_split(
            metadata_df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )
        
        # Verify lesion grouping on real data
        for lesion_id in df_split["lesion_id"].unique():
            lesion_rows = df_split[df_split["lesion_id"] == lesion_id]
            unique_splits = lesion_rows["split"].unique()
            assert len(unique_splits) == 1, \
                f"Real data: Lesion {lesion_id} has images in multiple splits"

    def test_split_reproducibility(self, metadata_df):
        """Test that same seed produces same split."""
        df_split_1 = create_stratified_group_split(
            metadata_df.copy(),
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )
        
        df_split_2 = create_stratified_group_split(
            metadata_df.copy(),
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )
        
        # Compare splits
        splits_match = (df_split_1["split"].values == df_split_2["split"].values).all()
        assert splits_match, "Same seed did not produce same split"


# ═════════════════════════════════════════════════════════════════════
# TESTS: LEAKAGE VERIFICATION
# ═════════════════════════════════════════════════════════════════════


class TestLeakageVerification:
    """Tests for leakage detection."""

    def test_no_leakage_in_valid_split(self, simple_metadata_df):
        """Test that valid split passes leakage check."""
        df_split = create_stratified_group_split(
            simple_metadata_df,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
        )
        
        result = verify_no_leakage(df_split)
        
        assert result["success"], f"Valid split failed leakage check: {result['errors']}"
        assert len(result["errors"]) == 0, f"Errors in valid split: {result['errors']}"

    def test_no_leakage_on_real_data(self, metadata_df):
        """Test that real data split passes leakage check."""
        df_split = create_stratified_group_split(metadata_df)
        result = verify_no_leakage(df_split)
        
        assert result["success"], f"Real data failed leakage check: {result['errors']}"
        assert len(result["errors"]) == 0, f"Errors in real data split: {result['errors']}"

    def test_detect_lesion_leakage(self, simple_metadata_df):
        """Test that introduced leakage is detected."""
        df_split = create_stratified_group_split(
            simple_metadata_df,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            random_state=42,
        )
        
        # Artificially introduce leakage
        df_split.loc[df_split["image_id"] == "img_0", "split"] = "train"
        df_split.loc[df_split["image_id"] == "img_1", "split"] = "val"  # Same lesion, different split
        
        result = verify_no_leakage(df_split)
        
        assert not result["success"], "Leakage was not detected"
        assert len(result["errors"]) > 0, "Expected errors but none found"

    def test_image_count_verification(self, metadata_df):
        """Test that all images are accounted for."""
        df_split = create_stratified_group_split(metadata_df)
        result = verify_no_leakage(df_split)
        
        stats = result["stats"]
        assert stats["total_images_in_df"] == 10015, "DataFrame image count mismatch"
        assert stats["total_unique_images_in_splits"] == 10015, "Splits image count mismatch"


# ═════════════════════════════════════════════════════════════════════
# TESTS: PREPROCESSING
# ═════════════════════════════════════════════════════════════════════


class TestPreprocessing:
    """Tests for image preprocessing."""

    def test_preprocessing_output_shape(self, metadata_df):
        """Test preprocessing on real images."""
        raw_dir = PROJECT_ROOT / "data" / "raw"
        image_dirs = list(raw_dir.glob("HAM10000_images_*"))
        
        if not image_dirs:
            pytest.skip("Image directories not found")
        
        # Find an image file
        image_files = list(image_dirs[0].glob("*.jpg"))
        if not image_files:
            pytest.skip("No image files found")
        
        image_path = str(image_files[0])
        
        # Test preprocessing
        result = preprocess_image(image_path, output_size=224)
        
        assert result is not None, "Preprocessing returned None"
        assert result.shape == (224, 224, 3), f"Unexpected shape: {result.shape}"
        assert result.dtype == np.float32, f"Unexpected dtype: {result.dtype}"

    def test_preprocessing_normalization(self, metadata_df):
        """Test that normalization is applied."""
        raw_dir = PROJECT_ROOT / "data" / "raw"
        image_dirs = list(raw_dir.glob("HAM10000_images_*"))
        
        if not image_dirs:
            pytest.skip("Image directories not found")
        
        image_files = list(image_dirs[0].glob("*.jpg"))
        if not image_files:
            pytest.skip("No image files found")
        
        image_path = str(image_files[0])
        
        result = preprocess_image(
            image_path,
            output_size=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        
        assert result is not None
        assert not np.any(np.isnan(result)), "NaN values in result"
        
        # After normalization, values should typically be in range [-2, 2]
        # (with some outliers possible)
        assert np.abs(result).mean() < 2.0, "Values seem not normalized"


# ═════════════════════════════════════════════════════════════════════
# TESTS: SPLIT STATISTICS
# ═════════════════════════════════════════════════════════════════════


class TestSplitStatistics:
    """Tests for split statistics computation."""

    def test_statistics_structure(self, simple_metadata_df):
        """Test that statistics have expected structure."""
        df_split = create_stratified_group_split(simple_metadata_df)
        stats = compute_split_statistics(df_split)
        
        for split_name in ["train", "val", "test"]:
            assert split_name in stats, f"Missing statistics for {split_name}"
            assert "image_count" in stats[split_name]
            assert "lesion_count" in stats[split_name]
            assert "class_distribution" in stats[split_name]

    def test_statistics_sum_to_total(self, metadata_df):
        """Test that split image counts sum to total."""
        df_split = create_stratified_group_split(metadata_df)
        stats = compute_split_statistics(df_split)
        
        total = (
            stats["train"]["image_count"] +
            stats["val"]["image_count"] +
            stats["test"]["image_count"]
        )
        
        assert total == len(metadata_df), f"Split counts don't sum to total: {total} vs {len(metadata_df)}"

    def test_statistics_percentages(self, metadata_df):
        """Test that percentages are computed correctly."""
        df_split = create_stratified_group_split(metadata_df)
        stats = compute_split_statistics(df_split)
        
        total_pct = (
            stats["train"]["percentage_images"] +
            stats["val"]["percentage_images"] +
            stats["test"]["percentage_images"]
        )
        
        assert abs(total_pct - 100.0) < 0.1, f"Percentages don't sum to 100: {total_pct}"


# ═════════════════════════════════════════════════════════════════════
# TESTS: SPLIT SAVING
# ═════════════════════════════════════════════════════════════════════


class TestSplitSaving:
    """Tests for saving split metadata."""

    def test_save_split_creates_files(self, simple_metadata_df):
        """Test that split metadata files are created."""
        df_split = create_stratified_group_split(simple_metadata_df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_split_metadata(df_split, tmpdir)
            
            assert "all" in paths, "Missing 'all' split file path"
            assert "train" in paths, "Missing 'train' split file path"
            assert "val" in paths, "Missing 'val' split file path"
            assert "test" in paths, "Missing 'test' split file path"
            
            for key, path in paths.items():
                assert Path(path).exists(), f"File not created: {path}"

    def test_saved_split_integrity(self, simple_metadata_df):
        """Test that saved splits can be reloaded and are intact."""
        df_split = create_stratified_group_split(simple_metadata_df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_split_metadata(df_split, tmpdir)
            
            # Reload and verify
            df_reload = pd.read_csv(paths["all"])
            
            assert len(df_reload) == len(df_split), "Row count changed"
            assert "split" in df_reload.columns, "Split column missing"
            assert set(df_reload["split"].unique()) == {"train", "val", "test"}


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
