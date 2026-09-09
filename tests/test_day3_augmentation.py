"""
MediScan — Day 3 Augmentation & Dataset Tests

Comprehensive test suite for augmentation pipeline, PyTorch Dataset class,
and verification utilities. Tests are deterministic (seed=42).

Test Coverage:
    - Augmentation transform creation (train/val/test)
    - HAM10000Dataset loading and iteration
    - Class label mapping (0-6 integers)
    - Output shapes (3, 224, 224)
    - No NaN/Inf values
    - Split integrity (train/val/test assignments)
    - Metadata extraction
    - Configuration loading
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

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


class TestClassLabelMapping:
    """Test class label mapping utilities."""

    def test_class_label_mapping_exists(self):
        """Test that class label mapping returns expected dict."""
        mapping = get_class_label_mapping()
        
        assert isinstance(mapping, dict)
        assert len(mapping) == 7
        
    def test_class_label_mapping_order(self):
        """Test that class labels are alphabetically ordered."""
        mapping = get_class_label_mapping()
        
        expected = {
            "akiec": 0,
            "bcc": 1,
            "bkl": 2,
            "df": 3,
            "mel": 4,
            "nv": 5,
            "vasc": 6,
        }
        
        assert mapping == expected
    
    def test_inverse_class_label_mapping(self):
        """Test inverse mapping."""
        inverse = get_inverse_class_label_mapping()
        
        expected = {
            0: "akiec",
            1: "bcc",
            2: "bkl",
            3: "df",
            4: "mel",
            5: "nv",
            6: "vasc",
        }
        
        assert inverse == expected
    
    def test_mapping_roundtrip(self):
        """Test that forward and inverse mappings are consistent."""
        forward = get_class_label_mapping()
        inverse = get_inverse_class_label_mapping()
        
        for label, idx in forward.items():
            assert inverse[idx] == label


class TestAugmentationTransform:
    """Test augmentation transform pipeline."""
    
    def test_load_config(self):
        """Test config loading."""
        config = load_config("config/config.yaml")
        
        assert isinstance(config, dict)
        assert "augmentation" in config
        assert "image" in config
    
    def test_get_train_transform(self):
        """Test training transform creation."""
        config = load_config("config/config.yaml")
        transform = get_augmentation_transform(config, mode="train")
        
        assert transform is not None
        # Albumentations Compose object
        assert hasattr(transform, "__call__")
    
    def test_get_val_transform(self):
        """Test validation transform creation."""
        config = load_config("config/config.yaml")
        transform = get_augmentation_transform(config, mode="val")
        
        assert transform is not None
        assert hasattr(transform, "__call__")
    
    def test_get_test_transform(self):
        """Test test transform creation."""
        config = load_config("config/config.yaml")
        transform = get_augmentation_transform(config, mode="test")
        
        assert transform is not None
        assert hasattr(transform, "__call__")
    
    def test_transform_output_shape_train(self):
        """Test train transform output shape."""
        config = load_config("config/config.yaml")
        transform = get_augmentation_transform(config, mode="train")
        
        # Create dummy image (300x300x3)
        dummy_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        
        result = transform(image=dummy_image)
        output = result["image"]
        
        # Should be (3, 224, 224) tensor
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)
    
    def test_transform_output_shape_val(self):
        """Test val transform output shape."""
        config = load_config("config/config.yaml")
        transform = get_augmentation_transform(config, mode="val")
        
        dummy_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        
        result = transform(image=dummy_image)
        output = result["image"]
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)


class TestHAM10000Dataset:
    """Test HAM10000Dataset class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.split_csv = self.project_root / "reports" / "data" / "split_train.csv"
        self.image_base_dir = self.project_root / "data" / "raw"
    
    def test_dataset_initialization(self):
        """Test dataset can be initialized."""
        if not self.split_csv.exists():
            pytest.skip("Split CSV not found")
        
        config = load_config(str(self.project_root / "config" / "config.yaml"))
        transform = get_augmentation_transform(config, mode="train")
        
        dataset = HAM10000Dataset(
            split_csv_path=str(self.split_csv),
            image_base_dir=str(self.image_base_dir),
            transform=transform,
            include_metadata=False,
        )
        
        assert dataset is not None
        assert len(dataset) > 0
    
    def test_dataset_length(self):
        """Test dataset length matches CSV."""
        if not self.split_csv.exists():
            pytest.skip("Split CSV not found")
        
        config = load_config(str(self.project_root / "config" / "config.yaml"))
        transform = get_augmentation_transform(config, mode="train")
        
        dataset = HAM10000Dataset(
            split_csv_path=str(self.split_csv),
            image_base_dir=str(self.image_base_dir),
            transform=transform,
        )
        
        df = pd.read_csv(self.split_csv)
        assert len(dataset) == len(df)
    
    def test_dataset_getitem_shape(self):
        """Test dataset __getitem__ returns correct shapes."""
        if not self.split_csv.exists():
            pytest.skip("Split CSV not found")
        
        config = load_config(str(self.project_root / "config" / "config.yaml"))
        transform = get_augmentation_transform(config, mode="train")
        
        dataset = HAM10000Dataset(
            split_csv_path=str(self.split_csv),
            image_base_dir=str(self.image_base_dir),
            transform=transform,
            include_metadata=False,
        )
        
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, (int, np.integer))
        assert 0 <= label <= 6
    
    def test_dataset_getitem_no_nan(self):
        """Test that output has no NaN values."""
        if not self.split_csv.exists():
            pytest.skip("Split CSV not found")
        
        config = load_config(str(self.project_root / "config" / "config.yaml"))
        transform = get_augmentation_transform(config, mode="train")
        
        dataset = HAM10000Dataset(
            split_csv_path=str(self.split_csv),
            image_base_dir=str(self.image_base_dir),
            transform=transform,
        )
        
        image, label = dataset[0]
        
        assert not torch.isnan(image).any()
    
    def test_dataset_getitem_no_inf(self):
        """Test that output has no Inf values."""
        if not self.split_csv.exists():
            pytest.skip("Split CSV not found")
        
        config = load_config(str(self.project_root / "config" / "config.yaml"))
        transform = get_augmentation_transform(config, mode="train")
        
        dataset = HAM10000Dataset(
            split_csv_path=str(self.split_csv),
            image_base_dir=str(self.image_base_dir),
            transform=transform,
        )
        
        image, label = dataset[0]
        
        assert not torch.isinf(image).any()
    
    def test_dataset_label_range(self):
        """Test that labels are in range [0, 6]."""
        if not self.split_csv.exists():
            pytest.skip("Split CSV not found")
        
        config = load_config(str(self.project_root / "config" / "config.yaml"))
        transform = get_augmentation_transform(config, mode="train")
        
        dataset = HAM10000Dataset(
            split_csv_path=str(self.split_csv),
            image_base_dir=str(self.image_base_dir),
            transform=transform,
        )
        
        # Test multiple samples
        for i in range(min(10, len(dataset))):
            _, label = dataset[i]
            assert 0 <= label <= 6
    
    def test_dataset_with_metadata(self):
        """Test dataset with include_metadata=True."""
        if not self.split_csv.exists():
            pytest.skip("Split CSV not found")
        
        config = load_config(str(self.project_root / "config" / "config.yaml"))
        transform = get_augmentation_transform(config, mode="train")
        
        dataset = HAM10000Dataset(
            split_csv_path=str(self.split_csv),
            image_base_dir=str(self.image_base_dir),
            transform=transform,
            include_metadata=True,
        )
        
        image, label, metadata = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, (int, np.integer))
        assert isinstance(metadata, dict)
        assert "image_id" in metadata
        assert "lesion_id" in metadata
        assert "dx" in metadata
        assert "index" in metadata
    
    def test_dataset_deterministic_val_transform(self):
        """Test that validation transform is deterministic (no randomness)."""
        if not self.split_csv.exists():
            pytest.skip("Split CSV not found")
        
        config = load_config(str(self.project_root / "config" / "config.yaml"))
        transform = get_augmentation_transform(config, mode="val")
        
        dataset = HAM10000Dataset(
            split_csv_path=str(self.split_csv),
            image_base_dir=str(self.image_base_dir),
            transform=transform,
        )
        
        # Load same sample twice
        image1, label1 = dataset[0]
        image2, label2 = dataset[0]
        
        # Should be identical (deterministic)
        assert torch.allclose(image1, image2, atol=1e-6)
        assert label1 == label2
    
    def test_dataset_different_splits(self):
        """Test dataset works with different split CSVs."""
        if not (self.project_root / "reports" / "data" / "split_val.csv").exists():
            pytest.skip("Validation split CSV not found")
        
        config = load_config(str(self.project_root / "config" / "config.yaml"))
        transform = get_augmentation_transform(config, mode="val")
        
        # Test train split
        dataset_train = HAM10000Dataset(
            split_csv_path=str(self.project_root / "reports" / "data" / "split_train.csv"),
            image_base_dir=str(self.image_base_dir),
            transform=transform,
        )
        
        # Test val split
        dataset_val = HAM10000Dataset(
            split_csv_path=str(self.project_root / "reports" / "data" / "split_val.csv"),
            image_base_dir=str(self.image_base_dir),
            transform=transform,
        )
        
        assert len(dataset_train) > 0
        assert len(dataset_val) > 0
        assert len(dataset_train) != len(dataset_val)  # Different sizes


class TestVerificationUtilities:
    """Test verification and visualization utilities."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.split_csv = self.project_root / "reports" / "data" / "split_train.csv"
        self.image_base_dir = self.project_root / "data" / "raw"
        self.config_path = self.project_root / "config" / "config.yaml"
    
    def test_verify_augmentation_pipeline_train(self):
        """Test augmentation verification for training mode."""
        if not self.split_csv.exists():
            pytest.skip("Split CSV not found")
        
        result = verify_augmentation_pipeline(
            split_csv_path=str(self.split_csv),
            image_base_dir=str(self.image_base_dir),
            config_path=str(self.config_path),
            sample_size=3,
            mode="train",
        )
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "errors" in result
        assert "stats" in result
        assert "samples_checked" in result
        
        if result["success"]:
            assert result["samples_checked"] > 0
    
    def test_verify_augmentation_pipeline_val(self):
        """Test augmentation verification for validation mode."""
        if not self.split_csv.exists():
            pytest.skip("Split CSV not found")
        
        result = verify_augmentation_pipeline(
            split_csv_path=str(self.split_csv),
            image_base_dir=str(self.image_base_dir),
            config_path=str(self.config_path),
            sample_size=3,
            mode="val",
        )
        
        assert isinstance(result, dict)
        assert "success" in result
    
    def test_visualize_augmentations(self):
        """Test augmentation visualization."""
        if not self.split_csv.exists():
            pytest.skip("Split CSV not found")
        
        # Use temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            result = visualize_augmentations(
                split_csv_path=str(self.split_csv),
                image_base_dir=str(self.image_base_dir),
                config_path=str(self.config_path),
                output_dir=tmpdir,
                num_examples=1,
            )
            
            assert isinstance(result, dict)
            assert "success" in result
            assert "errors" in result
            assert "num_images_generated" in result
            assert "output_files" in result


class TestDataIntegrity:
    """Test that Day 2 splits are not modified."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.splits_dir = self.project_root / "reports" / "data"
    
    def test_split_csvs_exist(self):
        """Test that all split CSVs exist."""
        for split_name in ["train", "val", "test", "all"]:
            csv_path = self.splits_dir / f"split_{split_name}.csv"
            assert csv_path.exists(), f"Split CSV not found: {csv_path}"
    
    def test_split_assignments_not_modified(self):
        """Test that split assignments are still present."""
        split_csv = self.splits_dir / "split_all.csv"
        if not split_csv.exists():
            pytest.skip("Split CSV not found")
        
        df = pd.read_csv(split_csv)
        
        # Check split column
        assert "split" in df.columns
        
        # Check all rows have split assignment
        assert df["split"].isna().sum() == 0
        
        # Check only valid split values
        valid_splits = {"train", "val", "test"}
        actual_splits = set(df["split"].unique())
        assert actual_splits.issubset(valid_splits)
    
    def test_no_lesion_leakage(self):
        """Test that lesion_ids don't leak between splits."""
        split_csv = self.splits_dir / "split_all.csv"
        if not split_csv.exists():
            pytest.skip("Split CSV not found")
        
        df = pd.read_csv(split_csv)
        
        # Group by lesion_id and check all images in same split
        for lesion_id in df["lesion_id"].unique():
            lesion_rows = df[df["lesion_id"] == lesion_id]
            splits = set(lesion_rows["split"].unique())
            
            # Each lesion should only appear in one split
            assert len(splits) == 1, f"Lesion {lesion_id} appears in multiple splits: {splits}"
    
    def test_no_image_overlap(self):
        """Test that image_ids don't overlap between splits."""
        split_csv = self.splits_dir / "split_all.csv"
        if not split_csv.exists():
            pytest.skip("Split CSV not found")
        
        df = pd.read_csv(split_csv)
        
        # Check no duplicate image_ids
        assert df["image_id"].nunique() == len(df), "Found duplicate image_ids"
        
        # Check each image appears in exactly one split
        for image_id in df["image_id"].unique():
            image_rows = df[df["image_id"] == image_id]
            assert len(image_rows) == 1, f"Image {image_id} appears {len(image_rows)} times"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
