"""
MediScan — Day 4: DataLoader Pipeline Tests

Comprehensive test suite covering:
    - Dataset lengths (train=6982, val=1521, test=1512)
    - DataLoader creation for all three splits
    - Batch shape [B, 3, 224, 224]
    - Label type (integer) and range [0, 6]
    - Correct class mapping across all datasets
    - Image path resolution for every split image
    - Image loading (no NaN/Inf)
    - Train shuffle=True, val/test shuffle=False
    - Dataset accounting (total == 10,015)
    - DataLoader configuration from config.yaml
    - No leakage / split membership unchanged

Requires the real HAM10000 dataset and Day 2 split CSVs on disk.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing_day3 import (
    HAM10000Dataset,
    get_augmentation_transform,
    get_class_label_mapping,
    get_inverse_class_label_mapping,
    load_config,
)
from src.data.dataloader import (
    create_dataloaders,
    verify_image_paths,
    verify_batch,
    verify_dataset_accounting,
    verify_class_labels,
    benchmark_dataloader,
)


# ═════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def config():
    """Load project configuration."""
    return load_config(str(PROJECT_ROOT / "config" / "config.yaml"))


@pytest.fixture(scope="module")
def dataloaders():
    """Create all three DataLoaders once per test module."""
    return create_dataloaders(
        config_path=str(PROJECT_ROOT / "config" / "config.yaml"),
        project_root=str(PROJECT_ROOT),
    )


@pytest.fixture(scope="module")
def split_csvs():
    """Load all three split CSVs."""
    splits_dir = PROJECT_ROOT / "reports" / "data"
    return {
        "train": pd.read_csv(splits_dir / "split_train.csv"),
        "val": pd.read_csv(splits_dir / "split_val.csv"),
        "test": pd.read_csv(splits_dir / "split_test.csv"),
    }


# ═════════════════════════════════════════════════════════════════════
# TESTS: DATASET LENGTH & ACCOUNTING
# ═════════════════════════════════════════════════════════════════════


class TestDatasetAccounting:
    """Verify dataset sizes match expected split counts."""

    def test_train_dataset_length(self, dataloaders):
        assert len(dataloaders["train"].dataset) == 6982, (
            f"Train dataset: expected 6982, got {len(dataloaders['train'].dataset)}"
        )

    def test_val_dataset_length(self, dataloaders):
        assert len(dataloaders["val"].dataset) == 1521, (
            f"Val dataset: expected 1521, got {len(dataloaders['val'].dataset)}"
        )

    def test_test_dataset_length(self, dataloaders):
        assert len(dataloaders["test"].dataset) == 1512, (
            f"Test dataset: expected 1512, got {len(dataloaders['test'].dataset)}"
        )

    def test_total_samples(self, dataloaders):
        total = sum(len(dataloaders[s].dataset) for s in ("train", "val", "test"))
        assert total == 10015, f"Total samples: expected 10015, got {total}"

    def test_verify_dataset_accounting_utility(self, dataloaders):
        result = verify_dataset_accounting(dataloaders)
        assert result["success"], f"Accounting errors: {result['errors']}"


# ═════════════════════════════════════════════════════════════════════
# TESTS: DATALOADER CREATION
# ═════════════════════════════════════════════════════════════════════


class TestDataLoaderCreation:
    """Verify DataLoader objects are constructed correctly."""

    def test_loaders_contain_three_splits(self, dataloaders):
        assert set(dataloaders.keys()) == {"train", "val", "test"}

    def test_loaders_are_dataloader_instances(self, dataloaders):
        for name, loader in dataloaders.items():
            assert isinstance(loader, DataLoader), f"{name} is not a DataLoader"

    def test_train_batch_size_from_config(self, dataloaders, config):
        expected_bs = config.get("dataloader", {}).get("batch_size", 32)
        assert dataloaders["train"].batch_size == expected_bs

    def test_val_batch_size_from_config(self, dataloaders, config):
        expected_bs = config.get("dataloader", {}).get("batch_size", 32)
        assert dataloaders["val"].batch_size == expected_bs

    def test_test_batch_size_from_config(self, dataloaders, config):
        expected_bs = config.get("dataloader", {}).get("batch_size", 32)
        assert dataloaders["test"].batch_size == expected_bs


# ═════════════════════════════════════════════════════════════════════
# TESTS: BATCH SHAPE & VALUES
# ═════════════════════════════════════════════════════════════════════


class TestBatchVerification:
    """Verify batch shapes, dtypes, and label ranges."""

    def _get_batch(self, dataloader):
        return next(iter(dataloader))

    def test_train_batch_shape(self, dataloaders):
        images, labels = self._get_batch(dataloaders["train"])
        assert images.dim() == 4, f"Expected 4-D tensor, got {images.dim()}-D"
        assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
        assert images.shape[2] == 224, f"Expected H=224, got {images.shape[2]}"
        assert images.shape[3] == 224, f"Expected W=224, got {images.shape[3]}"

    def test_val_batch_shape(self, dataloaders):
        images, labels = self._get_batch(dataloaders["val"])
        assert images.shape[1:] == (3, 224, 224), f"Val batch shape: {images.shape}"

    def test_test_batch_shape(self, dataloaders):
        images, labels = self._get_batch(dataloaders["test"])
        assert images.shape[1:] == (3, 224, 224), f"Test batch shape: {images.shape}"

    def test_train_labels_type(self, dataloaders):
        _, labels = self._get_batch(dataloaders["train"])
        assert labels.dtype in (torch.int64, torch.int32, torch.long), (
            f"Label dtype: {labels.dtype}"
        )

    def test_train_labels_range(self, dataloaders):
        _, labels = self._get_batch(dataloaders["train"])
        assert labels.min() >= 0, f"Min label: {labels.min()}"
        assert labels.max() <= 6, f"Max label: {labels.max()}"

    def test_val_labels_range(self, dataloaders):
        _, labels = self._get_batch(dataloaders["val"])
        assert labels.min() >= 0 and labels.max() <= 6

    def test_test_labels_range(self, dataloaders):
        _, labels = self._get_batch(dataloaders["test"])
        assert labels.min() >= 0 and labels.max() <= 6

    def test_train_no_nan(self, dataloaders):
        images, _ = self._get_batch(dataloaders["train"])
        assert not torch.isnan(images).any(), "Train batch contains NaN"

    def test_train_no_inf(self, dataloaders):
        images, _ = self._get_batch(dataloaders["train"])
        assert not torch.isinf(images).any(), "Train batch contains Inf"

    def test_val_no_nan_inf(self, dataloaders):
        images, _ = self._get_batch(dataloaders["val"])
        assert not torch.isnan(images).any(), "Val batch contains NaN"
        assert not torch.isinf(images).any(), "Val batch contains Inf"

    def test_test_no_nan_inf(self, dataloaders):
        images, _ = self._get_batch(dataloaders["test"])
        assert not torch.isnan(images).any(), "Test batch contains NaN"
        assert not torch.isinf(images).any(), "Test batch contains Inf"

    def test_verify_batch_utility_train(self, dataloaders):
        result = verify_batch(dataloaders["train"], split_name="train")
        assert result["success"], f"Train batch verification errors: {result['errors']}"

    def test_verify_batch_utility_val(self, dataloaders):
        result = verify_batch(dataloaders["val"], split_name="val")
        assert result["success"], f"Val batch verification errors: {result['errors']}"

    def test_verify_batch_utility_test(self, dataloaders):
        result = verify_batch(dataloaders["test"], split_name="test")
        assert result["success"], f"Test batch verification errors: {result['errors']}"


# ═════════════════════════════════════════════════════════════════════
# TESTS: CLASS LABEL MAPPING
# ═════════════════════════════════════════════════════════════════════


class TestClassLabelMapping:
    """Verify the shared class-label mapping is consistent."""

    EXPECTED_MAPPING = {
        "akiec": 0,
        "bcc": 1,
        "bkl": 2,
        "df": 3,
        "mel": 4,
        "nv": 5,
        "vasc": 6,
    }

    def test_canonical_mapping(self):
        mapping = get_class_label_mapping()
        assert mapping == self.EXPECTED_MAPPING

    def test_all_datasets_use_same_mapping(self, dataloaders):
        for name, loader in dataloaders.items():
            ds = loader.dataset
            assert ds.class_label_mapping == self.EXPECTED_MAPPING, (
                f"{name} dataset mapping mismatch"
            )

    def test_verify_class_labels_utility(self, dataloaders):
        result = verify_class_labels(dataloaders)
        assert result["success"], f"Class label errors: {result['errors']}"

    def test_no_unexpected_labels_in_csvs(self, split_csvs):
        expected_classes = set(self.EXPECTED_MAPPING.keys())
        for name, df in split_csvs.items():
            actual = set(df["dx"].unique())
            unexpected = actual - expected_classes
            assert not unexpected, (
                f"[{name}] Unexpected class labels: {unexpected}"
            )


# ═════════════════════════════════════════════════════════════════════
# TESTS: IMAGE PATH RESOLUTION
# ═════════════════════════════════════════════════════════════════════


class TestImagePathResolution:
    """Verify every image in split CSVs can be found on disk."""

    def test_all_split_images_exist(self):
        result = verify_image_paths(project_root=str(PROJECT_ROOT))
        assert result["success"], (
            f"Missing images: {result['missing'][:10]} "
            f"(total missing: {len(result['missing'])})"
        )

    def test_image_count_on_disk(self):
        result = verify_image_paths(project_root=str(PROJECT_ROOT))
        # At least 10015 images should be on disk (could be more)
        assert result["total_available_on_disk"] >= 10015, (
            f"Only {result['total_available_on_disk']} images on disk"
        )

    def test_image_loading_samples(self, dataloaders):
        """Load a few random samples from each split to confirm images load."""
        rng = np.random.RandomState(42)
        for name, loader in dataloaders.items():
            ds = loader.dataset
            indices = rng.choice(len(ds), size=min(5, len(ds)), replace=False)
            for idx in indices:
                image, label = ds[int(idx)]
                assert isinstance(image, torch.Tensor), (
                    f"[{name}] Sample {idx} did not return a tensor"
                )
                assert image.shape == (3, 224, 224), (
                    f"[{name}] Sample {idx} shape: {image.shape}"
                )


# ═════════════════════════════════════════════════════════════════════
# TESTS: SHUFFLE BEHAVIOUR
# ═════════════════════════════════════════════════════════════════════


class TestShuffleBehaviour:
    """Verify train shuffles and val/test do not."""

    def test_train_shuffle_enabled(self, dataloaders):
        sampler = dataloaders["train"].sampler
        assert isinstance(sampler, torch.utils.data.sampler.RandomSampler), (
            f"Train sampler is {type(sampler).__name__}, expected RandomSampler"
        )

    def test_val_shuffle_disabled(self, dataloaders):
        sampler = dataloaders["val"].sampler
        assert isinstance(sampler, torch.utils.data.sampler.SequentialSampler), (
            f"Val sampler is {type(sampler).__name__}, expected SequentialSampler"
        )

    def test_test_shuffle_disabled(self, dataloaders):
        sampler = dataloaders["test"].sampler
        assert isinstance(sampler, torch.utils.data.sampler.SequentialSampler), (
            f"Test sampler is {type(sampler).__name__}, expected SequentialSampler"
        )


# ═════════════════════════════════════════════════════════════════════
# TESTS: SPLIT INTEGRITY (no leakage, no membership changes)
# ═════════════════════════════════════════════════════════════════════


class TestSplitIntegrity:
    """Confirm that Day 4 has not altered split membership."""

    def test_split_csvs_exist(self):
        splits_dir = PROJECT_ROOT / "reports" / "data"
        for name in ("train", "val", "test"):
            assert (splits_dir / f"split_{name}.csv").exists()

    def test_no_image_overlap_between_splits(self, split_csvs):
        train_ids = set(split_csvs["train"]["image_id"])
        val_ids = set(split_csvs["val"]["image_id"])
        test_ids = set(split_csvs["test"]["image_id"])

        assert not (train_ids & val_ids), "Train/val image overlap"
        assert not (train_ids & test_ids), "Train/test image overlap"
        assert not (val_ids & test_ids), "Val/test image overlap"

    def test_no_lesion_overlap_between_splits(self, split_csvs):
        train_lesions = set(split_csvs["train"]["lesion_id"])
        val_lesions = set(split_csvs["val"]["lesion_id"])
        test_lesions = set(split_csvs["test"]["lesion_id"])

        assert not (train_lesions & val_lesions), "Train/val lesion overlap"
        assert not (train_lesions & test_lesions), "Train/test lesion overlap"
        assert not (val_lesions & test_lesions), "Val/test lesion overlap"

    def test_all_csv_rows_have_split_column(self, split_csvs):
        for name, df in split_csvs.items():
            assert "split" in df.columns, f"[{name}] Missing split column"
            assert df["split"].isna().sum() == 0, f"[{name}] NaN splits"

    def test_dataset_does_not_resplit(self, dataloaders, split_csvs):
        """Confirm the Dataset reads the CSV as-is without creating a new split."""
        for name, loader in dataloaders.items():
            ds_len = len(loader.dataset)
            csv_len = len(split_csvs[name])
            assert ds_len == csv_len, (
                f"[{name}] Dataset len ({ds_len}) != CSV len ({csv_len}). "
                f"The Dataset may be creating a new split."
            )


# ═════════════════════════════════════════════════════════════════════
# TESTS: CONFIGURATION
# ═════════════════════════════════════════════════════════════════════


class TestConfiguration:
    """Verify DataLoader settings come from config."""

    def test_config_has_dataloader_section(self, config):
        assert "dataloader" in config, "config.yaml missing 'dataloader' section"

    def test_config_batch_size(self, config):
        assert "batch_size" in config["dataloader"]
        assert isinstance(config["dataloader"]["batch_size"], int)
        assert config["dataloader"]["batch_size"] > 0

    def test_config_num_workers(self, config):
        assert "num_workers" in config["dataloader"]
        assert isinstance(config["dataloader"]["num_workers"], int)
        assert config["dataloader"]["num_workers"] >= 0

    def test_config_pin_memory(self, config):
        assert "pin_memory" in config["dataloader"]
        assert isinstance(config["dataloader"]["pin_memory"], bool)

    def test_config_drop_last(self, config):
        assert "drop_last" in config["dataloader"]
        assert isinstance(config["dataloader"]["drop_last"], bool)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
