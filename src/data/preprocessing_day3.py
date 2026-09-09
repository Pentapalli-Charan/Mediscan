"""Day 3 augmentation and HAM10000 Dataset utilities."""

from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_class_label_mapping():
    return {label: index for index, label in enumerate(
        ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    )}


def get_inverse_class_label_mapping():
    return {index: label for label, index in get_class_label_mapping().items()}


def get_augmentation_transform(config, mode="train"):
    image_config = config.get("image", {})
    augmentation = config.get("augmentation", {})
    size = image_config.get("size", 224)
    mean = image_config.get("normalize_mean", [0.485, 0.456, 0.406])
    std = image_config.get("normalize_std", [0.229, 0.224, 0.225])
    transforms = []
    if mode == "train":
        transforms.extend([
            A.HorizontalFlip(p=augmentation.get("horizontal_flip_p", 0.5)),
            A.VerticalFlip(p=augmentation.get("vertical_flip_p", 0.5)),
            A.Rotate(limit=augmentation.get("rotate_limit", 30), p=augmentation.get("rotate90_p", 0.5), border_mode=1),
            A.ShiftScaleRotate(shift_limit=augmentation.get("shift_limit", 0.1), scale_limit=augmentation.get("scale_limit", 0.15), rotate_limit=augmentation.get("rotate_limit", 30), p=augmentation.get("shift_scale_rotate_p", 0.5), border_mode=1),
            A.RandomBrightnessContrast(brightness_limit=augmentation.get("brightness_limit", 0.2), contrast_limit=augmentation.get("contrast_limit", 0.2), p=augmentation.get("brightness_contrast_p", 0.5)),
        ])
    transforms.extend([
        A.Resize(size, size, interpolation=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return A.Compose(transforms)


class HAM10000Dataset(Dataset):
    def __init__(self, split_csv_path, image_base_dir="data/raw", transform=None, include_metadata=False):
        self.df = pd.read_csv(split_csv_path)
        self.image_base_dir = Path(image_base_dir)
        self.transform = transform
        self.include_metadata = include_metadata
        self.class_label_mapping = get_class_label_mapping()
        self.image_dirs = sorted(path for path in self.image_base_dir.iterdir() if path.is_dir() and "HAM10000_images" in path.name)
        if not self.image_dirs:
            raise FileNotFoundError(f"No HAM10000 image directories found in {self.image_base_dir}")

    def __len__(self):
        return len(self.df)

    def _find_image_file(self, image_id):
        for directory in self.image_dirs:
            for extension in (".jpg", ".jpeg", ".png"):
                path = directory / f"{image_id}{extension}"
                if path.exists():
                    return path
        raise FileNotFoundError(f"Image not found: {image_id}")

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = np.array(Image.open(self._find_image_file(row["image_id"])).convert("RGB"))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        label = self.class_label_mapping[row["dx"]]
        if not self.include_metadata:
            return image, label
        metadata = {"image_id": str(row["image_id"]), "lesion_id": str(row["lesion_id"]), "dx": str(row["dx"]), "index": index}
        return image, label, metadata


def verify_augmentation_pipeline(split_csv_path, image_base_dir="data/raw", config_path="config/config.yaml", sample_size=5, mode="train"):
    errors = []
    stats = {"shapes": [], "dtypes": [], "has_nan": [], "has_inf": [], "label_ranges": []}
    try:
        dataset = HAM10000Dataset(split_csv_path, image_base_dir, get_augmentation_transform(load_config(config_path), mode), False)
        indices = np.random.RandomState(42).choice(len(dataset), size=min(sample_size, len(dataset)), replace=False)
        for index in indices:
            image, label = dataset[index]
            stats["shapes"].append(tuple(image.shape))
            stats["dtypes"].append(str(image.dtype))
            stats["has_nan"].append(bool(torch.isnan(image).any()))
            stats["has_inf"].append(bool(torch.isinf(image).any()))
            stats["label_ranges"].append(int(label))
            if not isinstance(image, torch.Tensor) or tuple(image.shape) != (3, 224, 224):
                errors.append(f"Sample {index}: invalid image output")
            if stats["has_nan"][-1] or stats["has_inf"][-1] or not 0 <= label <= 6:
                errors.append(f"Sample {index}: invalid values or label")
    except Exception as error:
        errors.append(str(error))
    return {"success": not errors, "errors": errors, "stats": stats, "samples_checked": len(stats["shapes"])}


def visualize_augmentations(split_csv_path, image_base_dir="data/raw", config_path="config/config.yaml", output_dir="reports/figures", num_examples=3):
    import matplotlib.pyplot as plt
    errors = []
    output_files = []
    try:
        dataset = HAM10000Dataset(split_csv_path, image_base_dir, None, True)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Prefer different diagnostic classes so visual evidence is not dominated by nv.
        representative_indices = []
        for class_label in get_class_label_mapping():
            matches = dataset.df.index[dataset.df["dx"] == class_label].tolist()
            if matches:
                representative_indices.append(matches[0])
        indices = representative_indices[:num_examples]
        for index in indices:
            image, _, metadata = dataset[index]
            transforms = [
                ("Original", lambda x: x),
                ("HorizontalFlip", lambda x: A.HorizontalFlip(p=1)(image=x)["image"]),
                ("VerticalFlip", lambda x: A.VerticalFlip(p=1)(image=x)["image"]),
                ("Rotate", lambda x: A.Rotate(limit=30, p=1, border_mode=1)(image=x)["image"]),
                ("BrightnessContrast", lambda x: A.RandomBrightnessContrast(p=1)(image=x)["image"]),
                ("CombinedMild", lambda x: A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5, border_mode=1),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                ])(image=x)["image"]),
            ]
            figure, axes = plt.subplots(1, len(transforms), figsize=(16, 4))
            for axis, (title, transform) in zip(axes, transforms):
                axis.imshow(transform(image))
                axis.set_title(title)
                axis.axis("off")
            figure.tight_layout()
            file_path = output_path / f"day3_augmentation_examples_{metadata['image_id']}.png"
            figure.savefig(file_path, dpi=100)
            plt.close(figure)
            output_files.append(str(file_path))
    except Exception as error:
        errors.append(str(error))
    return {"success": not errors, "errors": errors, "num_images_generated": len(output_files), "output_files": output_files}
