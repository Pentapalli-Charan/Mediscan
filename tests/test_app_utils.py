"""
MediScan — Unit Tests for app/utils.py (Day 12 Pre-Flight Audit)

Validates:
1. Model construction, checkpoint loading, offline safety, architecture verification
2. Rejection of invalid checkpoints and non-EfficientNet models
3. Image validation (formats, sizes, corruption, empty files)
4. Deterministic preprocessing (224x224, ImageNet normalized, [1, 3, 224, 224])
5. Forward inference (7 probabilities summing to ~1.0, finite, top prediction)
6. Grad-CAM execution on model.features[8] and bitwise weight preservation
7. Safe Streamlit caching wrapper and device handling
8. Sample catalog discovery
"""

import io
from pathlib import Path
import sys
import pytest
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    NUM_CLASSES,
    SUPPORTED_EXTENSIONS,
    compute_gradcam_explanation,
    get_ham10000_sample_catalog,
    get_inference_device,
    load_cached_mediscan_model,
    load_mediscan_model,
    preprocess_image_for_inference,
    resolve_project_path,
    run_model_inference,
    validate_and_load_image,
)
from src.explainability.gradcam import snapshot_model_weights, verify_model_weights_unchanged

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@pytest.fixture(scope="module")
def shared_model():
    model, _ = load_mediscan_model(CKPT_PATH, CONFIG_PATH, device=torch.device("cpu"))
    return model


class TestModelConstructionAndCheckpoint:
    """Verify model construction, checkpoint loading, and integrity."""

    def test_model_construction_and_eval_mode(self):
        """Model must load EfficientNet-B0 with 7 classes in eval mode."""
        model, meta = load_mediscan_model(
            checkpoint_path=CKPT_PATH,
            config_path=CONFIG_PATH,
            device=torch.device("cpu"),
        )
        assert isinstance(model, nn.Module)
        assert model.training is False, "Model must be in eval mode."
        assert hasattr(model, "classifier"), "Model must have classifier head."
        assert model.classifier[1].out_features == NUM_CLASSES == 7
        assert model.classifier[1].in_features == 1280
        assert meta["architecture"] == "efficientnet_b0"
        assert meta["num_classes"] == 7
        assert meta["epoch"] == 14

    def test_all_parameters_frozen(self):
        """All model parameters must have requires_grad=False for inference."""
        model, _ = load_mediscan_model(CKPT_PATH, CONFIG_PATH, device=torch.device("cpu"))
        for name, param in model.named_parameters():
            assert not param.requires_grad, f"Parameter '{name}' should have requires_grad=False."

    def test_missing_checkpoint_raises_filenotfound(self):
        """Missing checkpoint path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
            load_mediscan_model(checkpoint_path="non_existent_path.pth", config_path=CONFIG_PATH)

    def test_missing_config_raises_filenotfound(self):
        """Missing config path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_mediscan_model(checkpoint_path=CKPT_PATH, config_path="non_existent_config.yaml")

    def test_reject_resnet50_checkpoint(self, tmp_path):
        """Checkpoint tagged as resnet50 must be rejected."""
        fake_ckpt = tmp_path / "resnet50_dummy.pth"
        torch.save({
            "model_state_dict": {},
            "architecture": "resnet50",
            "num_classes": 7,
        }, str(fake_ckpt))

        with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
            load_mediscan_model(checkpoint_path=fake_ckpt, config_path=CONFIG_PATH)

    def test_reject_corrupt_checkpoint(self, tmp_path):
        """Corrupt checkpoint without model_state_dict must be rejected."""
        corrupt_ckpt = tmp_path / "corrupt.pth"
        torch.save({"bad_key": 123}, str(corrupt_ckpt))

        with pytest.raises(ValueError, match="missing 'model_state_dict'"):
            load_mediscan_model(checkpoint_path=corrupt_ckpt, config_path=CONFIG_PATH)


class TestImageValidationAndLoading:
    """Verify robust image input validation."""

    def test_valid_pil_image(self):
        """Valid PIL Image should be converted to RGB."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        loaded = validate_and_load_image(img)
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)

    def test_valid_image_bytes(self):
        """Valid image bytes should load properly."""
        img = Image.new("RGB", (64, 64), color=(0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        loaded = validate_and_load_image(raw_bytes)
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
        assert loaded.size == (64, 64)

    def test_empty_bytes_raises_value_error(self):
        """Empty bytes (0 bytes) must raise ValueError."""
        with pytest.raises(ValueError, match="0 bytes"):
            validate_and_load_image(b"")

    def test_unsupported_file_extension_raises(self, tmp_path):
        """Unsupported file extensions must raise ValueError."""
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("not an image")
        with pytest.raises(ValueError, match="Unsupported image extension"):
            validate_and_load_image(txt_file)

    def test_corrupted_image_bytes_raises(self):
        """Corrupted image bytes must raise ValueError."""
        corrupt_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20  # Incomplete JPEG header
        with pytest.raises(ValueError, match="Failed to decode"):
            validate_and_load_image(corrupt_bytes)

    def test_nonexistent_file_path_raises(self):
        """Non-existent file path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validate_and_load_image("data/raw/non_existent_image_12345.jpg")

    def test_uploaded_file_mock(self):
        """Streamlit-style UploadedFile buffer should be parsed correctly."""
        class MockUploadedFile:
            def __init__(self, name, data):
                self.name = name
                self._data = data

            def getvalue(self):
                return self._data

        img = Image.new("RGB", (50, 50), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        mock_upload = MockUploadedFile("lesion.png", buf.getvalue())
        loaded = validate_and_load_image(mock_upload)
        assert loaded.size == (50, 50)
        assert loaded.mode == "RGB"


class TestPreprocessing:
    """Verify exact Day 2-4 deterministic inference preprocessing."""

    def test_preprocessing_output_shape_and_range(self):
        """Output tensor must have shape [1, 3, 224, 224] and resized_rgb [224, 224, 3]."""
        img = Image.new("RGB", (600, 450), color=(120, 80, 50))
        tensor, resized_rgb = preprocess_image_for_inference(img, CONFIG_PATH)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        assert torch.isfinite(tensor).all()

        assert isinstance(resized_rgb, np.ndarray)
        assert resized_rgb.shape == (224, 224, 3)
        assert resized_rgb.dtype == np.uint8

    def test_preprocessing_determinism(self):
        """Deterministic evaluation: same image produces bitwise identical tensors."""
        img = Image.new("RGB", (300, 300), color=(150, 100, 50))
        t1, r1 = preprocess_image_for_inference(img, CONFIG_PATH)
        t2, r2 = preprocess_image_for_inference(img, CONFIG_PATH)

        assert torch.equal(t1, t2), "Inference preprocessing must be strictly deterministic."
        assert np.array_equal(r1, r2)


class TestModelInference:
    """Verify forward prediction correctness and probability formatting."""

    def test_prediction_probabilities(self, shared_model):
        """Inference must produce 7 probabilities summing to ~1.0."""
        dummy_tensor = torch.randn(1, 3, 224, 224)
        results = run_model_inference(shared_model, dummy_tensor, device=torch.device("cpu"))

        assert "predicted_class" in results
        assert results["predicted_class"] in CLASS_NAMES
        assert results["predicted_class_idx"] in range(7)
        assert results["predicted_description"] == CLASS_DESCRIPTIONS[results["predicted_class"]]
        assert 0.0 <= results["confidence"] <= 1.0

        probs = results["probabilities"]
        assert len(probs) == 7
        assert set(probs.keys()) == set(CLASS_NAMES)
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities must sum to ~1.0, got {prob_sum}"

    def test_prediction_rejects_invalid_tensor(self, shared_model):
        """Inference must reject 3D or non-tensor inputs."""
        with pytest.raises(TypeError):
            run_model_inference(shared_model, np.zeros((1, 3, 224, 224)))

        with pytest.raises(ValueError, match=r"\[B, 3, H, W\]"):
            run_model_inference(shared_model, torch.randn(3, 224, 224))


class TestGradCAMExplanation:
    """Verify Grad-CAM targeting model.features[8] and weight preservation."""

    def test_gradcam_output_and_weight_immutability(self, shared_model):
        """Grad-CAM must generate 224x224 heatmaps and preserve model weights bitwise."""
        weights_before = snapshot_model_weights(shared_model)

        dummy_tensor = torch.randn(1, 3, 224, 224)
        resized_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

        cam_res = compute_gradcam_explanation(
            shared_model,
            dummy_tensor,
            resized_rgb,
            target_class_idx=4,  # mel
            alpha=0.45,
        )

        assert cam_res["heatmap"].shape == (224, 224)
        assert cam_res["heatmap_rgb"].shape == (224, 224, 3)
        assert cam_res["overlay"].shape == (224, 224, 3)
        assert cam_res["target_class_idx"] == 4
        assert cam_res["target_class_name"] == "mel"
        assert 0.0 <= cam_res["heatmap"].min() <= 1.0
        assert 0.0 <= cam_res["heatmap"].max() <= 1.0

        # Bitwise verification that weights did not change
        verify_model_weights_unchanged(weights_before, shared_model)


class TestStreamlitCachingAndCatalog:
    """Verify caching wrapper, device utility, and sample catalog."""

    def test_device_detection(self):
        """Inference device should be CPU or CUDA."""
        dev = get_inference_device()
        assert dev.type in ("cpu", "cuda")

    def test_load_cached_mediscan_model_outside_streamlit(self):
        """Cached model loader must work smoothly in bare Python without raising."""
        model, meta = load_cached_mediscan_model(
            checkpoint_path=CKPT_PATH,
            config_path=CONFIG_PATH,
            device=torch.device("cpu"),
        )
        assert isinstance(model, nn.Module)
        assert meta["architecture"] == "efficientnet_b0"

    def test_ham10000_sample_catalog(self):
        """Sample catalog should discover existing HAM10000 images if raw data exists."""
        meta_path = PROJECT_ROOT / "data" / "raw" / "HAM10000_metadata.csv"
        if meta_path.exists():
            catalog = get_ham10000_sample_catalog(
                metadata_csv=meta_path,
                samples_per_class=1,
            )
            assert len(catalog) > 0
            for item in catalog:
                assert "image_id" in item
                assert "dx" in item
                assert "path" in item
                assert Path(item["path"]).exists()
        else:
            pytest.skip("HAM10000 raw metadata not present in data/raw.")
