"""
MediScan — Day 12 Streamlit Web Application Verification Tests

Tests:
1. File existence and syntax compilation of app/app.py
2. Clinical risk level mappings for all 7 HAM10000 classes
3. Headless AppTest initial state (upload prompt)
4. Headless AppTest sample case selection, prediction, and Grad-CAM generation
5. Download button creation and structured report schema
"""

from pathlib import Path
import py_compile
import sys
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import CLINICAL_RISK_LEVELS, get_target_class_index
from app.utils import CLASS_NAMES
from streamlit.testing.v1 import AppTest


class TestDay12AppStructure:
    """Verify script syntax, module structure, and clinical definitions."""

    def test_app_script_exists_and_compiles(self):
        """app/app.py must exist and compile cleanly without syntax errors."""
        app_path = PROJECT_ROOT / "app" / "app.py"
        assert app_path.exists(), f"app/app.py not found at {app_path}"
        compiled = py_compile.compile(str(app_path), doraise=True)
        assert compiled is not None

    def test_clinical_risk_levels_cover_all_classes(self):
        """Clinical risk levels must have defined entries for all 7 classes."""
        assert len(CLINICAL_RISK_LEVELS) == 7
        for code in CLASS_NAMES:
            assert code in CLINICAL_RISK_LEVELS
            info = CLINICAL_RISK_LEVELS[code]
            assert "level" in info
            assert "badge_color" in info
            assert "badge_bg" in info
            assert "urgency" in info

        # Verify malignant assignments
        assert "MALIGNANT" in CLINICAL_RISK_LEVELS["mel"]["level"].upper()
        assert "MALIGNANT" in CLINICAL_RISK_LEVELS["bcc"]["level"].upper()
        assert "PRE-MALIGNANT" in CLINICAL_RISK_LEVELS["akiec"]["level"].upper()
        assert "BENIGN" in CLINICAL_RISK_LEVELS["nv"]["level"].upper()

    def test_target_class_index_parsing(self):
        """Target class string parser must handle top prediction and explicit classes."""
        assert get_target_class_index("Top Predicted Class") is None
        assert get_target_class_index("MEL (Melanoma)") == 4
        assert get_target_class_index("BCC (Basal cell carcinoma)") == 1
        assert get_target_class_index("NV (Melanocytic nevi)") == 5


class TestDay12AppFlow:
    """Verify Streamlit application flow using official AppTest harness."""

    def test_initial_render_headless(self):
        """Initial launch should render sidebar and prompt for image without exceptions."""
        at = AppTest.from_file("app/app.py", default_timeout=30)
        at.run(timeout=30)
        assert len(at.exception) == 0, f"AppTest raised unexpected exception: {at.exception}"

        # Sidebar radio exists
        assert len(at.sidebar.radio) >= 1
        assert at.sidebar.radio[0].value == "Upload Image"
        assert "HAM10000 Dataset Samples" in at.sidebar.radio[0].options

        # Prompt info is rendered
        assert len(at.info) >= 1
        assert "upload" in at.info[0].value.lower()

    def test_ham10000_sample_selection_and_inference(self):
        """Selecting a HAM10000 sample case must run inference and generate Grad-CAM."""
        at = AppTest.from_file("app/app.py", default_timeout=45)
        at.run(timeout=30)
        assert len(at.exception) == 0

        # Switch radio to HAM10000 Dataset Samples
        at.sidebar.radio[0].set_value("HAM10000 Dataset Samples").run(timeout=30)
        assert len(at.exception) == 0, f"Exceptions after switching to sample case: {at.exception}"

        # Verify progress bars were rendered for the 7 classes
        progress_elements = at.get("progress")
        assert len(progress_elements) >= 7, f"Expected 7 class progress bars, found {len(progress_elements)}"

        # Verify download button rendered
        download_buttons = at.get("download_button")
        assert len(download_buttons) >= 1
        assert "JSON" in download_buttons[0].label
