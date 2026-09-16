"""
MediScan — Streamlit Web Application (Day 12)

Interactive dermoscopic image classification platform powered by EfficientNet-B0
with Gradient-weighted Class Activation Mapping (Grad-CAM) visual explainability.

Features:
- Live image upload (JPG, JPEG, PNG) or selection from verified HAM10000 dataset samples.
- Preprocessing pipeline matching exact Day 2-4 test specifications (224x224, ImageNet normalized).
- 7-class probability prediction with confidence scoring and clinical risk stratifications.
- Interactive Grad-CAM visualization targeting model.features[8] with adjustable overlay alpha.
- Comprehensive dermatology class guide and clinical diagnostic disclaimers.
"""

import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    NUM_CLASSES,
    compute_gradcam_explanation,
    get_ham10000_sample_catalog,
    get_inference_device,
    load_cached_mediscan_model,
    preprocess_image_for_inference,
    resolve_project_path,
    run_model_inference,
    validate_and_load_image,
)

# Clinical risk classification mapping
CLINICAL_RISK_LEVELS: Dict[str, Dict[str, str]] = {
    "mel": {
        "level": "HIGH RISK (Malignant)",
        "badge_color": "#ef4444",
        "badge_bg": "#fee2e2",
        "urgency": "Urgent Dermatologic Evaluation & Biopsy Recommended",
    },
    "bcc": {
        "level": "HIGH RISK (Malignant)",
        "badge_color": "#f97316",
        "badge_bg": "#ffedd5",
        "urgency": "Specialist Consultation & Excision Assessment Recommended",
    },
    "akiec": {
        "level": "MODERATE RISK (Pre-Malignant)",
        "badge_color": "#eab308",
        "badge_bg": "#fef9c3",
        "urgency": "Dermatologic Examination & Treatment Planning Recommended",
    },
    "bkl": {
        "level": "LOW RISK (Benign)",
        "badge_color": "#10b981",
        "badge_bg": "#d1fae5",
        "urgency": "Routine Clinical Monitoring",
    },
    "df": {
        "level": "LOW RISK (Benign)",
        "badge_color": "#10b981",
        "badge_bg": "#d1fae5",
        "urgency": "Routine Clinical Monitoring",
    },
    "nv": {
        "level": "LOW RISK (Benign)",
        "badge_color": "#10b981",
        "badge_bg": "#d1fae5",
        "urgency": "Periodic ABCD Dermoscopy Monitoring",
    },
    "vasc": {
        "level": "LOW RISK (Benign)",
        "badge_color": "#10b981",
        "badge_bg": "#d1fae5",
        "urgency": "Routine Clinical Follow-up",
    },
}


def setup_page_config() -> None:
    """Configure Streamlit page title, icon, and layout."""
    st.set_page_config(
        page_title="MediScan — AI Skin Lesion Diagnostic Platform",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_styles() -> None:
    """Inject polished medical aesthetic CSS."""
    st.markdown(
        """
        <style>
        /* Modern Clean Medical Styling */
        .main-header {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin-bottom: 0.5rem;
        }
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .prediction-badge {
            display: inline-block;
            font-weight: 600;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
        }
        .disclaimer-box {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #92400e;
            margin-top: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(device: torch.device) -> Dict[str, Any]:
    """Render sidebar with system metrics, mode selector, and explainability controls."""
    with st.sidebar:
        st.title("🔬 MediScan")
        st.caption("AI-Assisted Dermoscopic Lesion Diagnostic System")
        st.markdown("---")

        # System Architecture & Status Card
        st.subheader("System Architecture")
        st.markdown(
            f"""
            - **Primary Backbone:** EfficientNet-B0
            - **Parameters:** 4,016,515 (Frozen Feature Extractor)
            - **Classification Head:** Linear (1280 → 7)
            - **Input Resolution:** $224 \\times 224$ RGB
            - **Hardware Device:** `{device.type.upper()}`
            - **Test Top-1 Accuracy:** **75.79%**
            - **Test Top-2 Accuracy:** **89.55%**
            """
        )
        st.markdown("---")

        # Input Selection Mode
        st.subheader("Input Source")
        input_mode = st.radio(
            "Select Dermoscopic Input Source:",
            options=["Upload Image", "HAM10000 Dataset Samples"],
            index=0,
            help="Choose between uploading your own dermoscopic photo or testing an authenticated HAM10000 case.",
        )
        st.markdown("---")

        # Grad-CAM Controls
        st.subheader("Explainability (Grad-CAM)")
        st.markdown(
            "Target Layer: **`model.features[8]`** *(Verified Final Conv Block)*"
        )
        gradcam_alpha = st.slider(
            "Heatmap Blend Weight (α)",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Controls transparency of the Grad-CAM heatmap superimposed onto the original lesion image.",
        )

        gradcam_target_choice = st.selectbox(
            "Grad-CAM Target Class",
            options=["Top Predicted Class"] + [f"{c.upper()} ({CLASS_DESCRIPTIONS[c]})" for c in CLASS_NAMES],
            index=0,
            help="Target class for gradient backpropagation. Defaults to the model's highest-confidence prediction.",
        )
        st.markdown("---")

        # Diagnostic Classes Reference Guide
        with st.expander("📚 HAM10000 7 Diagnostic Classes"):
            for code in CLASS_NAMES:
                risk = CLINICAL_RISK_LEVELS[code]
                st.markdown(
                    f"**{code.upper()}** — {CLASS_DESCRIPTIONS[code]}  \n"
                    f"<span style='color: {risk['badge_color']}; font-size: 0.8rem; font-weight: bold;'>"
                    f"● {risk['level']}</span>",
                    unsafe_allow_html=True,
                )
                st.write("")

    return {
        "input_mode": input_mode,
        "alpha": gradcam_alpha,
        "target_choice": gradcam_target_choice,
    }


def get_target_class_index(choice_str: str) -> Optional[int]:
    """Parse selected target class string into class index, or None for top prediction."""
    if choice_str == "Top Predicted Class":
        return None
    code = choice_str.split()[0].lower()
    if code in CLASS_NAMES:
        return CLASS_NAMES.index(code)
    return None


def main() -> None:
    """Main application entry point."""
    setup_page_config()
    inject_custom_styles()

    device = get_inference_device()
    sidebar_params = render_sidebar(device)

    st.markdown(
        """
        <div class="main-header">
            <h2>MediScan — Clinical Skin Lesion Diagnostic Platform</h2>
            <p style="color: #64748b; font-size: 1.05rem;">
                Dermatoscopic Computer-Aided Diagnosis with Interpretable Neural Attention (Grad-CAM)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Model Initialization
    try:
        model, checkpoint_meta = load_cached_mediscan_model(
            checkpoint_path="models/checkpoints/efficientnet_b0_best.pth",
            config_path="config/config.yaml",
            device=device,
        )
    except Exception as e:
        st.error(f"❌ Failed to load MediScan model: {e}")
        st.info("Ensure `models/checkpoints/efficientnet_b0_best.pth` and `config/config.yaml` are present.")
        return

    # 2. Image Selection / Upload Interface
    pil_image: Optional[Image.Image] = None
    image_source_label = ""

    col_input, col_meta = st.columns([2, 1])

    with col_input:
        if sidebar_params["input_mode"] == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload a dermoscopic lesion image (JPEG, PNG)",
                type=["jpg", "jpeg", "png"],
                help="Upload a high-resolution or cropped dermoscopic skin lesion photograph.",
            )
            if uploaded_file is not None:
                try:
                    pil_image = validate_and_load_image(uploaded_file)
                    image_source_label = f"Uploaded File: {uploaded_file.name}"
                except Exception as e:
                    st.error(f"Invalid image file: {e}")
                    return
        else:
            # Load HAM10000 sample catalog
            catalog = get_ham10000_sample_catalog(samples_per_class=2)
            if not catalog:
                st.warning("⚠️ No HAM10000 sample cases found in `data/raw/`.")
            else:
                options = [item["label"] for item in catalog]
                selected_label = st.selectbox(
                    "Select a validated HAM10000 test case from the dataset:",
                    options=options,
                    index=0,
                )
                selected_item = next((item for item in catalog if item["label"] == selected_label), None)
                if selected_item:
                    try:
                        pil_image = validate_and_load_image(selected_item["path"])
                        image_source_label = f"HAM10000 Case: {selected_item['image_id']} (Ground Truth: {selected_item['dx'].upper()})"
                    except Exception as e:
                        st.error(f"Failed to load sample image: {e}")
                        return

    with col_meta:
        if pil_image is not None:
            st.markdown("#### Image Metadata")
            st.markdown(
                f"""
                - **Source:** `{image_source_label}`
                - **Dimensions:** {pil_image.width} × {pil_image.height} px
                - **Color Space:** {pil_image.mode} (3 channels)
                - **Inference Res:** $224 \\times 224$ px
                """
            )

    if pil_image is None:
        st.info("👈 Please upload a dermoscopic image or select a sample case from the sidebar to begin analysis.")
        return

    st.markdown("---")

    # 3. Preprocessing & Model Inference
    with st.spinner("Processing dermoscopic image and evaluating neural network..."):
        try:
            tensor, resized_rgb = preprocess_image_for_inference(pil_image)
            prediction_res = run_model_inference(model, tensor, device=device)
        except Exception as e:
            st.error(f"Inference execution failed: {e}")
            return

    # 4. Diagnostic Prediction Dashboard
    pred_code = prediction_res["predicted_class"]
    confidence = prediction_res["confidence"]
    confidence_pct = prediction_res["confidence_pct"]
    risk_info = CLINICAL_RISK_LEVELS.get(pred_code, {
        "level": "UNKNOWN",
        "badge_color": "#6b7280",
        "badge_bg": "#f3f4f6",
        "urgency": "Specialist Consultation Recommended",
    })

    st.subheader("📊 Diagnostic Assessment")

    diag_col1, diag_col2 = st.columns([1.2, 1.8])

    with diag_col1:
        st.markdown(
            f"""
            <div style="background-color: {risk_info['badge_bg']}; border-left: 5px solid {risk_info['badge_color']}; padding: 1.25rem; border-radius: 8px; margin-bottom: 1rem;">
                <span style="color: {risk_info['badge_color']}; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">
                    {risk_info['level']}
                </span>
                <h2 style="margin: 0.25rem 0 0.5rem 0; color: #1e293b;">
                    {pred_code.upper()} — {prediction_res['predicted_description']}
                </h2>
                <div style="font-size: 1.15rem; font-weight: 600; color: #334155; margin-bottom: 0.5rem;">
                    Model Confidence: <span style="color: {risk_info['badge_color']};">{confidence_pct}%</span>
                </div>
                <p style="margin: 0; font-size: 0.9rem; color: #475569;">
                    <strong>Clinical Recommendation:</strong> {risk_info['urgency']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with diag_col2:
        st.markdown("##### 7-Class Probability Distribution")
        probs_df = pd.DataFrame(
            prediction_res["sorted_probabilities"],
            columns=["Diagnosis Code", "Probability"],
        )
        probs_df["Class Name"] = probs_df["Diagnosis Code"].map(CLASS_DESCRIPTIONS)
        probs_df["Confidence (%)"] = (probs_df["Probability"] * 100).round(2)

        for _, row in probs_df.iterrows():
            code_str = row["Diagnosis Code"]
            p_val = float(row["Probability"])
            p_pct = row["Confidence (%)"]
            desc_str = row["Class Name"]

            c_risk = CLINICAL_RISK_LEVELS.get(code_str, {})
            bar_color = c_risk.get("badge_color", "#0ea5e9")

            c1, c2, c3 = st.columns([1.5, 3.5, 1])
            with c1:
                st.markdown(f"**{code_str.upper()}** ({desc_str.split()[0]})")
            with c2:
                st.progress(min(max(p_val, 0.0), 1.0))
            with c3:
                st.markdown(f"`{p_pct:.1f}%`")

    st.markdown("---")

    # 5. Explainable AI: Grad-CAM Visualization
    st.subheader("🔍 Interpretable AI — Grad-CAM Localization")
    st.markdown(
        """
        **Gradient-weighted Class Activation Mapping (Grad-CAM)** reveals which visual features in the dermoscopic image 
        most strongly activated the final convolutional layer (**`model.features[8]`**) of EfficientNet-B0.
        """
    )

    target_idx = get_target_class_index(sidebar_params["target_choice"])
    gradcam_success = False
    gradcam_res = None

    try:
        with st.spinner("Computing Grad-CAM gradients and synthesizing heatmaps..."):
            gradcam_res = compute_gradcam_explanation(
                model=model,
                input_tensor=tensor,
                resized_rgb=resized_rgb,
                target_class_idx=target_idx,
                alpha=sidebar_params["alpha"],
            )
            gradcam_success = True
    except Exception as e:
        st.warning(f"⚠️ Grad-CAM visualization encountered an issue: {e}. Diagnostic predictions above remain valid.")

    if gradcam_success and gradcam_res is not None:
        target_name_display = gradcam_res["target_class_name"].upper()
        target_desc_display = CLASS_DESCRIPTIONS.get(gradcam_res["target_class_name"], "")

        st.caption(
            f"Displaying Grad-CAM activation for class: **{target_name_display} — {target_desc_display}** "
            f"(Alpha = `{sidebar_params['alpha']:.2f}`)"
        )

        g_col1, g_col2, g_col3 = st.columns(3)

        with g_col1:
            st.image(
                resized_rgb,
                caption="[1] Standardized Lesion Input (224x224)",
                width="stretch",
            )

        with g_col2:
            st.image(
                gradcam_res["heatmap_rgb"],
                caption="[2] Neural Activation Heatmap (Jet Colormap)",
                width="stretch",
            )

        with g_col3:
            st.image(
                gradcam_res["overlay"],
                caption="[3] Grad-CAM Superimposed Diagnostic Overlay",
                width="stretch",
            )

        # Interpretation Guide
        st.markdown(
            """
            > **How to interpret this diagnostic visualization:**
            > - **Red & Yellow Regions:** Visual patterns that most strongly drove the model's classification decision (e.g. pigment network asymmetries, atypical dots/globules, focal pigmentation).
            > - **Blue & Cyan Regions:** Background skin, illumination artifacts, or lesion zones with negligible influence on this diagnostic category.
            """
        )

    st.markdown("---")

    # 6. Structured Exportable Clinical Summary
    st.subheader("📑 Diagnostic Export")
    summary_data = {
        "source": image_source_label,
        "model": "EfficientNet-B0 (Day 9 Locked Checkpoint)",
        "prediction": {
            "code": pred_code,
            "description": prediction_res["predicted_description"],
            "risk_level": risk_info["level"],
            "confidence": float(confidence),
            "confidence_pct": float(confidence_pct),
        },
        "probabilities": {k: round(v, 6) for k, v in prediction_res["probabilities"].items()},
        "gradcam_target_class": gradcam_res["target_class_name"] if gradcam_res else None,
    }
    summary_json = json.dumps(summary_data, indent=2)

    st.download_button(
        label="📥 Download Structured Diagnostic Report (JSON)",
        data=summary_json,
        file_name=f"mediscan_report_{pred_code}_{int(confidence_pct)}pct.json",
        mime="application/json",
        help="Export machine-readable diagnostic findings and probability distribution.",
    )

    # 7. Clinical Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ CLINICAL AND REGULATORY DISCLAIMER:</strong><br>
            MediScan is an investigational computer-aided diagnostic support tool developed for research, education, 
            and clinical algorithmic auditing. It is not approved by the FDA or CE as a standalone automated diagnostic medical device. 
            Automated predictions and Grad-CAM visualizations must never supersede histological biopsy, dermoscopic inspection by a certified 
            dermatologist, or formal clinical judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
