# MediScan — Medical Image Classification System

> ⚠️ **DISCLAIMER:** This project is for **educational/research purposes only** and is **not a clinical diagnostic system**. Predictions are not medical diagnoses and should not be used for clinical decision-making. The model may have bias and generalization limitations.

## Project Objective

Build an end-to-end deep-learning medical image classification system using transfer learning, including proper preprocessing, augmentation, training, evaluation, Grad-CAM explainability, experiment tracking, and a deployed web interface.

## Description

MediScan classifies dermatoscopic skin lesion images into diagnostic categories using convolutional neural networks with transfer learning. The system includes:

- **Data Pipeline:** Preprocessing, augmentation, and efficient data loading for the HAM10000 dataset
- **Model Training:** Transfer learning with EfficientNet-B0 (primary) and ResNet50 (comparison) using staged unfreezing
- **Evaluation:** Comprehensive metrics including per-class precision/recall/F1, confusion matrix, and ROC-AUC curves
- **Explainability:** Grad-CAM heatmap visualizations showing model attention regions
- **Web Application:** Interactive Streamlit interface for image upload, classification, and visualization
- **Experiment Tracking:** MLflow-based logging of hyperparameters, metrics, and artifacts
- **Deployment:** Hosted on Hugging Face Spaces

## Dataset

**HAM10000** ("Human Against Machine with 10000 training images")
- Source: [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- 10,015 dermatoscopic images across 7 diagnostic categories
- License: CC BY-NC-SA 4.0 (non-commercial, share-alike)

## Technology Stack

| Component | Choice |
|---|---|
| Framework | PyTorch |
| Primary Model | EfficientNet-B0 (ImageNet pretrained) |
| Comparison Model | ResNet50 (ImageNet pretrained) |
| Dataset | HAM10000 |
| Augmentation | Albumentations |
| Web Application | Streamlit |
| Experiment Tracking | MLflow |
| Explainability | pytorch-grad-cam |
| Deployment | Hugging Face Spaces |

## Current Status

| Milestone | Status |
|---|---|
| Day 1: Project Setup & EDA | 🔄 In Progress |
| Day 2: Data Preprocessing | ⬜ Not Started |
| Day 3: Data Augmentation | ⬜ Not Started |
| Day 4: DataLoader Setup | ⬜ Not Started |
| Day 5: Base Model Setup | ⬜ Not Started |
| Day 6: Training Loop | ⬜ Not Started |
| Day 7: Training Execution | ⬜ Not Started |
| Day 8: Hyperparameter Tuning | ⬜ Not Started |
| Day 9: Model Evaluation | ⬜ Not Started |
| Day 10: Grad-CAM | ⬜ Not Started |
| Day 11: Model Comparison | ⬜ Not Started |
| Day 12: Web Interface | ⬜ Not Started |
| Day 13: Experiment Tracking | ⬜ Not Started |
| Day 14: Edge-Case Analysis | ⬜ Not Started |
| Day 15: Finalization & Deployment | ⬜ Not Started |

## Project Roadmap

1. **Days 1–4:** Data pipeline — setup, preprocessing, augmentation, DataLoader
2. **Days 5–7:** Model architecture and training execution
3. **Days 8–9:** Hyperparameter tuning and evaluation
4. **Days 10–11:** Grad-CAM explainability and model comparison
5. **Days 12–13:** Web interface and experiment tracking
6. **Days 14–15:** Edge-case analysis, deployment, and finalization

## Setup Instructions

### 1. Clone / Open the Project

```bash
cd "MediScan Project"
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset Setup

**Option A: Kaggle API (recommended)**

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account Settings → API → Create New API Token
3. Place the downloaded `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
4. Run the download script:

```bash
python scripts/download_dataset.py
```

**Option B: Manual Download**

1. Download from https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Extract the ZIP contents into `data/raw/`
3. Ensure `HAM10000_metadata.csv` (or `HAM10000_metadata`) is present in `data/raw/`
4. Run the EDA script to verify:

```bash
python scripts/eda.py
```

### 5. Run Initial EDA

```bash
python scripts/eda.py
```

This generates dataset statistics, visualizations, and a summary report in `reports/`.

## Repository Structure

```
MediScan Project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── config/
│   └── config.yaml              # Centralized configuration
├── data/
│   ├── raw/                     # Original HAM10000 images (gitignored)
│   ├── processed/               # Preprocessed data (gitignored)
│   └── metadata/                # CSV metadata & split indices
├── src/
│   ├── data/                    # Data processing modules
│   ├── models/                  # Model architectures (Day 5+)
│   ├── training/                # Training loop (Day 6+)
│   ├── evaluation/              # Metrics & evaluation (Day 9+)
│   ├── explainability/          # Grad-CAM (Day 10+)
│   ├── experiment/              # MLflow tracking (Day 13+)
│   └── utils/                   # Utility functions
├── scripts/                     # Runnable scripts
├── notebooks/                   # Jupyter notebooks
├── models/checkpoints/          # Saved model weights (gitignored)
├── reports/                     # Generated reports & figures
├── app/                         # Streamlit web app (Day 12+)
├── mlruns/                      # MLflow tracking data (gitignored)
└── tests/                       # Unit tests
```

## Ethical Considerations

- This model is trained on the HAM10000 dataset, which has known demographic biases
- The dataset predominantly represents certain skin tones; performance may vary across demographics
- Predictions should **never** replace professional medical diagnosis
- The model's confidence scores do not represent clinical probability
- DICOM support is provided as format conversion only — the model is specifically trained for dermatoscopic images and cannot meaningfully classify unrelated medical imaging modalities (CT, MRI, X-ray, etc.)

## License

This project uses the HAM10000 dataset under CC BY-NC-SA 4.0 license.
This project is for educational purposes as part of an internship program.
