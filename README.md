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
| Day 1: Project Setup & EDA | ✅ Completed |
| Day 2: Data Preprocessing | ✅ Completed |
| Day 3: Data Augmentation | ✅ Completed |
| Day 4: DataLoader Setup | ✅ Completed |
| Day 5: Base Model Setup | ✅ Completed |
| Day 6: Training Loop | ✅ Completed |
| Day 7: Training Execution | ✅ Completed |
| Day 8: Hyperparameter Tuning | ✅ Completed |
| Day 9: Model Evaluation | ✅ Completed |
| Day 10: Grad-CAM | ✅ Completed |
| Day 11: Model Comparison | ✅ Completed |
| Day 12: Web Interface | ✅ Completed |
| Day 13: Experiment Tracking | ✅ Completed |
| Day 14: Edge-Case Analysis | ✅ Completed |
| Day 15: Finalization & Deployment | ⬜ Up Next |

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
├── app/                         # Streamlit web app (Day 12+)
├── mlruns/                      # Local MLflow tracking store (Day 13+)
└── tests/                       # Unit & regression tests
```

## MLflow Experiment Tracking (Day 13)

MediScan incorporates **MLflow** for local, reproducible, offline experiment tracking, parameter logging, epoch metric curves, and model artifact governance.

### Tracking Configuration
- **Tracking Location:** Local file-based store at `mlruns/`
- **Experiment Name:** `MediScan`
- **Dependency:** `mlflow>=3.16.0` (100% offline, no remote server or internet connection required)

### Tracked Runs & Scientific Integrity
All runs recorded in MLflow reflect actual executed runs from the project:
1. **`EfficientNet-B0-Baseline` (Day 6 / Day 8 Exp 0)**:
   - Primary deployment model. 15 epochs, Adam optimizer (LR=1e-3, Dropout=0.2).
   - Best validation loss: `0.6422` (Epoch 14), validation accuracy: `77.38%`, peak validation accuracy: `77.65%`.
   - Contains locked Day 9 held-out evaluation metrics tagged with provenance.
2. **`EfficientNet-B0-LR-5e-4` (Day 8 Exp 1)**:
   - Evaluated lower learning rate on EfficientNet-B0. 5 epochs.
   - Best validation loss: `0.7283`, validation accuracy: `75.21%`.
3. **`ResNet50-Benchmark` (Day 11 Benchmark)**:
   - Controlled architecture comparison (23.52M parameters). 1-epoch CPU budget benchmark.
   - Validation loss: `0.7646`, validation accuracy: `72.52%`, epoch duration: `5265.4s` (20.3× slower than EfficientNet-B0).

> **Scientific Fairness & Test Isolation Protocol:**
> - **Test-set metrics were not used for hyperparameter selection.**
> - Day 9 held-out test metrics (`Top-1: 75.79%`, `Top-2: 89.55%`, `Macro-F1: 0.5157`, `Weighted-F1: 0.7448`) are explicitly tagged as `historical/final held-out evaluation` and remain permanently locked.
> - Planned Day 8 experiments 2, 3, and 4 were skipped due to CPU compute constraints and are **not** fabricated as MLflow runs.

### Tracked Metadata
- **Parameters:** Architecture, pretrained weights, class count (7), classification head, dropout, frozen/unfrozen status, dataset split counts (train: 6982, val: 1521, test: 1512), split grouping column (`lesion_id`), input resolution ($224 \times 224$), normalization parameters, data augmentation pipeline, optimizer, learning rate, batch size, scheduler, early stopping, random seed (42), compute device, software versions (Python, PyTorch, torchvision), and git commit hash.
- **Metrics:** Epoch-by-epoch `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `learning_rate`, `epoch_time_seconds`, and overall best validation summary metrics.
- **Artifacts:** Epoch history CSVs, configuration YAML snapshots, Day 9 classification reports, and training comparison curves.
- **Canonical Model Checkpoint:** The official production model checkpoint remains locked at `models/checkpoints/efficientnet_b0_best.pth` (MD5: `7c1c6fcbe02e93f0ff8b4a20f62e29e3`).

### Launching the MLflow UI Locally
To inspect the experiments, comparison tables, and metric curves in your browser:

```bash
# Launch local MLflow UI server (binds locally, zero internet required)
mlflow ui --backend-store-uri mlruns --port 5000
```
Open `http://localhost:5000` in your web browser.

To synchronize or refresh the MLflow store via script:
```bash
python scripts/sync_day13_mlflow.py
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
