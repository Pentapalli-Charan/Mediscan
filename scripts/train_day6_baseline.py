"""
MediScan — Day 6 Baseline Training Script

Executes the Day 6 baseline training pipeline:
- EfficientNet-B0 with ImageNet pretrained weights
- Frozen backbone, trainable 7-class linear head
- Unweighted CrossEntropyLoss
- Adam optimizer (lr=1e-3)
- ReduceLROnPlateau scheduler on validation loss
- Early stopping (patience=3 on validation loss)
- Saves best checkpoint, history CSV, and training curves
- Strictly protects the test set (test data is NEVER loaded)
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 65)
    logger.info("MediScan — Day 6 Baseline Model Training Execution")
    logger.info("=" * 65)

    results = train_model(
        config_path="config/config.yaml",
        project_root=PROJECT_ROOT,
    )

    # Save structured summary JSON
    summary_path = PROJECT_ROOT / "reports" / "day6_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved training summary JSON to: {summary_path}")
    logger.info("=" * 65)
    logger.info("DAY 6 BASELINE TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
