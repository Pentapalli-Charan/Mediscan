"""
MediScan — Day 13: Synchronize Historical Experiment Runs to MLflow

Populates the canonical 'MediScan' experiment in local mlruns/ tracking store
with actual historical runs:
1. EfficientNet-B0-Baseline (Day 6 / Day 8 Exp 0 + Day 9 test metrics)
2. EfficientNet-B0-LR-5e-4 (Day 8 Exp 1)
3. ResNet50-Benchmark (Day 11 controlled CPU benchmark)
"""

import logging
import os
import sys
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.mlflow_tracker import (
    EXPERIMENT_NAME,
    get_experiment_summary_df,
    populate_historical_experiments,
    setup_mlflow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Initializing MLflow tracking store for MediScan...")
    uri, exp_id = setup_mlflow()
    logger.info(f"Tracking URI: {uri}")
    logger.info(f"Experiment: '{EXPERIMENT_NAME}' (ID: {exp_id})")

    logger.info("Populating verified historical experiment runs...")
    run_ids = populate_historical_experiments()

    logger.info("Successfully synchronized runs:")
    for name, run_id in run_ids.items():
        logger.info(f"  - {name}: {run_id}")

    logger.info("Experiment Run Comparison Table:")
    summary_df = get_experiment_summary_df()
    print("\n" + "=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
