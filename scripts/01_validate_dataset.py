"""
Script 01: Validate Dataset Integrity & Diagnostics.

Usage:
    python scripts/01_validate_dataset.py [--config configs/baseline.yaml]
"""

import sys
import os
import argparse
import logging
import json

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.validation import DatasetValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("01_ValidateDataset")


def main():
    parser = argparse.ArgumentParser(description="01 - Validate Dataset Health")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    args = parser.parse_args()

    logger.info("=== STEP 01: Dataset Integrity & Health Audit ===")
    validator = DatasetValidator(raw_dir=args.raw_dir)
    report = validator.validate()

    os.makedirs("results/tables", exist_ok=True)
    report_path = "results/tables/dataset_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Validation finished with status: {report['status']}")
    logger.info(f"Summary saved to '{report_path}'.")


if __name__ == "__main__":
    main()
