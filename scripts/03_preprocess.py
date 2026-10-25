"""
Script 03: Preprocess MRI Dataset (Clipping, Z-Score Normalization, Resizing).

Usage:
    python scripts/03_preprocess.py [--config configs/baseline.yaml]
"""

import sys
import os
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders
from src.preprocessing.preprocessing import MRIPreprocessingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("03_Preprocess")


def main():
    parser = argparse.ArgumentParser(description="03 - Run MRI Preprocessing Pipeline")
    args = parser.parse_args()

    logger.info("=== STEP 03: Preprocessing Pipeline Execution ===")
    pipeline = MRIPreprocessingPipeline(normalize=True, norm_type="zscore", clip_percentiles=(0.5, 99.5), target_size=(128, 128))

    dataloaders = get_dataloaders(raw_dir="data/raw", batch_size=4)
    inputs, targets, slice_ids = next(iter(dataloaders["train"]))

    sample_mri = inputs[0].numpy()
    processed_mri = pipeline(sample_mri)

    logger.info(f"Input shape: {sample_mri.shape} -> Processed shape: {processed_mri.shape}")
    logger.info(f"Processed channels mean: {np.mean(processed_mri, axis=(1, 2))}")
    logger.info("Preprocessing step verified successfully.")


if __name__ == "__main__":
    main()
