"""
Script 04: Test and Benchmark Bilateral, Black Top-Hat, and Hybrid MRI Filtering.

Usage:
    python scripts/04_test_filtering.py
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders
from src.filtering.bilateral_filter import BilateralFilterModule
from src.filtering.black_tophat import BlackTopHatFilterModule
from src.filtering.hybrid_filter import HybridMRIFilter
from src.visualization.visualize import plot_filtering_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("04_TestFiltering")


def main():
    logger.info("=== STEP 04: Hybrid Filtering Module Diagnostics ===")
    dataloaders = get_dataloaders(raw_dir="data/raw", batch_size=1)
    inputs, targets, slice_ids = next(iter(dataloaders["train"]))

    flair_slice = inputs[0, 3].numpy() # FLAIR channel

    b_filter = BilateralFilterModule()
    t_filter = BlackTopHatFilterModule()
    h_filter = HybridMRIFilter()

    smoothed = b_filter.apply(flair_slice)
    tophat = t_filter.apply(smoothed)
    hybrid = h_filter(inputs[0].numpy())[3]

    out_path = "results/figures/filtering_ablation_comparison.png"
    plot_filtering_comparison(flair_slice, smoothed, tophat, hybrid, save_path=out_path)

    logger.info(f"Filtering diagnostic panel saved to '{out_path}'.")


if __name__ == "__main__":
    main()
