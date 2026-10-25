"""
Script 10: Run Complete End-to-End Proposed Research Pipeline.

Pipeline Workflow:
1. Validate Dataset Integrity
2. Preprocess MRI Data
3. Apply Hybrid Filtering (Bilateral + Black Top-Hat)
4. Train Baseline & Hybrid U-Net Models
5. Generate Automatic Bounding Box & Centroid Prompts
6. Apply SAM 2 Tiny Boundary Refinement
7. Compute Quantitative Evaluation Metrics & Renders Publication Figures

Usage:
    python scripts/10_run_full_pipeline.py [--epochs 5]
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("10_RunFullPipeline")


def main():
    parser = argparse.ArgumentParser(description="10 - Run Complete Proposed Pipeline")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs for pipeline validation")
    args = parser.parse_args()

    logger.info("==========================================================================")
    logger.info("       STARTING FULL END-TO-END BRAIN TUMOR SEGMENTATION PIPELINE        ")
    logger.info("==========================================================================")

    py_cmd = sys.executable

    # Step 1: Validate Dataset
    logger.info("\n>>> Step 1/7: Validating Dataset Integrity...")
    os.system(f'"{py_cmd}" scripts/01_validate_dataset.py')

    # Step 2: Preprocess
    logger.info("\n>>> Step 2/7: Preprocessing MRI Slices...")
    os.system(f'"{py_cmd}" scripts/03_preprocess.py')

    # Step 3: Test Hybrid Filtering
    logger.info("\n>>> Step 3/7: Benchmarking Hybrid Filtering...")
    os.system(f'"{py_cmd}" scripts/04_test_filtering.py')

    # Step 4: Train Baseline U-Net
    logger.info(f"\n>>> Step 4/7: Training Baseline U-Net ({args.epochs} epochs)...")
    os.system(f'"{py_cmd}" scripts/05_train_baseline.py --epochs {args.epochs}')

    # Step 5: Train Hybrid U-Net
    logger.info(f"\n>>> Step 5/7: Training Hybrid Filtered U-Net ({args.epochs} epochs)...")
    os.system(f'"{py_cmd}" scripts/06_train_hybrid.py --epochs {args.epochs}')

    # Step 6: Automatic Prompting & SAM 2 Refinement
    logger.info("\n>>> Step 6/7: Executing Prompt Generation & SAM 2 Boundary Refinement...")
    os.system(f'"{py_cmd}" scripts/08_run_sam2.py')

    # Step 7: Final Quantitative Evaluation
    logger.info("\n>>> Step 7/7: Executing Final Evaluation & Metric Benchmarking...")
    os.system(f'"{py_cmd}" scripts/09_evaluate.py')

    logger.info("\n==========================================================================")
    logger.info("       SUCCESSFULLY COMPLETED END-TO-END PIPELINE EXECUTION              ")
    logger.info("==========================================================================")


if __name__ == "__main__":
    main()
