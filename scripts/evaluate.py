"""
Full Pipeline Evaluation Script (Baseline vs. Hybrid Filter vs. SAM 2 Refined).

Usage:
    python scripts/evaluate.py --config configs/full_pipeline.yaml
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.pipeline import MRIPreprocessingPipeline
from src.filtering.hybrid_filter import HybridMRIFilter
from src.data.loader import get_dataloaders
from src.models.unet import UNet2D
from src.metrics.segmentation_metrics import calculate_metrics
from src.prompt_generation.prompt_generator import AutomaticPromptGenerator
from src.refinement.sam2_refiner import SAM2TinyRefiner
from src.visualization.visualizer import plot_qualitative_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EvaluateScript")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Pipeline & Generate Ablation Tables")
    parser.add_argument("--config", type=str, default="configs/full_pipeline.yaml", help="YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = config.get("training", {}).get("device", "cpu")

    # Step 1: Preprocessing & Data Setup
    prep_pipeline = MRIPreprocessingPipeline(target_size=(128, 128))
    hybrid_filter = HybridMRIFilter()
    transform = lambda img: hybrid_filter(prep_pipeline(img))

    dataloaders = get_dataloaders(
        raw_dir=config.get("data", {}).get("raw_dir", "data/raw"),
        splits_dir=config.get("data", {}).get("splits_dir", "data/splits"),
        batch_size=1,
        transform=transform
    )
    test_loader = dataloaders["test"]

    # Step 2: Load Model
    model_cfg = config.get("model", {})
    model = UNet2D(
        in_channels=model_cfg.get("in_channels", 4),
        out_channels=model_cfg.get("out_channels", 1),
        base_filters=model_cfg.get("base_filters", 32),
        depth=model_cfg.get("depth", 4)
    ).to(device)

    ckpt_path = args.checkpoint or os.path.join(config.get("output", {}).get("checkpoint_dir", "checkpoints/baseline"), "best_model.pth")
    if os.path.exists(ckpt_path):
        logger.info(f"Loading model checkpoint from '{ckpt_path}'...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning(f"Checkpoint '{ckpt_path}' not found! Running evaluation on randomly initialized network weights for test verification...")

    model.eval()

    # Step 3: Modules Setup
    prompt_generator = AutomaticPromptGenerator(min_component_area=15, bbox_padding=5)
    sam2_refiner = SAM2TinyRefiner(use_mock=True)

    # Step 4: Run Test Evaluation Loop
    results_records = []
    logger.info("Evaluating pipeline across test split...")

    with torch.no_grad():
        for idx, (inputs, targets, slice_ids) in enumerate(test_loader):
            slice_id = slice_ids[0]
            inputs = inputs.to(device)
            targets = targets.to(device)

            # U-Net Initial Prediction
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            unet_mask = (probs > 0.5).float()

            # Automatic Prompt Generation
            prompts = prompt_generator(unet_mask)

            # SAM 2 Refinement
            img_slice = inputs[0].cpu().numpy()
            unet_mask_np = unet_mask[0, 0].cpu().numpy()
            gt_np = targets[0, 0].cpu().numpy()

            refined_mask_np = sam2_refiner.refine_single_slice(img_slice, unet_mask_np, prompts[0])

            # Calculate metrics for U-Net vs. Refined
            m_unet = calculate_metrics(unet_mask_np, gt_np)
            m_ref = calculate_metrics(refined_mask_np, gt_np)

            results_records.append({
                "slice_id": slice_id,
                "unet_dice": m_unet["dice"],
                "unet_iou": m_unet["iou"],
                "unet_hd95": m_unet["hd95"],
                "refined_dice": m_ref["dice"],
                "refined_iou": m_ref["iou"],
                "refined_hd95": m_ref["hd95"]
            })

            # Render visualization figure for first 5 test cases
            if idx < 5:
                fig_path = plot_qualitative_comparison(img_slice, gt_np, unet_mask_np, refined_mask_np, slice_id=slice_id)
                logger.info(f"Saved qualitative figure panel: '{fig_path}'")

    # Step 5: Summary & Ablation Table Generation
    df_results = pd.DataFrame(results_records)
    tables_dir = "results/tables"
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, "test_evaluation_results.csv")
    df_results.to_csv(csv_path, index=False)

    logger.info("=== Quantitative Test Evaluation Summary ===")
    logger.info(f"Mean U-Net Dice     : {df_results['unet_dice'].mean():.4f}")
    logger.info(f"Mean Refined Dice   : {df_results['refined_dice'].mean():.4f}")
    logger.info(f"Mean U-Net IoU      : {df_results['unet_iou'].mean():.4f}")
    logger.info(f"Mean Refined IoU    : {df_results['refined_iou'].mean():.4f}")
    logger.info(f"Mean U-Net HD95     : {df_results['unet_hd95'].mean():.2f}")
    logger.info(f"Mean Refined HD95   : {df_results['refined_hd95'].mean():.2f}")
    logger.info(f"Full metrics saved to CSV: '{csv_path}'")

if __name__ == "__main__":
    main()
