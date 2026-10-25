"""
Script 09: Quantitative Benchmark & Evaluation Across All Experiments.

Usage:
    python scripts/09_evaluate.py [--config configs/full_pipeline.yaml]
"""

import sys
import os
import argparse
import logging
import yaml
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders
from src.models.unet import UNet2D
from src.prompt_generation.automatic_prompt import AutomaticPromptGenerator
from src.refinement.sam2_refiner import SAM2TinyRefiner
from src.evaluation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("09_Evaluate")


def main():
    parser = argparse.ArgumentParser(description="09 - Quantitative Evaluation")
    parser.add_argument("--config", type=str, default="configs/full_pipeline.yaml", help="Path to config YAML")
    args = parser.parse_args()

    logger.info("=== STEP 09: Quantitative Metric Evaluation & Comparative Analysis ===")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataloaders = get_dataloaders(
        raw_dir=config["data"]["raw_dir"],
        splits_dir=config["data"]["splits_dir"],
        batch_size=config["training"]["batch_size"]
    )

    model = UNet2D(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_filters=config["model"]["base_filters"],
        depth=config["model"]["depth"],
        dropout=config["model"]["dropout"]
    )

    # Check for trained checkpoint
    ckpt_path = os.path.join(config["output"]["checkpoint_dir"], "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/baseline/best_model.pth"

    if os.path.exists(ckpt_path):
        logger.info(f"Loading evaluation checkpoint from '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning("No checkpoint file found. Running evaluation on untrained weights.")

    prompt_gen = AutomaticPromptGenerator()
    refiner = SAM2TinyRefiner(use_mock=True)

    evaluator = Evaluator(
        model=model,
        test_loader=dataloaders["test"],
        refiner=refiner,
        prompt_generator=prompt_gen,
        device=config["training"]["device"]
    )

    summary = evaluator.evaluate(experiment_name=config["experiment_name"], output_dir="results")
    logger.info("Quantitative evaluation complete.")
    print("\n" + "="*50)
    print("EVALUATION SUMMARY RESULTS:")
    print("="*50)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
