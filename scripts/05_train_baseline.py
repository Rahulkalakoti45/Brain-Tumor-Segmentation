"""
Script 05: Train Baseline 2D U-Net Model.

Usage:
    python scripts/05_train_baseline.py [--config configs/baseline.yaml]
"""

import sys
import os
import argparse
import logging
import yaml
import json
import torch
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders
from src.models.unet import UNet2D
from src.losses.segmentation_losses import DiceBCELoss
from src.training.trainer import ModelTrainer
from src.visualization.visualize import plot_training_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("05_TrainBaseline")


def main():
    parser = argparse.ArgumentParser(description="05 - Train Baseline U-Net Model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    logger.info("=== STEP 05: Baseline U-Net Model Training ===")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    device = config.get("training", {}).get("device", "cpu")
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

    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])

    trainer = ModelTrainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device
    )

    history = trainer.fit()

    exp_dir = "experiments/baseline_unet"
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    plot_training_curves(history, save_path=os.path.join(exp_dir, "baseline_training_curves.png"))
    logger.info(f"Baseline training finished. Results saved to '{exp_dir}'.")


if __name__ == "__main__":
    main()
