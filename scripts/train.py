"""
Main Model Training Script for Baseline and Hybrid Brain Tumor Segmentation.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/hybrid_filter.yaml
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import torch.optim as optim

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.pipeline import MRIPreprocessingPipeline
from src.filtering.hybrid_filter import HybridMRIFilter
from src.data.loader import get_dataloaders
from src.models.unet import UNet2D
from src.losses.dice_loss import DiceBCELoss
from src.training.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TrainScript")

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net Medical Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to YAML experiment config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Loading configuration from '{args.config}'...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # Set random seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Step 1: Preprocessing & Hybrid Filtering Setup
    prep_cfg = config.get("preprocessing", {})
    prep_pipeline = MRIPreprocessingPipeline(
        normalize=prep_cfg.get("normalize", True),
        clip_percentiles=tuple(prep_cfg.get("clip_percentiles", [0.5, 99.5])),
        target_size=tuple(prep_cfg.get("target_slice_size", [128, 128]))
    )

    filt_cfg = config.get("filtering", {})
    if filt_cfg.get("enabled", False):
        logger.info("Hybrid Filtering ENABLED (Bilateral Filter + Morphological Black Top-Hat).")
        b_cfg = filt_cfg.get("bilateral", {})
        th_cfg = filt_cfg.get("black_tophat", {})
        hybrid_filter = HybridMRIFilter(
            bilateral_d=b_cfg.get("d", 5),
            sigma_color=b_cfg.get("sigma_color", 75.0),
            sigma_space=b_cfg.get("sigma_space", 75.0),
            kernel_size=th_cfg.get("kernel_size", 5),
            kernel_shape=th_cfg.get("kernel_shape", "ellipse")
        )
        # Composite transform: Preprocess then Hybrid Filter
        transform = lambda img: hybrid_filter(prep_pipeline(img))
    else:
        logger.info("Hybrid Filtering DISABLED (Standard Baseline Preprocessing).")
        transform = prep_pipeline

    # Step 2: DataLoader Setup
    data_cfg = config.get("data", {})
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    splits_dir = data_cfg.get("splits_dir", "data/splits")
    batch_size = config.get("training", {}).get("batch_size", 16)

    dataloaders = get_dataloaders(
        raw_dir=raw_dir,
        splits_dir=splits_dir,
        batch_size=batch_size,
        transform=transform
    )

    # Step 3: Model Architecture
    model_cfg = config.get("model", {})
    model = UNet2D(
        in_channels=model_cfg.get("in_channels", 4),
        out_channels=model_cfg.get("out_channels", 1),
        base_filters=model_cfg.get("base_filters", 32),
        depth=model_cfg.get("depth", 4),
        dropout=model_cfg.get("dropout", 0.1)
    )

    # Step 4: Loss Function & Optimizer
    criterion = DiceBCELoss()
    lr = config.get("training", {}).get("learning_rate", 0.001)
    weight_decay = config.get("training", {}).get("weight_decay", 0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    device = config.get("training", {}).get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"

    # Step 5: Execute Training Harness
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
    logger.info("Model training pipeline completed successfully.")

if __name__ == "__main__":
    main()
