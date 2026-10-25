"""
Unified PyTorch Training Engine for Baseline and Hybrid Brain Tumor Segmentation Models.

Module Overview:
----------------
1. What it does:
   - Executes PyTorch training and validation loops across configured epochs.
   - Computes train/val loss, updates weights using AdamW optimizer.
   - Evaluates Dice score, IoU, Precision, Sensitivity, HD95 per epoch.
   - Saves best model checkpoint ('best_model.pth') when validation Dice improves.

2. Why it is required:
   - Provides a clean, reproducible training harness for baseline vs. hybrid experiment comparisons.

3. Inputs:
   - Model, DataLoaders, Loss Function, Optimizer, Config dictionary.
4. Outputs:
   - Trained PyTorch model checkpoint files and metric history dictionary.
"""

import os
import time
import logging
import numpy as np
import torch
from src.metrics.segmentation_metrics import calculate_metrics

logger = logging.getLogger("Trainer")

class ModelTrainer:
    """
    Modular PyTorch Trainer for Brain Tumor Segmentation.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config, device="cpu"):
        """
        Parameters:
            model (nn.Module): PyTorch segmentation network (UNet2D).
            train_loader (DataLoader): DataLoader for training split.
            val_loader (DataLoader): DataLoader for validation split.
            criterion (nn.Module): Loss function (DiceBCELoss).
            optimizer (Optimizer): PyTorch optimizer (AdamW).
            config (dict): Experiment configuration dictionary.
            device (str): Compute device ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        self.epochs = config.get("training", {}).get("epochs", 10)
        self.checkpoint_dir = config.get("output", {}).get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_precision": [],
            "val_sensitivity": [],
            "val_hd95": []
        }

    def train_epoch(self):
        """Executes one training epoch."""
        self.model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / max(1, len(self.train_loader))

    def validate_epoch(self):
        """Executes one validation epoch and computes evaluation metrics."""
        self.model.eval()
        running_loss = 0.0
        metrics_accumulator = {
            "dice": [], "iou": [], "precision": [], "sensitivity": [], "hd95": []
        }

        with torch.no_grad():
            for inputs, targets, _ in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                probs = torch.sigmoid(outputs)
                m = calculate_metrics(probs, targets)
                
                for k in metrics_accumulator:
                    metrics_accumulator[k].append(m[k])

        avg_loss = running_loss / max(1, len(self.val_loader))
        avg_metrics = {k: float(np.mean(v)) for k, v in metrics_accumulator.items()}
        avg_metrics["loss"] = avg_loss
        return avg_metrics

    def fit(self):
        """Executes complete multi-epoch training pipeline."""
        logger.info(f"Starting training for {self.epochs} epochs on device '{self.device}'...")
        best_val_dice = -1.0
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.validate_epoch()
            epoch_time = time.time() - t0

            val_loss = val_metrics["loss"]
            val_dice = val_metrics["dice"]
            val_iou = val_metrics["iou"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_dice)
            self.history["val_iou"].append(val_iou)
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_sensitivity"].append(val_metrics["sensitivity"])
            self.history["val_hd95"].append(val_metrics["hd95"])

            logger.info(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] ({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | Val HD95: {val_metrics['hd95']:.2f}"
            )

            # Checkpoint best model
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_dice": val_dice,
                    "config": self.config
                }, best_ckpt_path)
                logger.info(f"--> Saved new best checkpoint to '{best_ckpt_path}' (Val Dice: {val_dice:.4f})")

        total_duration = time.time() - start_time
        logger.info(f"Training completed in {total_duration / 60.0:.2f} minutes. Best Val Dice: {best_val_dice:.4f}")
        return self.history
