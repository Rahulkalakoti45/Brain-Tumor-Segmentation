"""
Script 07: Automatic Prompt Generation from U-Net Predictions.

Usage:
    python scripts/07_generate_prompts.py
"""

import sys
import os
import argparse
import logging
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders
from src.models.unet import UNet2D
from src.prompt_generation.automatic_prompt import AutomaticPromptGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("07_GeneratePrompts")


def main():
    logger.info("=== STEP 07: Automatic Prompt Generation ===")
    dataloaders = get_dataloaders(raw_dir="data/raw", batch_size=4)
    inputs, targets, slice_ids = next(iter(dataloaders["val"]))

    model = UNet2D(in_channels=4, out_channels=1)
    ckpt_path = "checkpoints/baseline/best_model.pth"
    if os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint from '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.sigmoid(logits)

    generator = AutomaticPromptGenerator(mode="bbox")
    batch_prompts = generator(probs)

    for i in range(len(batch_prompts)):
        p = batch_prompts[i]
        logger.info(f"Slice '{slice_ids[i]}' -> Has Tumor: {p['has_tumor']} | BBoxes: {p['bboxes']} | Centroids: {p['points']}")

    logger.info("Automatic prompt generation completed successfully.")


if __name__ == "__main__":
    main()
