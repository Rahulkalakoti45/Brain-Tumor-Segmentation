"""
Script 08: Run SAM 2 Tiny Boundary Refinement.

Usage:
    python scripts/08_run_sam2.py
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
from src.refinement.sam2_refiner import SAM2TinyRefiner
from src.visualization.visualize import plot_qualitative_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("08_RunSAM2")


def main():
    logger.info("=== STEP 08: SAM 2 Boundary Refinement ===")
    dataloaders = get_dataloaders(raw_dir="data/raw", batch_size=2)
    inputs, targets, slice_ids = next(iter(dataloaders["test"]))

    model = UNet2D(in_channels=4, out_channels=1)
    ckpt_path = "checkpoints/hybrid_filter/best_model.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(inputs))

    prompt_gen = AutomaticPromptGenerator()
    refiner = SAM2TinyRefiner(use_mock=True)

    for i in range(inputs.shape[0]):
        slice_id = slice_ids[i]
        p_dict = prompt_gen.generate_prompts_single(probs[i, 0])
        img_slice = inputs[i].numpy()
        gt_mask = targets[i, 0].numpy()
        u_mask = (probs[i, 0].numpy() > 0.5).astype(float)

        refined_mask = refiner.refine_single_slice(img_slice, u_mask, p_dict)

        save_path = plot_qualitative_comparison(
            mri_4ch=img_slice,
            ground_truth=gt_mask,
            unet_pred=u_mask,
            refined_pred=refined_mask,
            slice_id=slice_id,
            save_dir="results/figures"
        )
        logger.info(f"Saved qualitative SAM 2 refinement output for '{slice_id}' to '{save_path}'")


if __name__ == "__main__":
    main()
