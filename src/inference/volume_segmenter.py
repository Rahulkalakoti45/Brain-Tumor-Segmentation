"""
Volume-level segmentation shared by the CLI script and the web app.

Runs the full slice-wise pipeline on one patient volume:
    Preprocessing -> Hybrid Filter -> 2D U-Net -> Automatic Prompts -> SAM 2 Refinement

Input:
    mri_4d (np.ndarray): Shape (4, H, W, D), channels ordered as MODALITIES.
Output:
    Binary tumor mask of shape (H, W, D), uint8.
"""

import os
import logging
import numpy as np
import torch

from src.preprocessing.pipeline import MRIPreprocessingPipeline
from src.filtering.hybrid_filter import HybridMRIFilter
from src.models.unet import UNet2D
from src.prompt_generation.prompt_generator import AutomaticPromptGenerator
from src.refinement.sam2_refiner import SAM2TinyRefiner

logger = logging.getLogger("VolumeSegmenter")

MODALITIES = ["t1", "t1ce", "t2", "flair"]


def load_model(config, checkpoint_path=None, device="cpu"):
    """
    Builds the U-Net from config and loads weights if a checkpoint exists.

    Returns:
        (model, checkpoint_loaded): model in eval mode, and whether trained weights were loaded.
    """
    model_cfg = config.get("model", {})
    model = UNet2D(
        in_channels=model_cfg.get("in_channels", 4),
        out_channels=model_cfg.get("out_channels", 1)
    ).to(device)

    checkpoint_loaded = False
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from '{checkpoint_path}'...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        checkpoint_loaded = True
    else:
        logger.warning(f"No checkpoint found at '{checkpoint_path}'. Using untrained weights.")

    model.eval()
    return model, checkpoint_loaded


def segment_volume(mri_4d, config, checkpoint_path=None, progress_callback=None):
    """
    Segments a whole 3D patient volume slice by slice.

    Parameters:
        mri_4d (np.ndarray): Shape (4, H, W, D) MRI volume.
        config (dict): Parsed YAML config (model, filtering and training sections are used).
        checkpoint_path (str, optional): Path to a trained U-Net checkpoint.
        progress_callback (callable, optional): Called with a float in [0, 1] after each slice.

    Returns:
        (mask, checkpoint_loaded): binary mask (H, W, D) as uint8, and whether trained weights were used.
    """
    device = config.get("training", {}).get("device", "cpu")
    _, height, width, depth = mri_4d.shape

    prep = MRIPreprocessingPipeline(target_size=(height, width))
    hybrid_filt = HybridMRIFilter()
    prompt_gen = AutomaticPromptGenerator()
    sam2_refiner = SAM2TinyRefiner(use_mock=True)
    model, checkpoint_loaded = load_model(config, checkpoint_path, device)

    predicted_mask = np.zeros((height, width, depth), dtype=np.uint8)

    with torch.no_grad():
        for s_idx in range(depth):
            slice_4ch = mri_4d[:, :, :, s_idx]

            # Skip empty slices outside the brain
            if np.sum(slice_4ch > 0) == 0:
                if progress_callback:
                    progress_callback((s_idx + 1) / depth)
                continue

            proc_slice = hybrid_filt(prep(slice_4ch))
            inp_tensor = torch.from_numpy(proc_slice).unsqueeze(0).to(device)

            logits = model(inp_tensor)
            unet_mask = (torch.sigmoid(logits) > 0.5).float()

            prompts = prompt_gen(unet_mask)
            refined = sam2_refiner.refine_single_slice(proc_slice, unet_mask[0, 0].cpu().numpy(), prompts[0])

            predicted_mask[:, :, s_idx] = (refined > 0.5).astype(np.uint8)

            if progress_callback:
                progress_callback((s_idx + 1) / depth)

    return predicted_mask, checkpoint_loaded
