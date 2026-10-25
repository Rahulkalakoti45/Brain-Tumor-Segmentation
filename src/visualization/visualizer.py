"""
High-Quality Medical Imaging Visualization and Figure Generation Module.

Module Overview:
----------------
1. What it does:
   - Renders 8-panel side-by-side qualitative comparison figures showing:
     (1) T1, (2) T1ce, (3) T2, (4) FLAIR, (5) Ground Truth, (6) U-Net Mask, (7) Refined Mask, (8) Error Overlay.
   - Plots training curves (Loss, Dice, IoU vs. Epochs).
   - Saves figures into 'results/figures/' for reports and presentations.

2. Why it is required:
   - Qualitative visual comparison is essential for verifying boundary sharpness and identifying false positive regions.

3. Inputs:
   - 4-channel MRI slice, ground truth mask, predicted mask, refined mask.
4. Outputs:
   - Saved PNG/PDF figures in 'results/figures/'.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def plot_qualitative_comparison(mri_4ch, ground_truth, unet_pred, refined_pred, slice_id="sample", save_dir="results/figures"):
    """
    Renders 8-panel qualitative MRI comparison grid.

    Parameters:
        mri_4ch (np.ndarray): Shape (4, H, W) -> [T1, T1ce, T2, FLAIR]
        ground_truth (np.ndarray): Shape (H, W)
        unet_pred (np.ndarray): Shape (H, W)
        refined_pred (np.ndarray): Shape (H, W)
        slice_id (str): Identifier name for output file.
        save_dir (str): Folder to save output figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if hasattr(mri_4ch, "detach"):
        mri_4ch = mri_4ch.detach().cpu().numpy()
    if hasattr(ground_truth, "detach"):
        ground_truth = ground_truth.detach().cpu().numpy()
    if hasattr(unet_pred, "detach"):
        unet_pred = unet_pred.detach().cpu().numpy()
    if hasattr(refined_pred, "detach"):
        refined_pred = refined_pred.detach().cpu().numpy()

    if ground_truth.ndim == 3:
        ground_truth = ground_truth[0]
    if unet_pred.ndim == 3:
        unet_pred = unet_pred[0]
    if refined_pred.ndim == 3:
        refined_pred = refined_pred[0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Multimodal MRI Segmentation Analysis - Slice: {slice_id}", fontsize=14, fontweight="bold")

    titles = ["(a) T1 MRI", "(b) T1ce MRI", "(c) T2 MRI", "(d) FLAIR MRI",
              "(e) Ground Truth", "(f) U-Net Prediction", "(g) SAM 2 Refined", "(h) Error Overlay"]

    # Normalize MRI modalities for visualization
    for i in range(4):
        ax = axes[i // 4, i % 4]
        img = mri_4ch[i]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        ax.imshow(img_norm, cmap="gray")
        ax.set_title(titles[i])
        ax.axis("off")

    # (e) Ground Truth
    axes[1, 0].imshow(mri_4ch[3], cmap="gray")
    axes[1, 0].imshow(ground_truth, cmap="jet", alpha=0.5)
    axes[1, 0].set_title(titles[4])
    axes[1, 0].axis("off")

    # (f) U-Net Prediction
    axes[1, 1].imshow(mri_4ch[3], cmap="gray")
    axes[1, 1].imshow(unet_pred > 0.5, cmap="Reds", alpha=0.5)
    axes[1, 1].set_title(titles[5])
    axes[1, 1].axis("off")

    # (g) SAM 2 Refined
    axes[1, 2].imshow(mri_4ch[3], cmap="gray")
    axes[1, 2].imshow(refined_pred > 0.5, cmap="Greens", alpha=0.5)
    axes[1, 2].set_title(titles[6])
    axes[1, 2].axis("off")

    # (h) Error Overlay: Green = True Positive, Red = False Positive, Blue = False Negative
    gt_b = (ground_truth > 0.5).astype(int)
    pr_b = (refined_pred > 0.5).astype(int)
    
    rgb = np.zeros((*gt_b.shape, 3), dtype=np.float32)
    rgb[..., 1] = (gt_b & pr_b) # Green = TP
    rgb[..., 0] = (pr_b & ~gt_b) # Red = FP
    rgb[..., 2] = (gt_b & ~pr_b) # Blue = FN

    axes[1, 3].imshow(mri_4ch[3], cmap="gray")
    axes[1, 3].imshow(rgb, alpha=0.6)
    axes[1, 3].set_title(titles[7] + " (G:TP, R:FP, B:FN)")
    axes[1, 3].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{slice_id}_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path
