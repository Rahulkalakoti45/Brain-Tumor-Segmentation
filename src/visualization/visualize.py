"""
Visualization Utilities Module for Multimodal MRI, Filtering, and Segmentation Mask Overlays.

Module Overview:
----------------
1. What it does:
   - Renders 8-panel side-by-side qualitative comparison figures.
   - Plots training loss and metric curves across epochs.
   - Renders filtering diagnostic comparisons (Raw vs Bilateral vs Black Top-Hat vs Hybrid).
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_qualitative_comparison(mri_4ch, ground_truth, unet_pred, refined_pred, slice_id="sample", save_dir="results/figures"):
    """
    Renders 8-panel qualitative MRI comparison grid.
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

    for i in range(4):
        ax = axes[i // 4, i % 4]
        img = mri_4ch[i]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        ax.imshow(img_norm, cmap="gray")
        ax.set_title(titles[i])
        ax.axis("off")

    axes[1, 0].imshow(mri_4ch[3], cmap="gray")
    axes[1, 0].imshow(ground_truth, cmap="jet", alpha=0.5)
    axes[1, 0].set_title(titles[4])
    axes[1, 0].axis("off")

    axes[1, 1].imshow(mri_4ch[3], cmap="gray")
    axes[1, 1].imshow(unet_pred > 0.5, cmap="Reds", alpha=0.5)
    axes[1, 1].set_title(titles[5])
    axes[1, 1].axis("off")

    axes[1, 2].imshow(mri_4ch[3], cmap="gray")
    axes[1, 2].imshow(refined_pred > 0.5, cmap="Greens", alpha=0.5)
    axes[1, 2].set_title(titles[6])
    axes[1, 2].axis("off")

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


def plot_filtering_comparison(raw_channel, bilateral_channel, tophat_channel, hybrid_channel, save_path="results/figures/filtering_comparison.png"):
    """
    Renders 4-panel filtering ablation diagnostic figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    imgs = [raw_channel, bilateral_channel, tophat_channel, hybrid_channel]
    titles = ["1. Raw Input", "2. Bilateral Filter", "3. Black Top-Hat", "4. Hybrid Filtered"]

    for i in range(4):
        img = imgs[i]
        norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        axes[i].imshow(norm, cmap="gray")
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_training_curves(history, save_path="results/figures/training_curves.png"):
    """
    Plots training loss and validation Dice/IoU curves across epochs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    ax1.set_title("Loss Trajectory")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["val_dice"], "g-o", label="Val Dice")
    ax2.plot(epochs, history["val_iou"], "m--s", label="Val IoU")
    ax2.set_title("Segmentation Accuracy Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path
