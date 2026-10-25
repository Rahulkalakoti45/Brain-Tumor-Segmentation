"""
Script 02: Dataset Exploration & Multimodal MRI Visualization.

Usage:
    python scripts/02_visualize_dataset.py [--num-samples 3]
"""

import sys
import os
import argparse
import logging
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("02_VisualizeDataset")


def main():
    parser = argparse.ArgumentParser(description="02 - Dataset Exploration & Visualization")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of sample slices to render")
    args = parser.parse_args()

    logger.info("=== STEP 02: Dataset Multimodal Visualization ===")
    dataloaders = get_dataloaders(raw_dir="data/raw", batch_size=args.num_samples)

    train_loader = dataloaders["train"]
    inputs, targets, slice_ids = next(iter(train_loader))

    save_dir = "results/figures"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(args.num_samples, inputs.shape[0])):
        slice_id = slice_ids[i]
        mri = inputs[i].numpy() # (4, H, W)
        mask = targets[i, 0].numpy() # (H, W)

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        titles = ["T1", "T1ce", "T2", "FLAIR", "Ground Truth Mask"]

        for c in range(4):
            ch = mri[c]
            norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
            axes[c].imshow(norm, cmap="gray")
            axes[c].set_title(titles[c])
            axes[c].axis("off")

        axes[4].imshow(mri[3], cmap="gray")
        axes[4].imshow(mask, cmap="Reds", alpha=0.5)
        axes[4].set_title(titles[4])
        axes[4].axis("off")

        plt.suptitle(f"Sample Dataset Slice: {slice_id}", fontsize=14, fontweight="bold")
        out_path = os.path.join(save_dir, f"dataset_sample_{i+1}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved dataset visualization to '{out_path}'")


if __name__ == "__main__":
    main()
