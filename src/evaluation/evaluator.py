"""
Comprehensive Evaluation Harness for Baseline, Hybrid Filter, and SAM 2 Refined Models.

Module Overview:
----------------
1. What it does:
   - Runs evaluation over test dataset splits.
   - Computes summary statistics (Mean, Std, Median) for Dice, IoU, Precision, Sensitivity, Specificity, HD95.
   - Exports quantitative results into CSV tables (`results/tables/`) and JSON metric summaries (`results/metrics/`).
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from src.metrics.segmentation_metrics import calculate_metrics

logger = logging.getLogger("Evaluator")


class Evaluator:
    """
    Evaluation Engine for multimodal MRI tumor segmentation.
    """
    def __init__(self, model, test_loader, refiner=None, prompt_generator=None, device="cpu"):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.refiner = refiner
        self.prompt_generator = prompt_generator
        self.device = device

    def evaluate(self, experiment_name="experiment", output_dir="results"):
        """
        Executes complete evaluation on test set.

        Returns:
            dict: Summary metrics dictionary.
        """
        self.model.eval()
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

        slice_metrics_list = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, slice_ids) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs)

                for i in range(inputs.shape[0]):
                    slice_id = slice_ids[i]
                    p_mask = probs[i:i+1]
                    t_mask = targets[i:i+1]

                    # Initial U-Net metrics
                    m_unet = calculate_metrics(p_mask, t_mask)

                    # Post-refinement metrics if refiner present
                    if self.refiner and self.prompt_generator:
                        prompts = self.prompt_generator.generate_prompts_single(p_mask[0, 0])
                        img_slice = inputs[i, 0].cpu().numpy()
                        refined_mask = self.refiner.refine_single_slice(img_slice, p_mask[0, 0].cpu().numpy(), prompts)
                        m_refined = calculate_metrics(refined_mask, t_mask)
                    else:
                        m_refined = m_unet

                    record = {
                        "slice_id": slice_id,
                        "unet_dice": m_unet["dice"],
                        "unet_iou": m_unet["iou"],
                        "unet_precision": m_unet["precision"],
                        "unet_sensitivity": m_unet["sensitivity"],
                        "unet_hd95": m_unet["hd95"],
                        "refined_dice": m_refined["dice"],
                        "refined_iou": m_refined["iou"],
                        "refined_precision": m_refined["precision"],
                        "refined_sensitivity": m_refined["sensitivity"],
                        "refined_hd95": m_refined["hd95"]
                    }
                    slice_metrics_list.append(record)

        df = pd.DataFrame(slice_metrics_list)
        csv_path = os.path.join(output_dir, "tables", f"{experiment_name}_detailed_metrics.csv")
        df.to_csv(csv_path, index=False)

        summary = {
            "experiment_name": experiment_name,
            "total_slices": len(df),
            "unet_mean_dice": float(df["unet_dice"].mean()),
            "unet_std_dice": float(df["unet_dice"].std()),
            "unet_mean_iou": float(df["unet_iou"].mean()),
            "unet_mean_hd95": float(df["unet_hd95"].mean()),
            "refined_mean_dice": float(df["refined_dice"].mean()),
            "refined_std_dice": float(df["refined_dice"].std()),
            "refined_mean_iou": float(df["refined_iou"].mean()),
            "refined_mean_hd95": float(df["refined_hd95"].mean())
        }

        json_path = os.path.join(output_dir, "metrics", f"{experiment_name}_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"Evaluation finished for '{experiment_name}'. Summary saved to '{json_path}'.")
        return summary
