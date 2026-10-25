"""
Automatic Prompt Generation Module for Downstream SAM 2 Refinement.

Module Overview:
----------------
1. What it does:
   - Extracts bounding boxes and point prompts (centroids) from predicted U-Net tumor masks.
   - Connected Component Analysis isolates distinct tumor lesion regions while filtering small false-positive noise.
   - Robustly handles empty masks (zero predictions) and multi-focal tumor lesions.
"""

import numpy as np
import cv2
import torch


class AutomaticPromptGenerator:
    """
    Automatic Prompt Generator extracting bounding boxes and centroids from U-Net predictions.
    """
    def __init__(self, mode="bbox", min_component_area=15, bbox_padding=5):
        self.mode = mode
        self.min_component_area = min_component_area
        self.bbox_padding = bbox_padding

    def generate_prompts_single(self, binary_mask):
        if hasattr(binary_mask, "detach"):
            binary_mask = binary_mask.detach().cpu().numpy()

        if binary_mask.ndim == 3:
            binary_mask = binary_mask[0]

        mask_uint8 = (binary_mask > 0.5).astype(np.uint8)
        H, W = mask_uint8.shape

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

        bboxes = []
        points = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_component_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            ymin = max(0, y - self.bbox_padding)
            xmin = max(0, x - self.bbox_padding)
            ymax = min(H, y + h + self.bbox_padding)
            xmax = min(W, x + w + self.bbox_padding)

            cx, cy = centroids[i]

            bboxes.append([int(ymin), int(xmin), int(ymax), int(xmax)])
            points.append([float(cy), float(cx)])

        if len(bboxes) == 0:
            default_box = [int(H * 0.25), int(W * 0.25), int(H * 0.75), int(W * 0.75)]
            default_point = [float(H / 2.0), float(W / 2.0)]
            return {
                "bboxes": [default_box],
                "points": [default_point],
                "has_tumor": False
            }

        return {
            "bboxes": bboxes,
            "points": points,
            "has_tumor": True
        }

    def __call__(self, pred_masks):
        if hasattr(pred_masks, "detach"):
            pred_masks = pred_masks.detach().cpu().numpy()

        if pred_masks.ndim == 3:
            pred_masks = pred_masks[:, np.newaxis, :, :]

        batch_prompts = []
        for i in range(pred_masks.shape[0]):
            prompt = self.generate_prompts_single(pred_masks[i, 0])
            batch_prompts.append(prompt)

        return batch_prompts
