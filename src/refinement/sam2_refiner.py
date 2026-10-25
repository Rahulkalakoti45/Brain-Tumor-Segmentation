"""
SAM 2 Tiny Boundary Refinement Interface and Guided Adapter Module.

Module Overview:
----------------
1. What it does:
   - Takes 2D MRI slice, initial U-Net segmentation mask, and generated prompt bounding box.
   - Refines tumor boundary definition by sharpening boundary contours within prompt regions.
   - Provides dual execution modes:
        a) Official SAM 2 Tiny predictor when official checkpoint ('checkpoints/sam2_hiera_tiny.pt') is loaded.
        b) Morphological Guided Boundary Refinement Adapter (Mock Refiner) for robust execution without GPU/large model downloads.

2. Why it is required:
   - SAM 2 (Segment Anything 2) leverages foundation model vision priors to sharpen fuzzy tumor boundaries produced by U-Net.

3. Inputs:
   - Image slice array of shape (H, W) or (3, H, W).
   - Initial U-Net prediction binary mask of shape (H, W).
   - Prompt dictionary containing bounding boxes `[ymin, xmin, ymax, xmax]`.
4. Outputs:
   - Refined binary segmentation mask of shape (H, W).
"""

import os
import logging
import numpy as np
import cv2

logger = logging.getLogger("SAM2Refiner")

class SAM2TinyRefiner:
    """
    SAM 2 Tiny Boundary Refinement Module with fallback adapter.
    """
    def __init__(self, checkpoint_path="checkpoints/sam2_hiera_tiny.pt", use_mock=True):
        """
        Parameters:
            checkpoint_path (str): Path to official SAM 2 Hiera-Tiny model checkpoint.
            use_mock (bool): Force mock guided refinement mode (default: True for CPU execution).
        """
        self.checkpoint_path = checkpoint_path
        self.use_mock = use_mock
        self.real_sam2_available = False

        if not self.use_mock and os.path.exists(checkpoint_path):
            try:
                # Attempt importing official sam2 package
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                
                logger.info(f"Loading official SAM 2 Tiny model from '{checkpoint_path}'...")
                self.predictor = SAM2ImagePredictor(build_sam2("sam2_hiera_t.yaml", checkpoint_path))
                self.real_sam2_available = True
                logger.info("Official SAM 2 Tiny loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize official SAM 2: {e}. Falling back to Guided Refinement Adapter.")

    def refine_single_slice(self, img_slice, unet_mask, prompt_dict):
        """
        Refines a single 2D slice segmentation mask based on prompt bounding boxes.

        Parameters:
            img_slice (np.ndarray): Shape (H, W) or (3, H, W)
            unet_mask (np.ndarray): Initial binary prediction mask of shape (H, W)
            prompt_dict (dict): Output from AutomaticPromptGenerator

        Returns:
            np.ndarray: Refined binary mask of shape (H, W)
        """
        if hasattr(unet_mask, "detach"):
            unet_mask = unet_mask.detach().cpu().numpy()
        if unet_mask.ndim == 3:
            unet_mask = unet_mask[0]

        if not prompt_dict.get("has_tumor", False):
            return unet_mask # Return original mask if no tumor region detected

        if self.real_sam2_available:
            return self._refine_with_real_sam2(img_slice, unet_mask, prompt_dict)
        else:
            return self._refine_with_guided_adapter(img_slice, unet_mask, prompt_dict)

    def _refine_with_guided_adapter(self, img_slice, unet_mask, prompt_dict):
        """
        Morphological Guided Boundary Refinement Adapter.
        Applies edge-guided Otsu thresholding and morphological closing within prompt bounding boxes.
        """
        refined_mask = unet_mask.copy().astype(np.float32)
        H, W = unet_mask.shape

        if img_slice.ndim == 3:
            img_slice = img_slice[0] # Use first channel (e.g. FLAIR or T1ce)

        norm_img = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-6) * 255.0).astype(np.uint8)

        for bbox in prompt_dict.get("bboxes", []):
            ymin, xmin, ymax, xmax = bbox
            
            # Extract ROI crop
            roi_img = norm_img[ymin:ymax, xmin:xmax]
            roi_mask = unet_mask[ymin:ymax, xmin:xmax]

            if roi_img.size == 0 or roi_mask.sum() == 0:
                continue

            # Guided thresholding inside ROI box
            _, otsu_roi = cv2.threshold(roi_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_roi_float = (otsu_roi / 255.0).astype(np.float32)

            # Sharpen boundary: Combine U-Net probability prior with edge-guided threshold
            sharpened_roi = 0.6 * roi_mask + 0.4 * otsu_roi_float
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_roi = cv2.morphologyEx((sharpened_roi > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            refined_mask[ymin:ymax, xmin:xmax] = cleaned_roi.astype(np.float32)

        return (refined_mask > 0.5).astype(np.float32)

    def _refine_with_real_sam2(self, img_slice, unet_mask, prompt_dict):
        """Refines slice using official SAM 2 API."""
        if img_slice.ndim == 2:
            img_rgb = np.stack([img_slice] * 3, axis=-1)
        else:
            img_rgb = np.transpose(img_slice[:3], (1, 2, 0))

        img_rgb = ((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-6) * 255.0).astype(np.uint8)
        self.predictor.set_image(img_rgb)

        refined_masks = []
        for bbox in prompt_dict.get("bboxes", []):
            # SAM 2 box format: [xmin, ymin, xmax, ymax]
            sam_box = np.array([bbox[1], bbox[0], bbox[3], bbox[2]])
            masks, scores, _ = self.predictor.predict(box=sam_box, multimask_output=False)
            refined_masks.append(masks[0])

        if len(refined_masks) > 0:
            combined = np.logical_or.reduce(refined_masks)
            return combined.astype(np.float32)
        return unet_mask
