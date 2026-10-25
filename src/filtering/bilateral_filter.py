"""
Bilateral Filtering Module for Edge-Preserving MRI Noise Reduction.

Module Overview:
----------------
1. What it does:
   - Applies non-linear Bilateral Filtering to 2D MRI slice channels using OpenCV.
   - Smooths background noise while preserving sharp boundaries between brain tissue and tumor regions.

2. Why it is required:
   - MRI acquisitions contain high-frequency noise from magnetic field fluctuations.
   - Standard Gaussian blurring degrades sharp tumor edges. Bilateral filtering preserves edges by weighting pixels based on both spatial distance and intensity difference.

3. Inputs:
   - Single-channel 2D MRI slice (NumPy float32 or uint8 array).
4. Outputs:
   - Edge-smoothed 2D MRI slice of identical shape.
"""

import cv2
import numpy as np

class BilateralFilterModule:
    """
    Edge-preserving Bilateral Filter module for medical images.
    """
    def __init__(self, d=5, sigma_color=75.0, sigma_space=75.0):
        """
        Parameters:
            d (int): Diameter of each pixel neighborhood used during filtering.
            sigma_color (float): Filter sigma in intensity/color space (larger = wider intensity blend).
            sigma_space (float): Filter sigma in coordinate space (larger = farther pixels influence each other).
        """
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, img_slice):
        """
        Applies bilateral filter to 2D numpy array.

        Input:
            img_slice (np.ndarray): Shape (H, W), dtype float32
        Output:
            np.ndarray: Filtered array of shape (H, W)
        """
        # Convert float32 slice to [0, 255] uint8 for OpenCV bilateral filter
        min_val, max_val = img_slice.min(), img_slice.max()
        if max_val - min_val < 1e-6:
            return img_slice

        norm_img = ((img_slice - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        filtered_uint8 = cv2.bilateralFilter(norm_img, self.d, self.sigma_color, self.sigma_space)

        # Scale back to original float32 range
        filtered_float = (filtered_uint8.astype(np.float32) / 255.0) * (max_val - min_val) + min_val
        return filtered_float
