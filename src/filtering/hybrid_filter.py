"""
Hybrid Filtering Module Combining Bilateral Filtering and Morphological Black Top-Hat Enhancement.

Module Overview:
----------------
1. What it does:
   - Combines Bilateral Filtering (edge-preserving noise reduction) with Black Top-Hat (dark feature isolation).
   - Pipeline per modality channel:
        MRI Slice -> Bilateral Filter -> Black Top-Hat -> Fusion (Bilateral + alpha * TopHat) -> Enhanced MRI

2. Why it is required:
   - Proposed central hypothesis: Combining spatial non-linear smoothing with morphological boundary accentuation boosts U-Net feature learning for subtle tumor boundaries.

3. Inputs:
   - NumPy array of shape (4, Height, Width) representing 4 MRI modalities.
4. Outputs:
   - Enhanced NumPy array of shape (4, Height, Width).
"""

import numpy as np
from src.filtering.bilateral_filter import BilateralFilterModule
from src.filtering.black_tophat import BlackTopHatFilterModule

class HybridMRIFilter:
    """
    Hybrid Filter pipeline integrating Bilateral Filtering and Morphological Black Top-Hat enhancement.
    """
    def __init__(self, bilateral_d=5, sigma_color=75.0, sigma_space=75.0, kernel_size=5, kernel_shape="ellipse", alpha=0.5):
        """
        Parameters:
            bilateral_d (int): Bilateral filter neighborhood diameter.
            sigma_color (float): Bilateral filter intensity sigma.
            sigma_space (float): Bilateral filter spatial sigma.
            kernel_size (int): Black Top-Hat morphological kernel size.
            kernel_shape (str): 'ellipse', 'rect', or 'cross'.
            alpha (float): Scaling weight for Black Top-Hat enhancement feature map (default: 0.5).
        """
        self.bilateral = BilateralFilterModule(d=bilateral_d, sigma_color=sigma_color, sigma_space=sigma_space)
        self.black_tophat = BlackTopHatFilterModule(kernel_size=kernel_size, kernel_shape=kernel_shape)
        self.alpha = alpha

    def apply_channel(self, img_channel):
        """Applies hybrid filtering to a single 2D modality slice."""
        # Step 1: Edge-preserving Bilateral Filter
        smoothed = self.bilateral.apply(img_channel)
        
        # Step 2: Black Top-Hat Feature Map
        tophat_map = self.black_tophat.apply(smoothed)
        
        # Step 3: Feature Fusion
        enhanced = smoothed + self.alpha * tophat_map
        return enhanced

    def __call__(self, mri_array):
        """
        Applies hybrid filtering across all 4 MRI channels.

        Input:
            mri_array (np.ndarray): Shape (4, H, W)
        Output:
            np.ndarray: Enhanced array of shape (4, H, W)
        """
        num_channels = mri_array.shape[0]
        enhanced_channels = []

        for c in range(num_channels):
            enhanced_c = self.apply_channel(mri_array[c])
            enhanced_channels.append(enhanced_c)

        return np.stack(enhanced_channels, axis=0).astype(np.float32)
