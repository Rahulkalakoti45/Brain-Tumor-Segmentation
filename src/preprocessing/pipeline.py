"""
Modular Preprocessing Pipeline for Multimodal MRI Slices.

Module Overview:
----------------
1. What it does:
   - Applies per-channel Z-score intensity normalization: (X - mean) / std.
   - Applies intensity percentile clipping (e.g., 0.5% to 99.5%) to remove scanner artifact outliers.
   - Resizes or crops 2D slices to target uniform dimensions (e.g., 128x128).

2. Why it is required:
   - Multimodal MRI intensities (T1, T1ce, T2, FLAIR) are non-standardized across scanners and patients.
   - Normalization stabilizes gradient updates during neural network backpropagation.

3. Inputs:
   - NumPy array of shape (4, Height, Width) representing 4 MRI modalities.
4. Outputs:
   - Preprocessed NumPy array of shape (4, Target_Height, Target_Width).
"""

import numpy as np
import cv2

class MRIPreprocessingPipeline:
    """
    Modular Preprocessing Pipeline for 4-channel MRI slice arrays.
    """
    def __init__(self, normalize=True, clip_percentiles=(0.5, 99.5), target_size=(128, 128)):
        """
        Parameters:
            normalize (bool): Whether to apply Z-score normalization.
            clip_percentiles (tuple): Lower and upper intensity percentiles for clipping.
            target_size (tuple): Target (Height, Width) spatial resolution.
        """
        self.normalize = normalize
        self.clip_percentiles = clip_percentiles
        self.target_size = target_size

    def _zscore_normalize(self, channel):
        """Applies Z-score normalization to non-zero brain voxels."""
        mask = channel > 0
        if not np.any(mask):
            return channel
            
        mean = np.mean(channel[mask])
        std = np.std(channel[mask])
        if std > 1e-8:
            channel[mask] = (channel[mask] - mean) / std
        return channel

    def _clip_outliers(self, channel):
        """Clips intensity values to percentile bounds."""
        mask = channel > 0
        if not np.any(mask):
            return channel
            
        p_low, p_high = np.percentile(channel[mask], self.clip_percentiles)
        channel[mask] = np.clip(channel[mask], p_low, p_high)
        return channel

    def _resize(self, channel):
        """Resizes spatial slice to target_size."""
        if channel.shape[:2] == self.target_size:
            return channel
        return cv2.resize(channel, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)

    def __call__(self, mri_array):
        """
        Applies preprocessing pipeline to 4-channel MRI slice array.

        Input:
            mri_array (np.ndarray): Shape (4, H, W)
        Output:
            np.ndarray: Preprocessed array of shape (4, target_H, target_W)
        """
        num_channels = mri_array.shape[0]
        processed_channels = []

        for c in range(num_channels):
            channel = mri_array[c].copy()

            # Step 1: Outlier clipping
            if self.clip_percentiles:
                channel = self._clip_outliers(channel)

            # Step 2: Z-score normalization
            if self.normalize:
                channel = self._zscore_normalize(channel)

            # Step 3: Spatial resizing
            if self.target_size:
                channel = self._resize(channel)

            processed_channels.append(channel)

        return np.stack(processed_channels, axis=0).astype(np.float32)
