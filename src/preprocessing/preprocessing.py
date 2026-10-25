"""
Modular Preprocessing Pipeline for 4-Channel MRI Slices.

Module Overview:
----------------
1. What it does:
   - Integrates outlier clipping, Z-score intensity normalization, and spatial resizing/cropping.
   - Provides a unified callable interface for 4-channel MRI arrays.
"""

import numpy as np
import cv2
from src.preprocessing.normalization import clip_outliers, zscore_normalize, minmax_normalize


class MRIPreprocessingPipeline:
    """
    Modular Preprocessing Pipeline for 4-channel MRI slice arrays.
    """
    def __init__(self, normalize=True, norm_type="zscore", clip_percentiles=(0.5, 99.5), target_size=(128, 128)):
        self.normalize = normalize
        self.norm_type = norm_type
        self.clip_percentiles = clip_percentiles
        self.target_size = target_size

    def _resize(self, channel):
        if channel.shape[:2] == self.target_size:
            return channel
        return cv2.resize(channel, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)

    def __call__(self, mri_array):
        """
        Applies preprocessing pipeline to 4-channel MRI array.

        Input:
            mri_array (np.ndarray): Shape (4, H, W)
        Output:
            np.ndarray: Preprocessed array of shape (4, target_H, target_W)
        """
        num_channels = mri_array.shape[0]
        processed_channels = []

        for c in range(num_channels):
            channel = mri_array[c].copy()

            if self.clip_percentiles:
                channel = clip_outliers(channel, percentiles=self.clip_percentiles)

            if self.normalize:
                if self.norm_type == "zscore":
                    channel = zscore_normalize(channel)
                elif self.norm_type == "minmax":
                    channel = minmax_normalize(channel)

            if self.target_size:
                channel = self._resize(channel)

            processed_channels.append(channel)

        return np.stack(processed_channels, axis=0).astype(np.float32)
