"""
Intensity Normalization Utilities for Multimodal MRI Scans.

Module Overview:
----------------
1. What it does:
   - Implements per-channel Z-score intensity normalization: (X - mean) / std.
   - Implements Min-Max intensity scaling to [0.0, 1.0] range.
   - Implements percentile clipping (e.g., 0.5% to 99.5%) to remove scanner outlier artifacts.
"""

import numpy as np


def clip_outliers(channel, percentiles=(0.5, 99.5)):
    """Clips intensity values to lower and upper percentile bounds."""
    mask = channel > 0
    if not np.any(mask):
        return channel

    p_low, p_high = np.percentile(channel[mask], percentiles)
    clipped = channel.copy()
    clipped[mask] = np.clip(channel[mask], p_low, p_high)
    return clipped


def zscore_normalize(channel):
    """Applies Z-score normalization to non-zero brain voxels."""
    mask = channel > 0
    if not np.any(mask):
        return channel

    mean = np.mean(channel[mask])
    std = np.std(channel[mask])
    normed = channel.copy()
    if std > 1e-8:
        normed[mask] = (channel[mask] - mean) / std
    return normed


def minmax_normalize(channel):
    """Scales non-zero brain voxel intensities to [0, 1]."""
    mask = channel > 0
    if not np.any(mask):
        return channel

    min_val = np.min(channel[mask])
    max_val = np.max(channel[mask])
    normed = channel.copy()
    if max_val > min_val + 1e-8:
        normed[mask] = (channel[mask] - min_val) / (max_val - min_val)
    return normed
