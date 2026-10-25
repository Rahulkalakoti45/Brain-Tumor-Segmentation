"""
2D Spatial Patch Extraction and Volume Reconstruction Module.

Module Overview:
----------------
1. What it does:
   - Extracts overlapping or non-overlapping 2D spatial patches (e.g. 64x64 or 128x128) from 2D MRI slices.
   - Reconstructs full spatial slices from predicted patch arrays.
"""

import numpy as np


class PatchExtractor:
    """
    Extracts and reconstructs 2D spatial patches.
    """
    def __init__(self, patch_size=(64, 64), stride=(32, 32)):
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, img_slice):
        """
        Extracts 2D patches from an MRI slice.

        Parameters:
            img_slice (np.ndarray): Shape (C, H, W) or (H, W)

        Returns:
            np.ndarray: Extracted patches array.
        """
        is_multi_channel = img_slice.ndim == 3
        if is_multi_channel:
            C, H, W = img_slice.shape
        else:
            H, W = img_slice.shape
            C = 1
            img_slice = img_slice[np.newaxis, :, :]

        ph, pw = self.patch_size
        sh, sw = self.stride

        patches = []
        for y in range(0, H - ph + 1, sh):
            for x in range(0, W - pw + 1, sw):
                patch = img_slice[:, y:y+ph, x:x+pw]
                patches.append(patch if is_multi_channel else patch[0])

        return np.stack(patches, axis=0)

    def reconstruct_slice(self, patches, target_shape):
        """
        Reconstructs slice from non-overlapping or averaged overlapping patches.
        """
        if len(patches) == 0:
            return np.zeros(target_shape, dtype=np.float32)

        ph, pw = self.patch_size
        sh, sw = self.stride

        is_multi_channel = patches.ndim == 4
        if is_multi_channel:
            N, C, _, _ = patches.shape
            H, W = target_shape[1], target_shape[2]
            reconstructed = np.zeros((C, H, W), dtype=np.float32)
            counts = np.zeros((C, H, W), dtype=np.float32)
        else:
            N, _, _ = patches.shape
            H, W = target_shape[0], target_shape[1]
            reconstructed = np.zeros((H, W), dtype=np.float32)
            counts = np.zeros((H, W), dtype=np.float32)

        idx = 0
        for y in range(0, H - ph + 1, sh):
            for x in range(0, W - pw + 1, sw):
                if idx < N:
                    if is_multi_channel:
                        reconstructed[:, y:y+ph, x:x+pw] += patches[idx]
                        counts[:, y:y+ph, x:x+pw] += 1.0
                    else:
                        reconstructed[y:y+ph, x:x+pw] += patches[idx]
                        counts[y:y+ph, x:x+pw] += 1.0
                    idx += 1

        counts[counts == 0] = 1.0
        return reconstructed / counts
