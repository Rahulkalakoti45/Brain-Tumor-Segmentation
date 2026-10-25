"""
NIfTI Multimodal MRI Loader and Synthetic Mock Volume Generator.

Module Overview:
----------------
1. What it does:
   - Manages loading of 3D/4D NIfTI medical imaging files (.nii, .nii.gz) using Nibabel.
   - Extracts volume voxel data, spatial resolution (affine), and header metadata.
   - Provides a synthetic mock volume generator to simulate BraTS subjects (T1, T1ce, T2, FLAIR, SEG)
     when raw dataset folders are missing, enabling seamless 100% dry-run testing.

2. Why it is required:
   - Medical MRI scans require standardized IO handling across multiple modalities.
   - Robust mock data generation guarantees end-to-end execution capability without hard code dependency on local disk state.
"""

import os
import glob
import logging
import numpy as np
import nibabel as nib

logger = logging.getLogger("NiftiLoader")

MODALITIES = ["t1", "t1ce", "t2", "flair"]


class NiftiLoader:
    """
    Utility class for loading NIfTI MRI volumes and ground truth masks.
    """
    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = raw_dir

    def load_subject_volume(self, subject_id):
        """
        Loads all 4 MRI modalities and segmentation mask for a given subject ID.

        Returns:
            dict: {
                'modalities': np.ndarray of shape (4, H, W, D),
                'mask': np.ndarray of shape (H, W, D),
                'affine': np.ndarray (4x4 matrix),
                'is_mock': bool
            }
        """
        subject_path = os.path.join(self.raw_dir, subject_id)
        if not os.path.exists(subject_path):
            logger.info(f"Subject directory '{subject_path}' not found. Generating mock volume...")
            return self.generate_mock_volume(subject_id)

        mod_volumes = []
        affine = np.eye(4)

        for mod in MODALITIES:
            mod_files = glob.glob(os.path.join(subject_path, f"*{mod}.nii*"))
            if not mod_files:
                logger.warning(f"Modality '{mod}' missing for {subject_id}. Generating mock slice fallback.")
                return self.generate_mock_volume(subject_id)
            
            img = nib.load(mod_files[0])
            mod_volumes.append(img.get_fdata(dtype=np.float32))
            affine = img.affine

        modalities_stacked = np.stack(mod_volumes, axis=0) # Shape: (4, H, W, D)

        seg_files = glob.glob(os.path.join(subject_path, "*seg.nii*"))
        if seg_files:
            seg_img = nib.load(seg_files[0])
            seg_mask = seg_img.get_fdata(dtype=np.float32)
        else:
            seg_mask = np.zeros(modalities_stacked.shape[1:], dtype=np.float32)

        return {
            "modalities": modalities_stacked,
            "mask": seg_mask,
            "affine": affine,
            "is_mock": False
        }

    @staticmethod
    def generate_mock_volume(subject_id="mock_subject", spatial_shape=(128, 128, 64)):
        """
        Generates realistic synthetic 4-channel MRI volume with simulated spherical tumor mask.
        """
        H, W, D = spatial_shape
        modalities = np.zeros((4, H, W, D), dtype=np.float32)

        # Generate synthetic brain tissue ellipsoid
        y, x, z = np.ogrid[:H, :W, :D]
        center_y, center_x, center_z = H // 2, W // 2, D // 2
        brain_mask = ((y - center_y)**2 / (H*0.4)**2 + 
                      (x - center_x)**2 / (W*0.4)**2 + 
                      (z - center_z)**2 / (D*0.4)**2) <= 1.0

        for c in range(4):
            base_intensity = 100.0 + c * 50.0
            noise = np.random.normal(loc=0.0, scale=10.0, size=(H, W, D))
            modalities[c] = (base_intensity + noise) * brain_mask

        # Generate synthetic spherical tumor region
        tumor_cy, tumor_cx, tumor_cz = H // 2 + 10, W // 2 - 10, D // 2
        tumor_radius = min(H, W) // 6
        tumor_dist = (y - tumor_cy)**2 + (x - tumor_cx)**2 + (z - tumor_cz)**2
        tumor_mask = (tumor_dist <= tumor_radius**2).astype(np.float32)

        # Boost tumor intensity on FLAIR and T1ce
        modalities[1] += 80.0 * tumor_mask # T1ce enhancement
        modalities[3] += 120.0 * tumor_mask # FLAIR edema/enhancement

        return {
            "modalities": modalities,
            "mask": tumor_mask,
            "affine": np.eye(4),
            "is_mock": True
        }
