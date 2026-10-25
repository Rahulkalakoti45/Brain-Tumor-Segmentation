"""
Dataset Loader and Patient Split Manager for BraTS Multimodal MRI Volumes.

Module Overview:
----------------
1. What it does:
   - Scans 'data/raw/' for BraTS subject folders containing 4 MRI modalities (T1, T1ce, T2, FLAIR) and ground truth segmentation mask.
   - Creates patient-level train/validation/test splits (70% train, 15% val, 15% test) to prevent data leakage.
   - Extracts 2D axial slices from 3D volumes and builds a PyTorch Dataset for training.
   - Includes a mock volume generator if raw data is missing, enabling end-to-end execution testing.

2. Why it is required:
   - Deep learning models (U-Net) require standard PyTorch Tensors in batches.
   - MRI volumes are 3D, but running 2D slice-based models is essential for CPU execution.
   - Patient-level splitting ensures strict scientific validity (no slices from the same patient appear in both train and test sets).

3. Inputs:
   - Subject directories containing NIfTI files (.nii.gz).
4. Outputs:
   - PyTorch Tensors of shape (Batch_Size, 4, Height, Width) representing 4 MRI modalities.
   - Ground truth segmentation mask Tensors of shape (Batch_Size, 1, Height, Width).
"""

import os
import glob
import json
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

logger = logging.getLogger("DataLoader")

MODALITIES = ["t1", "t1ce", "t2", "flair"]

def generate_patient_splits(raw_dir="data/raw", splits_dir="data/splits", train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Generates reproducible patient-level train/val/test splits saved to a JSON manifest.

    Parameters:
        raw_dir (str): Directory containing raw BraTS subject folders.
        splits_dir (str): Target directory to save patient_splits.json.
        train_ratio (float): Fraction of patients for training (default: 0.70).
        val_ratio (float): Fraction of patients for validation (default: 0.15).
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Mapping of 'train', 'val', and 'test' to lists of patient folder names.
    """
    os.makedirs(splits_dir, exist_ok=True)
    split_file = os.path.join(splits_dir, "patient_splits.json")
    
    if os.path.exists(split_file):
        logger.info(f"Loading existing patient split manifest from '{split_file}'...")
        with open(split_file, "r") as f:
            return json.load(f)
            
    # List all subject directories
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        
    patients = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    if len(patients) == 0:
        logger.warning(f"No patient folders found in '{raw_dir}'. Creating mock split for verification...")
        patients = [f"BraTS2021_MOCK_{i:03d}" for i in range(10)]
        
    random.seed(seed)
    random.shuffle(patients)
    
    n_total = len(patients)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    if not test_patients:
        test_patients = val_patients # Fallback for small datasets
        
    splits = {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients
    }
    
    with open(split_file, "w") as f:
        json.dump(splits, f, indent=4)
        
    logger.info(f"Generated patient splits: Train={len(train_patients)}, Val={len(val_patients)}, Test={len(test_patients)}")
    return splits

class BraTS2DSliceDataset(Dataset):
    """
    PyTorch Dataset for 2D Multimodal MRI Slices.
    """
    def __init__(self, raw_dir, patient_list, transform=None, slice_plane="axial", min_signal_thresh=0.01):
        """
        Parameters:
            raw_dir (str): Path to raw dataset root.
            patient_list (list): List of patient folder names assigned to this split.
            transform (callable, optional): Preprocessing or hybrid filter transformation function.
            slice_plane (str): 'axial', 'coronal', or 'sagittal'.
            min_signal_thresh (float): Fraction of non-zero pixels required to include slice (skips empty brain space).
        """
        self.raw_dir = raw_dir
        self.patient_list = patient_list
        self.transform = transform
        self.slice_plane = slice_plane
        self.min_signal_thresh = min_signal_thresh
        self.slice_indices = []
        
        self._index_dataset()
        
    def _index_dataset(self):
        """Builds an index of valid (patient_id, slice_idx) tuples."""
        logger.info(f"Indexing slices for {len(self.patient_list)} patients...")
        for patient_id in self.patient_list:
            patient_path = os.path.join(self.raw_dir, patient_id)
            if not os.path.exists(patient_path):
                # Mock dataset fallback if real data folder is missing
                for s_idx in range(10, 20):
                    self.slice_indices.append((patient_id, s_idx, True))
                continue
                
            # Locate FLAIR or T1 to get slice count
            flair_files = glob.glob(os.path.join(patient_path, "*flair.nii*"))
            if not flair_files:
                continue
            
            try:
                flair_img = nib.load(flair_files[0])
                depth = flair_img.shape[2] # Axial depth
                
                # Sample middle 60% of slices containing brain tissue
                start_slice = int(depth * 0.2)
                end_slice = int(depth * 0.8)
                
                for s_idx in range(start_slice, end_slice):
                    self.slice_indices.append((patient_id, s_idx, False))
            except Exception as e:
                logger.error(f"Error reading volume for indexing patient {patient_id}: {e}")
                
        logger.info(f"Indexed total {len(self.slice_indices)} 2D slices.")
        
    def __len__(self):
        return len(self.slice_indices)
        
    def __getitem__(self, idx):
        patient_id, s_idx, is_mock = self.slice_indices[idx]
        
        if is_mock:
            # Generate synthetic 4-channel MRI slice and ground truth mask
            img_slice = np.random.normal(loc=0.5, scale=0.1, size=(4, 128, 128)).astype(np.float32)
            mask_slice = np.zeros((1, 128, 128), dtype=np.float32)
            mask_slice[0, 40:80, 40:80] = 1.0 # Mock tumor region
            
            if self.transform:
                img_slice = self.transform(img_slice)
                
            return torch.from_numpy(img_slice), torch.from_numpy(mask_slice), f"{patient_id}_slice_{s_idx}"
            
        patient_path = os.path.join(self.raw_dir, patient_id)
        mod_slices = []
        
        for mod in MODALITIES:
            mod_file = glob.glob(os.path.join(patient_path, f"*{mod}.nii*"))[0]
            vol = nib.load(mod_file).get_fdata()
            mod_slices.append(vol[:, :, s_idx])
            
        img_slice = np.stack(mod_slices, axis=0).astype(np.float32) # Shape: (4, H, W)
        
        # Load segmentation mask
        seg_files = glob.glob(os.path.join(patient_path, "*seg.nii*"))
        if seg_files:
            seg_vol = nib.load(seg_files[0]).get_fdata()
            mask_2d = seg_vol[:, :, s_idx]
            # Convert non-zero labels (Whole Tumor: ET + TC + ED) to binary mask 1.0
            mask_slice = (mask_2d > 0).astype(np.float32)[np.newaxis, :, :] # Shape: (1, H, W)
        else:
            mask_slice = np.zeros((1, img_slice.shape[1], img_slice.shape[2]), dtype=np.float32)
            
        if self.transform:
            img_slice = self.transform(img_slice)
            
        return torch.from_numpy(img_slice), torch.from_numpy(mask_slice), f"{patient_id}_slice_{s_idx}"

def get_dataloaders(raw_dir="data/raw", splits_dir="data/splits", batch_size=16, transform=None):
    """
    Creates PyTorch DataLoaders for Train, Validation, and Test splits.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    splits = generate_patient_splits(raw_dir=raw_dir, splits_dir=splits_dir)
    
    dataloaders = {}
    for split_name in ["train", "val", "test"]:
        dataset = BraTS2DSliceDataset(
            raw_dir=raw_dir,
            patient_list=splits[split_name],
            transform=transform
        )
        shuffle = (split_name == "train")
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0, # 0 for Windows multi-threading safety
            pin_memory=False
        )
        
    return dataloaders
