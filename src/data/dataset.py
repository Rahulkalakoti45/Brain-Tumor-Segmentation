"""
PyTorch Dataset and Patient-Level Split Manager for BraTS Multimodal MRI Slices.

Module Overview:
----------------
1. What it does:
   - Manages patient-level dataset splits (train, validation, test) to prevent data leakage across slices.
   - Wraps 2D slice indexing and PyTorch DataLoader creation for 4-channel MRI input arrays.
   - Automatically switches to mock volume loading when dataset directory is empty.
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
from src.data.nifti_loader import NiftiLoader

logger = logging.getLogger("Dataset")

MODALITIES = ["t1", "t1ce", "t2", "flair"]


def generate_patient_splits(raw_dir="data/raw", splits_dir="data/splits", train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Generates patient-level train/val/test splits manifest file.

    Returns:
        dict: Patient list mappings for 'train', 'val', and 'test'.
    """
    os.makedirs(splits_dir, exist_ok=True)
    split_file = os.path.join(splits_dir, "patient_splits.json")

    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            return json.load(f)

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)

    patients = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

    if len(patients) == 0:
        logger.info(f"No patient folders found in '{raw_dir}'. Creating mock split manifest.")
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
        test_patients = val_patients

    splits = {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients
    }

    with open(split_file, "w") as f:
        json.dump(splits, f, indent=4)

    return splits


class BraTS2DSliceDataset(Dataset):
    """
    PyTorch Dataset for 2D Multimodal MRI Slices.
    """
    def __init__(self, raw_dir, patient_list, transform=None, slice_plane="axial", min_signal_thresh=0.01):
        self.raw_dir = raw_dir
        self.patient_list = patient_list
        self.transform = transform
        self.slice_plane = slice_plane
        self.min_signal_thresh = min_signal_thresh
        self.loader = NiftiLoader(raw_dir=raw_dir)
        self.slice_indices = []

        self._index_dataset()

    def _index_dataset(self):
        for patient_id in self.patient_list:
            patient_path = os.path.join(self.raw_dir, patient_id)
            if not os.path.exists(patient_path):
                # Mock slice fallback
                for s_idx in range(10, 20):
                    self.slice_indices.append((patient_id, s_idx, True))
                continue

            flair_files = glob.glob(os.path.join(patient_path, "*flair.nii*"))
            if not flair_files:
                for s_idx in range(10, 20):
                    self.slice_indices.append((patient_id, s_idx, True))
                continue

            try:
                flair_img = nib.load(flair_files[0])
                depth = flair_img.shape[2]
                start_slice = int(depth * 0.2)
                end_slice = int(depth * 0.8)

                for s_idx in range(start_slice, end_slice):
                    self.slice_indices.append((patient_id, s_idx, False))
            except Exception as e:
                logger.error(f"Error indexing patient {patient_id}: {e}")

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        patient_id, s_idx, is_mock = self.slice_indices[idx]

        if is_mock:
            mock_vol = NiftiLoader.generate_mock_volume(patient_id)
            img_slice = mock_vol["modalities"][:, :, :, s_idx % mock_vol["modalities"].shape[3]]
            mask_slice = mock_vol["mask"][np.newaxis, :, :, s_idx % mock_vol["mask"].shape[2]]

            # Crop or resize spatial dims to 128x128 if needed
            if img_slice.shape[1:] != (128, 128):
                import cv2
                channels = [cv2.resize(img_slice[c], (128, 128)) for c in range(4)]
                img_slice = np.stack(channels, axis=0)
                mask_slice = cv2.resize(mask_slice[0], (128, 128))[np.newaxis, :, :]

            if self.transform:
                img_slice = self.transform(img_slice)

            return torch.from_numpy(img_slice.astype(np.float32)), torch.from_numpy(mask_slice.astype(np.float32)), f"{patient_id}_slice_{s_idx}"

        patient_path = os.path.join(self.raw_dir, patient_id)
        mod_slices = []

        for mod in MODALITIES:
            mod_file = glob.glob(os.path.join(patient_path, f"*{mod}.nii*"))[0]
            vol = nib.load(mod_file).get_fdata()
            mod_slices.append(vol[:, :, s_idx])

        img_slice = np.stack(mod_slices, axis=0).astype(np.float32)

        seg_files = glob.glob(os.path.join(patient_path, "*seg.nii*"))
        if seg_files:
            seg_vol = nib.load(seg_files[0]).get_fdata()
            mask_2d = seg_vol[:, :, s_idx]
            mask_slice = (mask_2d > 0).astype(np.float32)[np.newaxis, :, :]
        else:
            mask_slice = np.zeros((1, img_slice.shape[1], img_slice.shape[2]), dtype=np.float32)

        if self.transform:
            img_slice = self.transform(img_slice)

        return torch.from_numpy(img_slice.astype(np.float32)), torch.from_numpy(mask_slice.astype(np.float32)), f"{patient_id}_slice_{s_idx}"


def get_dataloaders(raw_dir="data/raw", splits_dir="data/splits", batch_size=16, transform=None):
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
            num_workers=0,
            pin_memory=False
        )

    return dataloaders
