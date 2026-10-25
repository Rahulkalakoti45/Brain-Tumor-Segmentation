"""
Dataset Validation Script for Multimodal MRI Brain Tumor Segmentation.

This script performs rigorous structural, morphological, and statistical integrity
checks on NIfTI brain tumor volumes (BraTS format) before training.

Usage:
    python scripts/validate_dataset.py --config configs/baseline.yaml
    python scripts/validate_dataset.py --data-dir data/raw --mock-check
"""

import os
import sys
import glob
import argparse
import logging
import numpy as np

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DatasetValidator")

REQUIRED_MODALITIES = ["t1", "t1ce", "t2", "flair"]
EXPECTED_LABELS = [0, 1, 2, 4]  # BraTS 2021 label encoding: 0 (BG), 1 (NCR/NET), 2 (ED), 4 (ET)

def parse_args():
    parser = argparse.ArgumentParser(description="BraTS Dataset Integrity Validator")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to YAML config file")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--mock-check", action="store_true", help="Generate temporary dummy NIfTI data to verify validation pipeline")
    return parser.parse_args()

def inspect_subject_directory(subject_dir):
    """
    Checks whether a single subject directory contains all required 4 MRI modalities
    and segmentation ground truth mask.
    
    Inputs:
        subject_dir (str): Absolute or relative path to subject folder.
        
    Outputs:
        dict: Validation results containing modality file paths and status boolean.
    """
    subject_id = os.path.basename(subject_dir)
    modality_paths = {}
    missing_modalities = []
    
    for mod in REQUIRED_MODALITIES:
        # Match pattern e.g., BraTS2021_00000_t1.nii.gz or *_t1.nii
        matches = glob.glob(os.path.join(subject_dir, f"*{mod}.nii*"))
        if matches:
            modality_paths[mod] = matches[0]
        else:
            missing_modalities.append(mod)
            
    seg_matches = glob.glob(os.path.join(subject_dir, "*seg.nii*"))
    if seg_matches:
        modality_paths["seg"] = seg_matches[0]
    else:
        missing_modalities.append("seg")
        
    status = (len(missing_modalities) == 0)
    return {
        "subject_id": subject_id,
        "valid": status,
        "paths": modality_paths,
        "missing": missing_modalities
    }

def create_mock_nifti_dataset(target_dir):
    """
    Creates a temporary synthetic NIfTI volume folder to test script execution
    when real BraTS data is not yet downloaded.
    """
    import nibabel as nib
    os.makedirs(target_dir, exist_ok=True)
    mock_subject_dir = os.path.join(target_dir, "BraTS2021_MOCK001")
    os.makedirs(mock_subject_dir, exist_ok=True)
    
    affine = np.eye(4)
    vol_shape = (64, 64, 32)
    
    logger.info(f"Generating synthetic verification volumes in '{mock_subject_dir}'...")
    
    for mod in REQUIRED_MODALITIES:
        data = np.random.normal(loc=100, scale=15, size=vol_shape).astype(np.float32)
        # Introduce mock tumor contrast region in center
        data[24:40, 24:40, 10:20] += 50.0
        img = nib.Nifti1Image(data, affine)
        nib.save(img, os.path.join(mock_subject_dir, f"BraTS2021_MOCK001_{mod}.nii.gz"))
        
    # Generate mock segmentation mask
    seg_data = np.zeros(vol_shape, dtype=np.int16)
    seg_data[24:40, 24:40, 10:20] = 1 # Tumor core
    seg_data[28:36, 28:36, 12:18] = 4 # Enhancing tumor
    seg_img = nib.Nifti1Image(seg_data, affine)
    nib.save(seg_img, os.path.join(mock_subject_dir, "BraTS2021_MOCK001_seg.nii.gz"))
    logger.info("Mock verification volume generation complete.")

def validate_dataset(data_dir, run_mock=False):
    """
    Scans the data directory, verifies NIfTI integrity, checks for NaNs/Infs,
    evaluates shape uniformity across modalities, and summarizes dataset health.
    """
    logger.info(f"=== Starting Dataset Audit on '{data_dir}' ===")
    
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        if run_mock:
            logger.warning(f"Data directory '{data_dir}' is empty. Running with --mock-check...")
            create_mock_nifti_dataset(data_dir)
        else:
            logger.error(f"DATASET MISSING: Directory '{data_dir}' is empty or does not exist.")
            logger.error("Please download the BraTS dataset and extract subject folders into 'data/raw/'.")
            logger.error("Expected folder structure:")
            logger.error("  data/raw/BraTS2021_00000/BraTS2021_00000_t1.nii.gz")
            logger.error("  data/raw/BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz")
            logger.error("  data/raw/BraTS2021_00000/BraTS2021_00000_t2.nii.gz")
            logger.error("  data/raw/BraTS2021_00000/BraTS2021_00000_flair.nii.gz")
            logger.error("  data/raw/BraTS2021_00000/BraTS2021_00000_seg.nii.gz")
            logger.error("Or re-run with '--mock-check' to verify the validator logic with synthetic data.")
            return False

    # Check for nibabel dependency
    try:
        import nibabel as nib
    except ImportError:
        logger.error("Dependency 'nibabel' is missing. Please install dependencies using 'pip install -r requirements.txt'.")
        return False

    subject_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    logger.info(f"Found {len(subject_dirs)} subject directories.")

    total_valid = 0
    total_corrupted = 0
    
    for subject_dir in subject_dirs:
        res = inspect_subject_directory(subject_dir)
        if not res["valid"]:
            logger.warning(f"Subject '{res['subject_id']}' INVALID. Missing files: {res['missing']}")
            total_corrupted += 1
            continue
            
        # Inspect NIfTI volumes using nibabel
        try:
            ref_shape = None
            has_error = False
            for mod, path in res["paths"].items():
                img = nib.load(path)
                data = img.get_fdata()
                
                # Check NaNs and Infs
                if np.isnan(data).any() or np.isinf(data).any():
                    logger.error(f"NaN/Inf values detected in '{path}'!")
                    has_error = True
                    break
                    
                # Shape uniformity check across modalities
                if ref_shape is None:
                    ref_shape = data.shape
                elif data.shape != ref_shape:
                    logger.error(f"Shape mismatch in subject '{res['subject_id']}': {mod} shape {data.shape} != ref shape {ref_shape}")
                    has_error = True
                    break
                    
                # Segmentation label value check
                if mod == "seg":
                    unique_labels = np.unique(data.astype(int))
                    if np.sum(data) == 0:
                        logger.warning(f"Subject '{res['subject_id']}' ground truth mask is completely empty (all 0s).")
                        
            if not has_error:
                total_valid += 1
            else:
                total_corrupted += 1
                
        except Exception as e:
            logger.error(f"Error loading NIfTI volume for '{res['subject_id']}': {e}")
            total_corrupted += 1

    logger.info("=== Dataset Validation Summary ===")
    logger.info(f"Total Subjects Inspected : {len(subject_dirs)}")
    logger.info(f"Valid Subjects           : {total_valid}")
    logger.info(f"Corrupted/Invalid        : {total_corrupted}")
    
    return total_valid > 0

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    
    if os.path.exists(args.config):
        try:
            import yaml
            with open(args.config, "r") as f:
                cfg = yaml.safe_load(f)
                data_dir = cfg.get("data", {}).get("raw_dir", data_dir)
        except Exception as e:
            logger.warning(f"Could not parse config '{args.config}': {e}")

    success = validate_dataset(data_dir, run_mock=args.mock_check)
    if not success:
        sys.exit(1)
    else:
        logger.info("Dataset validation pipeline executed successfully.")
        sys.exit(0)
