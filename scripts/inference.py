"""
End-to-End Inference Script for New Multimodal MRI Patient Scans.

Usage:
    python scripts/inference.py --subject-dir data/raw/BraTS2021_00000 --config configs/full_pipeline.yaml
"""

import os
import sys
import glob
import argparse
import logging
import yaml
import numpy as np
import nibabel as nib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import MODALITIES, segment_volume

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("InferenceScript")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Full Tumor Segmentation Inference on Patient Volume")
    parser.add_argument("--subject-dir", type=str, required=True, help="Path to patient NIfTI directory")
    parser.add_argument("--config", type=str, default="configs/full_pipeline.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/predictions", help="Directory to save predicted NIfTI mask")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Verify input patient files
    if not os.path.exists(args.subject_dir):
        logger.error(f"Subject directory '{args.subject_dir}' does not exist!")
        sys.exit(1)

    patient_id = os.path.basename(os.path.normpath(args.subject_dir))
    logger.info(f"=== Starting Tumor Segmentation Inference on Patient '{patient_id}' ===")

    mod_files = {}
    for mod in MODALITIES:
        matches = glob.glob(os.path.join(args.subject_dir, f"*{mod}.nii*"))
        if not matches:
            logger.error(f"Missing required modality '{mod}' in '{args.subject_dir}'!")
            sys.exit(1)
        mod_files[mod] = matches[0]

    # Load 3D volumes
    ref_nifti = nib.load(mod_files["flair"])
    affine = ref_nifti.affine
    logger.info(f"Patient volume dimensions: {ref_nifti.shape}")

    mri_4d = np.stack([nib.load(mod_files[m]).get_fdata() for m in MODALITIES], axis=0)

    checkpoint_path = args.checkpoint or os.path.join(
        config.get("output", {}).get("checkpoint_dir", "checkpoints/hybrid_filter"), "best_model.pth"
    )
    predicted_3d_mask, checkpoint_loaded = segment_volume(mri_4d, config, checkpoint_path)
    if not checkpoint_loaded:
        logger.warning("No trained checkpoint was loaded: the output is from untrained weights and is not meaningful.")

    # Save output 3D NIfTI segmentation mask
    out_file = os.path.join(args.output_dir, f"{patient_id}_pred_seg.nii.gz")
    nib.save(nib.Nifti1Image(predicted_3d_mask, affine), out_file)

    logger.info(f"=== Inference Complete ===")
    logger.info(f"Saved 3D segmentation mask NIfTI file: '{out_file}'")

if __name__ == "__main__":
    main()
