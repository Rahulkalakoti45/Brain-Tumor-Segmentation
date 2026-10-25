"""
Brain Tumor Segmentation Web App.

Upload the four MRI modalities for one patient (T1, T1ce, T2, FLAIR as NIfTI files),
run the full pipeline, and view the predicted tumor mask over the scan.

Run from the project root:
    streamlit run app/streamlit_app.py

Research use only. Not for clinical decisions.
"""

import os
import sys
import hashlib
import tempfile
import time

import numpy as np
import nibabel as nib
import streamlit as st
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.inference import MODALITIES, segment_volume

UPLOAD_LABELS = {"t1": "T1", "t1ce": "T1ce", "t2": "T2", "flair": "FLAIR"}


def normalize_slice(img):
    """Scales a 2D slice to [0, 1] using 0.5/99.5 percentiles for display."""
    low, high = np.percentile(img, [0.5, 99.5])
    if high <= low:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - low) / (high - low), 0, 1).astype(np.float32)


def overlay_mask(img, mask, alpha=0.4):
    """Returns an RGB image with the tumor mask tinted red."""
    gray = normalize_slice(img)
    rgb = np.stack([gray, gray, gray], axis=-1)
    red = np.array([1.0, 0.0, 0.0])
    rgb[mask > 0] = (1 - alpha) * rgb[mask > 0] + alpha * red
    return np.clip(rgb, 0, 1)


def load_nifti_from_upload(uploaded_file, workdir, modality):
    """Writes an uploaded NIfTI to disk and loads it with nibabel."""
    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
    path = os.path.join(workdir, f"{modality}{suffix}")
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return nib.load(path)


def run_pipeline(uploads, config_path, checkpoint_path):
    """Runs segmentation on the uploaded files and returns a result dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with tempfile.TemporaryDirectory() as workdir:
        images = {m: load_nifti_from_upload(uploads[m], workdir, m) for m in MODALITIES}

        shapes = {m: img.shape for m, img in images.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"All four scans must have the same shape. Got: {shapes}")
        if len(next(iter(shapes.values()))) != 3:
            raise ValueError("Each scan must be a 3D volume.")

        affine = images["flair"].affine
        zooms = images["flair"].header.get_zooms()[:3]
        mri_4d = np.stack([images[m].get_fdata(dtype=np.float32) for m in MODALITIES], axis=0)

        progress = st.progress(0.0, text="Segmenting slices...")
        start = time.time()
        mask, checkpoint_loaded = segment_volume(
            mri_4d,
            config,
            checkpoint_path=checkpoint_path,
            progress_callback=lambda p: progress.progress(p, text=f"Segmenting slices... {int(p * 100)}%"),
        )
        elapsed = time.time() - start
        progress.empty()

        out_path = os.path.join(workdir, "prediction.nii.gz")
        nib.save(nib.Nifti1Image(mask, affine), out_path)
        with open(out_path, "rb") as f:
            mask_bytes = f.read()

    return {
        "mask": mask,
        "mri": {m: mri_4d[i] for i, m in enumerate(MODALITIES)},
        "voxel_volume_mm3": float(np.prod(zooms)),
        "checkpoint_loaded": checkpoint_loaded,
        "elapsed": elapsed,
        "mask_bytes": mask_bytes,
    }


def upload_key(uploads, config_path, checkpoint_path):
    """Identifies a run by its inputs so results survive reruns (for example, slider moves)."""
    digest = hashlib.md5()
    for m in MODALITIES:
        digest.update(uploads[m].getvalue())
    digest.update(config_path.encode())
    digest.update(checkpoint_path.encode())
    return digest.hexdigest()


def show_results(result):
    mask = result["mask"]
    tumor_voxels = int(mask.sum())
    tumor_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]

    if not result["checkpoint_loaded"]:
        st.warning(
            "No trained checkpoint was loaded, so this mask comes from untrained weights "
            "and does not reflect tumor location. Train a model or set the checkpoint path in the sidebar."
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tumor voxels", f"{tumor_voxels:,}")
    col2.metric("Tumor volume", f"{tumor_voxels * result['voxel_volume_mm3'] / 1000:.2f} cm³")
    col3.metric("Slices with tumor", f"{len(tumor_slices)} / {mask.shape[2]}")
    col4.metric("Processing time", f"{result['elapsed']:.1f} s")

    default_slice = int(np.argmax(mask.sum(axis=(0, 1)))) if tumor_voxels > 0 else mask.shape[2] // 2
    slice_idx = st.slider("Slice", 0, mask.shape[2] - 1, default_slice)

    left, right = st.columns(2)
    for col, modality in zip((left, right), ("flair", "t1ce")):
        img = result["mri"][modality][:, :, slice_idx]
        view = np.rot90(overlay_mask(img, mask[:, :, slice_idx]))
        col.image(view, caption=f"{UPLOAD_LABELS[modality]} with predicted tumor (red)", use_container_width=True)

    st.download_button(
        "Download predicted mask (NIfTI)",
        data=result["mask_bytes"],
        file_name="prediction.nii.gz",
        mime="application/octet-stream",
    )


def main():
    st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
    st.title("Brain Tumor Segmentation")
    st.caption("Hybrid filtering, 2D U-Net, and SAM 2 refinement on multimodal MRI. Research use only, not for clinical decisions.")

    with st.sidebar:
        st.header("Settings")
        config_path = st.text_input("Config file", os.path.join(PROJECT_ROOT, "configs", "full_pipeline.yaml"))
        checkpoint_path = st.text_input(
            "Trained U-Net checkpoint",
            os.path.join(PROJECT_ROOT, "checkpoints", "hybrid_filter", "best_model.pth"),
        )

    if not os.path.exists(config_path):
        st.error(f"Config file not found: {config_path}")
        return

    st.subheader("1. Upload the four MRI scans for one patient")
    st.write("Each file is a 3D NIfTI volume (`.nii` or `.nii.gz`) with the same dimensions.")
    columns = st.columns(len(MODALITIES))
    uploads = {}
    for col, modality in zip(columns, MODALITIES):
        uploads[modality] = col.file_uploader(UPLOAD_LABELS[modality], type=["nii", "gz"], key=f"up_{modality}")

    ready = all(uploads[m] is not None for m in MODALITIES)
    st.subheader("2. Run segmentation")
    if st.button("Run segmentation", type="primary", disabled=not ready):
        try:
            key = upload_key(uploads, config_path, checkpoint_path)
            st.session_state["result"] = run_pipeline(uploads, config_path, checkpoint_path)
            st.session_state["result_key"] = key
        except Exception as exc:
            st.session_state.pop("result", None)
            st.error(f"Segmentation failed: {exc}")

    if not ready:
        st.info("Upload all four scans to enable segmentation.")

    result = st.session_state.get("result")
    if result is not None and ready and st.session_state.get("result_key") == upload_key(uploads, config_path, checkpoint_path):
        st.subheader("3. Results")
        show_results(result)


if __name__ == "__main__":
    main()
