# Project Plan: Brain Tumor Segmentation Using Hybrid Filtering with U-Net Architecture in Multimodal MRI Volumes

## 1. Problem Statement
Brain tumor segmentation from multimodal Magnetic Resonance Imaging (MRI) is a critical task in clinical neuro-oncology for diagnosis, surgical planning, and post-treatment monitoring. However, accurate automated segmentation remains challenging due to:
- Low inter-class contrast between healthy brain tissue, edema, non-enhancing tumor core, and active enhancing tumor regions.
- Complex, highly heterogeneous tumor morphology and irregular spatial boundaries across slices.
- Noise and intensity inhomogeneities across multimodal MRI acquisitions (T1, T1ce, T2, FLAIR).

## 2. Research Objective
The objective of this project is to design, implement, evaluate, and benchmark a complete, reproducible, modular medical image segmentation pipeline. 

The proposed pipeline investigates whether combining **Hybrid Image Filtering** (Bilateral Filter + Morphological Black Top-Hat) with **U-Net Segmentation** and a **SAM 2 (Segment Anything 2) Tiny Refinement stage** improves tumor localization and boundary delineation in multimodal MRI volumes.

> [!IMPORTANT]
> **Scientific Integrity Notice**: Performance gains will only be claimed after empirical validation through baseline comparisons and rigorous ablation studies.

## 3. Proposed Methodology
The complete end-to-end proposed architecture is:

```
Multimodal MRI (T1, T1ce, T2, FLAIR)
        ↓
Data Validation (File integrity, shape checks, range checks)
        ↓
Preprocessing (Z-score normalization, slice selection / patch extraction)
        ↓
Hybrid Image Filtering (Bilateral Filter + Black Top-Hat transform)
        ↓
U-Net Segmentation Network (Baseline 2D Slice-Based U-Net)
        ↓
Initial Tumor Mask Prediction
        ↓
Automatic Prompt Generation (Bounding boxes / point prompts from connected components)
        ↓
SAM 2 Tiny Refinement Stage (Slice-wise boundary sharpening)
        ↓
Final Refined Tumor Mask
        ↓
Evaluation & Visualization (Dice, IoU, Sensitivity, Precision, Hausdorff Distance)
```

## 4. Hardware & Technical Adaptations
Based on Phase 0 system inspection:
- **Operating System**: Windows 11 (64-bit)
- **Host GPU**: Intel Iris Xe Graphics (Integrated GPU - **No CUDA hardware available**)
- **Target Compute Mode**: CPU Execution / PyTorch CPU backend.
- **Architectural Adaptation**: Due to CPU memory and compute constraints, a **2D slice-based U-Net pipeline** operating on axial MRI slices (with patient-level volume reconstruction) is selected over full 3D dense volumes. This prevents Out-Of-Memory (OOM) crashes and provides practical training feasibility.

## 5. Dataset Requirements
- **Target Dataset**: BraTS 2021 (Brain Tumor Segmentation Challenge).
- **Modalities**: `T1`, `T1ce`, `T2`, `FLAIR` NIfTI volumes (`.nii.gz`).
- **Target Labels**: 
  - Whole Tumor (WT): Combined label of all tumor subregions.
  - Tumor Core (TC): Enhancing tumor + Necrotic/non-enhancing core.
  - Enhancing Tumor (ET): Active contrast-enhancing region.
- **Data Path**: `DATA_ROOT/raw/` (Configurable via YAML).

## 6. Software & Environment Requirements
- **Python Runtime**: Python 3.11.17 managed via `.venv` virtual environment.
- **Key Libraries**:
  - `torch`, `torchvision`: Deep learning framework and models.
  - `nibabel`, `SimpleITK`: NIfTI medical image I/O and spatial processing.
  - `opencv-python`, `scikit-image`: Image processing and morphological filters.
  - `monai`: Medical imaging loss functions and evaluation utilities.
  - `pandas`, `matplotlib`, `scipy`, `scikit-learn`: Data analysis, visualization, metrics.
  - `pyyaml`: Configuration file parsing.

## 7. Project Folder Structure
```
brain_tumor_segmentation/
│
├── README.md
├── PROJECT_PLAN.md
├── PROJECT_STATUS.md
├── EXPERIMENT_LOG.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── baseline.yaml
│   ├── hybrid_filter.yaml
│   └── full_pipeline.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── filtering/
│   ├── models/
│   ├── losses/
│   ├── metrics/
│   ├── prompt_generation/
│   ├── refinement/
│   ├── training/
│   ├── evaluation/
│   └── visualization/
│
├── scripts/
│   ├── validate_dataset.py
│   ├── preprocess_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── experiments/
├── checkpoints/
├── results/
│   ├── figures/
│   ├── tables/
│   └── predictions/
└── logs/
```

## 8. Implementation Phases & Milestones
- **Phase 0 (Environment Setup)**: Python 3.11 environment, directory tree, dependency manifests. *(Completed)*
- **Phase 1 (Dataset Validation)**: Create `scripts/validate_dataset.py` to audit NIfTI integrity, modalities, NaN/Inf values, shape consistency.
- **Phase 2 (Preprocessing Pipeline)**: Implement NIfTI loading, slice extraction, intensity normalization, and patient-level train/val/test splits.
- **Phase 3 (Hybrid Filtering Module)**: Implement Bilateral Filter + Black Top-Hat filtering module (`src/filtering/hybrid_filter.py`) with visual diagnostics.
- **Phase 4 (U-Net Baseline)**: Implement 2D slice-based U-Net architecture (`src/models/unet.py`) with configurable channels and loss functions.
- **Phase 5 (Baseline Experiment)**: Train baseline U-Net on standard preprocessed data. Record training curves, loss, Dice, IoU.
- **Phase 6 (Hybrid Filtering Experiment)**: Train U-Net on Hybrid-Filtered MRI data. Compare performance directly against baseline.
- **Phase 7 (Ablation Study)**: Compare Raw vs. Bilateral vs. Black Top-Hat vs. Hybrid Filtering.
- **Phase 8 (Automatic Prompt Generation)**: Extract bounding boxes and centroids from U-Net predictions for downstream refinement.
- **Phase 9 (SAM 2 Tiny Integration & Refinement)**: Implement slice-wise SAM 2 refinement module with clean fallbacks.
- **Phase 10 (Final Benchmark & Documentation)**: Generate final quantitative comparison tables, visual figure panels, and viva presentation materials.

## 9. Evaluation Metrics
- **Dice Similarity Coefficient (DSC)**: Overlap metric between prediction and ground truth.
- **Intersection over Union (IoU / Jaccard Index)**: Area of overlap divided by area of union.
- **Precision (Positive Predictive Value)**: Proportion of true positive segmentations.
- **Sensitivity (Recall)**: Proportion of actual tumor tissue correctly identified.
- **Hausdorff Distance (HD95)**: Maximum 95th percentile distance between surface boundaries.

## 10. Expected Outputs & Deliverables
1. Clean, modular Python package in `src/`.
2. Execution scripts in `scripts/`.
3. Configuration templates in `configs/`.
4. Automated dataset validation report.
5. Trained baseline & hybrid U-Net weights in `checkpoints/`.
6. Quantitative evaluation comparison tables in `results/tables/`.
7. Publication-grade visualization panels in `results/figures/`.
8. Complete project documentation and viva preparation guides.

## 11. Risks & Mitigations
- **Lack of Local BraTS Dataset**: The dataset is currently missing from `data/raw/`. 
  - *Mitigation*: Create dataset validation scripts and robust dummy-data mock loaders to allow 100% pipeline testing before actual dataset placement.
- **CPU Training Overhead**: Lack of discrete CUDA GPU makes 3D volume training infeasible.
  - *Mitigation*: Use 2D slice-based training, smaller batch sizes, and optimized patch/slice caching.

## 12. Next Steps
1. Execute `scripts/validate_dataset.py` to verify environment and dataset readiness.
2. Install dependencies into `.venv`.
3. Begin Phase 2 (Data Loader & Preprocessing).
