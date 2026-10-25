# Brain Tumor Segmentation Using Hybrid Filtering with U-Net Architecture in Multimodal MRI Volumes

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, modular, executable academic research framework for multimodal MRI brain tumor segmentation combining bilateral filtering, black top-hat morphological enhancement, 2D U-Net segmentation, automatic prompt generation, and SAM 2 (Segment Anything 2) Tiny boundary refinement.

---

## 1. Overview & Research Motivation
Accurate delineation of brain tumor boundaries from multimodal Magnetic Resonance Imaging (MRI) is essential for effective neuro-surgical planning, radiotherapy targeting, and patient assessment. Medical MRI volumes present significant challenges:
- High variability in tumor shape, location, and intensity contrast.
- Fuzzy or subtle tumor boundaries between healthy tissue and peritumoral edema.
- Complex subregion structures (Whole Tumor, Tumor Core, Enhancing Tumor).

This project investigates a hybrid pipeline combining classical non-linear spatial filtering (Bilateral + Black Top-Hat) to accentuate subtle anatomical boundaries, a deep 2D U-Net architecture for semantic segmentation, automatic bounding box and centroid prompt extraction, and a prompt-guided SAM 2 Tiny boundary refinement stage.

---

## 2. Project Directory Structure

```
Brain_Tumor_Segmentation/
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
│   │   ├── dataset.py
│   │   ├── nifti_loader.py
│   │   └── validation.py
│   │
│   ├── preprocessing/
│   │   ├── normalization.py
│   │   ├── preprocessing.py
│   │   └── patch_extraction.py
│   │
│   ├── filtering/
│   │   ├── bilateral_filter.py
│   │   ├── black_tophat.py
│   │   └── hybrid_filter.py
│   │
│   ├── models/
│   │   └── unet.py
│   │
│   ├── losses/
│   │   └── segmentation_losses.py
│   │
│   ├── metrics/
│   │   └── segmentation_metrics.py
│   │
│   ├── prompt_generation/
│   │   └── automatic_prompt.py
│   │
│   ├── refinement/
│   │   └── sam2_refiner.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   ├── evaluation/
│   │   └── evaluator.py
│   │
│   └── visualization/
│       └── visualize.py
│
├── scripts/
│   ├── 01_validate_dataset.py
│   ├── 02_visualize_dataset.py
│   ├── 03_preprocess.py
│   ├── 04_test_filtering.py
│   ├── 05_train_baseline.py
│   ├── 06_train_hybrid.py
│   ├── 07_generate_prompts.py
│   ├── 08_run_sam2.py
│   ├── 09_evaluate.py
│   └── 10_run_full_pipeline.py
│
├── experiments/
│   ├── baseline_unet/
│   ├── bilateral_unet/
│   ├── black_tophat_unet/
│   ├── hybrid_filter_unet/
│   └── full_pipeline/
│
├── checkpoints/
│
├── results/
│   ├── figures/
│   ├── predictions/
│   ├── metrics/
│   └── tables/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_filtering.ipynb
│   └── 04_results_analysis.ipynb
│
└── tests/
    ├── test_dataset.py
    ├── test_preprocessing.py
    ├── test_filtering.py
    ├── test_unet.py
    ├── test_metrics.py
    └── test_prompt_generation.py
```

---

## 3. Installation & Virtual Environment Setup

```powershell
# 1. Activate Python 3.11 environment
.\.venv\Scripts\activate

# 2. Install required dependencies
pip install -r requirements.txt
```

---

## 4. Pipeline Execution Scripts

Run the pipeline steps sequentially or run the end-to-end master script:

```powershell
# Step 01: Audit dataset integrity and file health
python scripts/01_validate_dataset.py

# Step 02: Visualize multimodal dataset MRI slices
python scripts/02_visualize_dataset.py

# Step 03: Run preprocessing (Clipping, Normalization, Resizing)
python scripts/03_preprocess.py

# Step 04: Benchmark Bilateral + Black Top-Hat Hybrid Filtering
python scripts/04_test_filtering.py

# Step 05: Train Baseline 2D U-Net Model
python scripts/05_train_baseline.py --epochs 10

# Step 06: Train Hybrid-Filtered 2D U-Net Model
python scripts/06_train_hybrid.py --epochs 10

# Step 07: Extract Automatic Bounding Box & Centroid Prompts
python scripts/07_generate_prompts.py

# Step 08: Execute SAM 2 Tiny Boundary Refinement
python scripts/08_run_sam2.py

# Step 09: Quantitative Benchmark & Evaluation
python scripts/09_evaluate.py

# Step 10: Run Full End-to-End Pipeline
python scripts/10_run_full_pipeline.py --epochs 3
```

---

## 4.1 Web App (Upload a Scan)

Upload the four MRI modalities for one patient and view the predicted tumor mask:

```powershell
streamlit run app/streamlit_app.py
```

Then open the URL it prints. The app runs locally. Results use the trained checkpoint in `checkpoints/hybrid_filter/best_model.pth` when it exists; otherwise the app shows a warning that the mask is not meaningful.

---

## 5. Running Unit Tests

Run unit tests via Pytest or standard unittest:

```powershell
pytest tests/
```
