# Dissertation: Wire Rope Anomaly Detection

This repository contains a PyTorch implementation of a Teacher-Student anomaly detection framework using DINOv2, augmented with techniques from the AAND paper (*Advancing Pre-trained Teacher: Towards Robust Feature Discrepancy for Anomaly Detection*, arxiv:2405.02068).

## Features
- **Vanilla Teacher-Student Framework:** Uses a frozen DINOv2 (ViT-L/14) as a teacher and a lightweight CNN as a student.
- **AAND Augmentations:**
  - **Stage 1 (Residual Anomaly Amplification):** Fine-tunes the teacher with synthetic anomalies using a Matching-guided Residual Gate and an Attribute-scaling Residual Generator.
  - **Stage 2 (Normality Distillation):** Trains the student using a Hard Knowledge Distillation (HKD) loss to better reconstruct challenging normal patterns.

## Setup Instructions

This project uses [`uv`](https://github.com/astral-sh/uv) for fast Python environment management.

1. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

## Running Instructions

### 1. Training

To train the AAND-augmented framework (both Stage 1 and Stage 2):

```bash
uv run python train_aand.py --data_path <path_to_dataset>
```

**Options:**
- `--texture_source dtd`: Use the DTD dataset for anomaly synthesis (requires `--dtd_path`).
- `--skip_stage1`: Skip Stage 1 and only train the student (Stage 2).
- `--skip_stage2`: Only train Stage 1.

*Note: The dataset is expected to be in YOLO format with `train`, `valid`, and `test` folders containing `images` and `labels` subdirectories.*

### 2. Inference

To run inference and generate anomaly heatmaps:

```bash
# Run AAND-enhanced inference (default)
uv run python infer.py --data_path <path_to_dataset> --mode aand

# Run original vanilla inference
uv run python infer.py --data_path <path_to_dataset> --mode vanilla
```

Results (anomaly overlays and AUROC scores) will be saved in the `results/` directory.

## Notebooks

There are two deep-dive Jupyter notebooks included to help understand the feature extraction and learning mechanisms:
- `01_teacher_feature_extraction.ipynb`: Analyzes the frozen DINOv2 teacher's patch embeddings, PCA, and attention maps.
- `02_student_learning.ipynb`: Explores the student CNN's learning dynamics, feature distillation, and anomaly map generation.
