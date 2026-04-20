# ECG Arrhythmia Benchmark — Five Model Families: Accuracy vs. Interpretability

## Overview

This repository implements a controlled beat-level ECG arrhythmia classification benchmark comparing five model families — 1D-ResNet, 1D-CNN, BiLSTM, Random Forest, and XGBoost — on the MIT-BIH Arrhythmia Database under the rigorous inter-patient De Chazal DS1/DS2 split, ensuring no patient overlap between training and test sets. SHAP GradientExplainer (deep models) and TreeExplainer (tree models) produce beat-level temporal attribution maps that localise model decisions to specific ECG waveform segments; ResNet1D Grad-CAM provides complementary spatial attention from the final residual block. The benchmark quantifies the accuracy–interpretability trade-off: tree ensembles dominate accuracy (Random Forest 93.13%, XGBoost 92.94%) while deep models offer richer minority-class sensitivity and temporally interpretable SHAP attributions tied to the P-wave, QRS complex, and T-wave.

**Paper:** *Accuracy Versus Interpretability in Inter-Patient ECG Arrhythmia Classification: A Controlled Benchmark of Five Model Families With SHAP Temporal Attribution on MIT-BIH* — submitted to IEEE Access (2026).

**Author:** S. Aljaloud, University of Ha'il, Saudi Arabia.

---

## Results Summary

### MIT-BIH DS2 Test Set (49,693 beats, inter-patient split)

| Model | Accuracy | Macro F1 | Weighted F1 | Macro AUC |
|---|---|---|---|---|
| ResNet1D | 38.72% | 0.259 | 0.509 | 0.707 |
| CNN1D | 46.04% | 0.310 | 0.584 | 0.748 |
| BiLSTM | 6.1% | 0.043 | 0.012 | 0.585 |
| **Random Forest** | **93.13%** | 0.356 | 0.910 | 0.774 |
| XGBoost | 92.94% | **0.371** | **0.914** | **0.837** |

### Key Findings

| Analysis | Finding |
|---|---|
| **Wilcoxon test (all pairwise)** | *p* < 0.001 for all comparisons |
| **Ablation — No Residual (A1)** | +26.1 pp accuracy (38.7% → 64.8%); macro F1: 0.259 → 0.353 |
| **Ablation — No BatchNorm (A3)** | 55.7% accuracy; BN is beneficial but skip connections are the primary instability source |
| **SHAP — Ventricular (V)** | Elevated attribution at QRS complex and post-QRS region |
| **SHAP — Supraventricular (S)** | Broadly distributed, higher overall attribution magnitude |

---

## Repository Structure

```
ecg-arrhythmia/
├── run_all.py               # End-to-end CLI runner (data → train → evaluate → explain → ablation)
├── src/
│   ├── config.py            # Paths, seed, AAMI map, DS1/DS2 split, hyperparameters
│   ├── data.py              # MIT-BIH download + preprocessing (bandpass → segment → z-score)
│   ├── models.py            # ResNet1D, CNN1D, BiLSTM (PyTorch); Random Forest, XGBoost (sklearn/xgboost)
│   ├── train.py             # Class-weighted CE training loop (deep) + ML fit
│   ├── evaluate.py          # Metrics, confusion matrices, ROC curves, Wilcoxon + McNemar tests
│   ├── explain.py           # SHAP GradientExplainer + TreeExplainer + Grad-CAM
│   └── ablation.py          # A1–A4 variants (NoResidual / Shallow / NoBN / NoDropout)
├── colab/
│   └── run_all.ipynb        # Google Colab one-click reproduction notebook
├── data/                    # (gitignored) raw MIT-BIH wfdb files + preprocessed NumPy arrays
├── outputs/
│   ├── tables/*.csv         # Overall, per-class, ablation, and significance results
│   ├── figures/*.png        # cm_*, roc_*, model_comparison, shap_temporal_*, gradcam_*
│   └── models/*.{pt,joblib} # Saved model weights
├── requirements.txt
└── README.md
```

---

## Reproduction

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Download and preprocess MIT-BIH Arrhythmia Database
```
python run_all.py --stop-after data
```
Downloads all 48 records from PhysioNet via `wfdb`, applies Butterworth bandpass filter (0.5–45 Hz), segments 260-sample windows centred on R-peaks, maps annotation symbols to AAMI 5-class labels (N, S, V, F, Q), z-score normalises each beat, and saves `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy` to `data/clean/`.

### 3. Train all five models and evaluate on DS2
```
python run_all.py --stop-after predict
```
Trains ResNet1D, CNN1D, BiLSTM (PyTorch, Adam + ReduceLROnPlateau, class-weighted cross-entropy), Random Forest, and XGBoost (`class_weight="balanced"`), evaluates each on DS2, and saves predictions to `outputs/`.

### 4. Generate evaluation plots and significance tables
```
python run_all.py --stop-after sig
```
Produces confusion matrices, per-class F1 bar charts, ROC curves, and Wilcoxon signed-rank and McNemar significance tables.

### 5. Run SHAP temporal attribution and Grad-CAM
```
python run_all.py --stop-after shap
```
Computes SHAP GradientExplainer maps for deep models, TreeExplainer maps for tree models (same 260-sample input space), and Grad-CAM from the ResNet1D final residual block; saves temporal attribution and beeswarm figures to `outputs/figures/`.

### 6. Run full pipeline including four-variant ablation
```
python run_all.py
```
Executes all stages end-to-end (data → train → predict → plots → sig → shap → ablation). Use `--bilstm-epochs 10` to shorten BiLSTM training time on CPU. All outputs are saved to `outputs/` and zipped as `outputs.zip`.

---

## Source Code (`src/`)

- `config.py` — project paths, random seed (42), `AAMI_MAP`, `DS1_RECORDS`/`DS2_RECORDS`, `WINDOW_SIZE` (260 samples), `SAMPLING_RATE` (360 Hz), and model hyperparameters
- `data.py` — downloads all 48 MIT-BIH records via `wfdb.dl_database`, applies Butterworth 0.5–45 Hz bandpass filter, segments R-peak-centred 260-sample windows, maps to AAMI 5 classes, z-score normalises each beat, and saves DS1/DS2 NumPy arrays
- `models.py` — defines `ResNet1D` (four residual blocks, 719 K parameters), `CNN1D` (36 K parameters), and `BiLSTM` (two-layer bidirectional, 531 K parameters) in PyTorch; wraps `RandomForestClassifier` and `XGBClassifier`
- `train.py` — class-weighted cross-entropy training loop with Adam and ReduceLROnPlateau for deep models; `class_weight="balanced"` for ML models; evaluates on DS2 and returns prediction and metric dictionaries
- `evaluate.py` — accuracy, macro/weighted precision, recall, F1, and AUC metrics; confusion matrices; ROC curves; Wilcoxon signed-rank and McNemar tests with effect sizes
- `explain.py` — unified explainability: SHAP GradientExplainer for deep models, TreeExplainer for tree models on the shared 260-sample input; Grad-CAM on ResNet1D final residual block; cross-architecture temporal attribution comparison figures
- `ablation.py` — four ResNet1D structural variants: A1 (no residual skip connections), A2 (shallow, 2 blocks), A3 (no batch normalisation), A4 (no dropout); trains each and saves per-variant metric tables

## Requirements

```
pip install -r requirements.txt
```

Key packages: `torch`, `wfdb`, `shap`, `scikit-learn`, `xgboost`, `scipy`, `numpy`, `pandas`, `matplotlib`, `seaborn`.

## Data Availability

The MIT-BIH Arrhythmia Database is publicly available at  
<https://physionet.org/content/mitdb/1.0.0/>.  
Downloaded automatically by `run_all.py` via `wfdb.dl_database`.

## License

MIT

## Contact

s.aljaloud@uoh.edu.sa
