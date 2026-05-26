# Reproduction Notes — MFE-FCGCN

**Paper:** Multi-frequency EEG and multi-functional connectivity graph convolutional network based detection method of patients with Alzheimer's disease  
**DOI:** 10.1007/s40747-025-01974-x  
**Journal:** Complex & Intelligent Systems, Vol. 11, Article 366, 2025

---

## What this implementation covers

This codebase implements the full MFE-FCGCN pipeline as described in §3:

- **Preprocessing** (`src/preprocessing.py`): raw EEG → per-band PSD features + MI/PC adjacency matrices.
- **Graph construction** (`src/graph_builder.py`): CSVs → PyG Data objects → `.pt` files.
- **Model** (`src/model.py`): SimpleCNN + two-layer GCNConv with dual edge sets + MLP classifier.
- **Training** (`src/train.py`): outer 80/20 subject split, inner 15-fold CV, Adam + ReduceLROnPlateau, best-AUC checkpointing.
- **Evaluation** (`src/evaluate.py`): accuracy, sensitivity, specificity, F1, AUC.

---

## Scope decisions

### Implemented
- Multi-frequency PSD feature extraction via Welch's method (5 bands) — §3.1
- Mutual-information adjacency matrix (nonlinear connectivity) — §3.2
- Pearson-correlation adjacency matrix (linear connectivity) — §3.2
- SimpleCNN feature compressor per band — §3.3
- Two-layer GCNConv with alternating MI/PC edge sets — §3.3
- Multi-band concatenation + MLP classifier — §3.3
- Full nested CV training loop — §4.1
- All metrics reported in Table 2/3 of the paper — §4.2

### Intentionally excluded
- FTD (frontotemporal dementia) subjects — paper uses only AD vs HC
- Baseline comparison methods (SVM, standalone GCN, etc.) — not the paper's contribution
- Ablation study variants — not needed for core reproduction

### Would need for full reproduction
- The ds004504 dataset — must be downloaded from https://openneuro.org/datasets/ds004504/versions/1.0.8
- Preprocessing is slow (~hours) due to pairwise mutual information computation; consider parallelising per-subject

---

## Specified items (directly from paper)

| Item | Value | Source |
|------|-------|--------|
| EEG channels | 19 (10-20 system: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz) | §3.1 |
| Frequency bands | Delta 0.5–4 Hz, Theta 4–8 Hz, Alpha 8–13 Hz, Beta 13–30 Hz, Gamma 30–45 Hz | §3.1 |
| PSD method | Welch (n_fft=2500, n_overlap=500) | §3.1 |
| Window size | 4 consecutive PSD segments averaged per sample | §3.2 |
| Connectivity 1 | Mutual information, row-max normalised | §3.2 |
| Connectivity 2 | Pearson correlation, absolute + row-sum normalised, diagonal preserved | §3.2 |
| GCN-1 edge set | MI (nonlinear) | §3.3 |
| GCN-2 edge set | PC (linear) | §3.3 |
| Optimiser | Adam | §4.1 |
| Learning rate | 0.00012 | §4.1 |
| Weight decay | 1e-4 | §4.1 |
| Epochs per fold | 200 | §4.1 |
| Outer split | 80/20 stratified at subject level | §4.1 |
| Inner folds | 15-fold stratified CV | §4.1 |
| Loss function | CrossEntropyLoss (2 classes) | §4.1 |
| AD subjects | sub-001 … sub-036 (label=1) | §4.1 |
| HC subjects | sub-037 … sub-065 (label=0) | §4.1 |
| Reported accuracy | 95.09% | §4.2 |
| Reported AUC | 98.36% | §4.2 |

---

## Unspecified items — choices made

### `[UNSPECIFIED]` — Sliding window stride
**What the paper says:** Window size = 4 consecutive PSD segments.  
**What is missing:** The stride between consecutive windows.  
**Choice:** stride = 1 (slide by one segment).  
**Alternatives:** stride = 4 (non-overlapping windows); this would reduce sample count significantly.  
**Impact:** Controls total sample count per subject. Affects train/test balance.

---

### `[UNSPECIFIED]` — Random seed for outer train/test split
**What the paper says:** 80/20 stratified split.  
**What is missing:** Random seed.  
**Choice:** `random_state=None` (non-deterministic). Set a fixed seed in `configs/base.yaml` for reproducibility.  
**Alternatives:** 42, 0, 1.  
**Impact:** Affects which subjects land in test set; may shift final accuracy by ±1–2%.

---

### `[UNSPECIFIED]` — MI estimator details
**What the paper says:** "mutual information" between channel pairs.  
**What is missing:** Whether k-NN, histogram, or kernel estimator is used.  
**Choice:** `sklearn.feature_selection.mutual_info_regression` with `discrete_features='auto'` (k-NN based).  
**Alternatives:** histogram MI estimator, MINE.  
**Impact:** MI estimates differ numerically across methods; ordering/ranking likely preserved.

---

### `[UNSPECIFIED]` — CNN kernel sizes / n_psd_bins per band
**What the paper says:** "a convolutional layer compresses PSD features."  
**What is missing:** The exact number of PSD frequency bins per band (depends on sfreq, n_fft, fmin, fmax).  
**Choice:** Kernel sizes {18→(1,3), 21→(1,6), 26→(1,11), 76→(1,61), 86→(1,71)} from paper implementation.  
**Action required:** Run preprocessing once and inspect actual `data.x.shape[1]` per band. If your sampling rate differs from the paper's dataset, these values will change and new Conv2d layers must be added to `SimpleCNN`.

---

### `[UNSPECIFIED]` — ReduceLROnPlateau factor and patience
**What the paper says:** Scheduler is used.  
**What is missing:** factor and patience values.  
**Choice:** factor=0.1, patience=5 (from paper implementation).  
**Alternatives:** factor=0.5, patience=10.

---

### `[UNSPECIFIED]` — Batch sizes
**What the paper says:** Not stated in the paper text.  
**Choice:** train=64, val=128, test=256 (from paper implementation).  
**Impact:** Affects BatchNorm statistics (not used here) and training speed; not expected to affect final accuracy.

---

### `[UNSPECIFIED]` — Inner fold random seed
**What the paper says:** 15-fold inner CV.  
**Choice:** `random_state=1` (from paper implementation).

---

### `[UNSPECIFIED]` — GCN input feature dimension (`max_features=16`)
**What the paper says:** "node features are compressed by CNN."  
**What is missing:** The output dimension of SimpleCNN used as GCN input.  
**Choice:** 16 (from paper implementation).  
**Note:** This is inconsistent — SimpleCNN outputs 1 feature per node, but GCNConv takes `max_features=16`. The paper implementation appears to re-use the same GCN regardless of CNN output width. Inspect `data.x` shape after `SimpleCNN.forward()` on your actual data.

---

## Two base paths — clarification

The paper's pipeline uses **two separate base paths**:

| Path constant | Purpose | Config key |
|---------------|---------|------------|
| `D:/gcn/ds004504` | Raw OpenNeuro BIDS dataset (input, read-only) | `raw_dataset_root` |
| `D:/const/sample` | Intermediate processed CSVs + graph .pt files (output) | `processed_root` |

The `sample/` sub-directory is used as the intermediate layer between `raw_data_processing.py` (producing per-subject CSVs) and `generate_graph_data.py` (consuming those CSVs). Both paths are configured in `configs/base.yaml`.

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download ds004504 from OpenNeuro and set raw_dataset_root in configs/base.yaml

# 3. Run preprocessing (slow — computes pairwise MI for all samples)
python -m src.preprocessing --config configs/base.yaml

# 4. Build PyG graph files
python -m src.graph_builder --config configs/base.yaml

# 5. Train
python -m src.train --config configs/base.yaml
```

---

## Expected results

Per §4.2 (Table 2/3), on the ds004504 dataset with the above pipeline:

| Metric | Reported |
|--------|----------|
| Accuracy | 95.09% |
| AUC | 98.36% |

Exact reproduction requires matching the outer split seed and inner fold seed. Due to `[UNSPECIFIED]` random states, results will vary run-to-run unless seeds are fixed.
