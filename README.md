
# What this implements

The full MFE-FCGCN pipeline — from raw EEG `.set` files to trained classifier:

1. **Multi-frequency PSD feature extraction** (5 EEG bands: Delta, Theta, Alpha, Beta, Gamma) via Welch's method.
2. **Dual functional connectivity graphs** per band: mutual information (nonlinear) + Pearson correlation (linear).
3. **GCNConv with alternating edge sets** — GCN layer 1 uses MI edges, layer 2 uses PC edges.
4. **Multi-band fusion** — concatenate 5 per-band GCN outputs → MLP classifier (AD vs HC).
5. **Nested cross-validation training** — outer 80/20 subject split, inner 15-fold CV.

Reported performance on OpenNeuro ds004504: **95.09% accuracy, 98.36% AUC**.

---

## Dataset

OpenNeuro ds004504: *A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia, and Healthy Subjects from Routine EEG.*

Download from: https://openneuro.org/datasets/ds004504/versions/1.0.8

Only AD (subjects 1–36) and HC (subjects 37–65) are used. Place the downloaded dataset at the path configured as `raw_dataset_root` in `configs/base.yaml`.

---

## Quick start

```bash
# 1. Install Python 3.11+ and dependencies
pip install -r requirements.txt

# 2. Configure paths in configs/base.yaml
#    - raw_dataset_root: path to downloaded ds004504
#    - processed_root:   e.g. D:/const/sample
#    - pt_root:          e.g. D:/const/dataset_pt
#    - (see configs/base.yaml for all options)

# 3. Preprocessing — extract PSD features and connectivity matrices
#    WARNING: Pairwise mutual information is slow (~minutes per subject).
python -m src.preprocessing --config configs/base.yaml

# 4. Graph construction — convert CSVs to PyG .pt files
python -m src.graph_builder --config configs/base.yaml

# 5. Train
python -m src.train --config configs/base.yaml
```

---

## File structure

```
mfe_fcgcn/
├── configs/
│   └── base.yaml           # All hyperparameters, cited to paper section
├── src/
│   ├── utils.py            # Path helpers, connectivity matrix functions, config loading
│   ├── preprocessing.py    # Stage 1: raw EEG → PSD features + adjacency CSVs
│   ├── graph_builder.py    # Stage 2: CSVs → PyG Data objects → .pt files
│   ├── model.py            # SimpleCNN + GCN + FiveGraphsModel architecture
│   ├── data.py             # Dataset class, data loading, train/test/CV splits
│   ├── evaluate.py         # Metrics: accuracy, sensitivity, specificity, F1, AUC
│   └── train.py            # Training loop with nested cross-validation
├── requirements.txt
├── REPRODUCTION_NOTES.md   # All ambiguities and unspecified choices documented
└── README.md
```

---

## Ambiguities and known gaps

See [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) for a full accounting of:
- Items specified in the paper vs. inferred from context
- `[UNSPECIFIED]` choices (sliding window stride, batch sizes, random seeds)
- Two base-path clarification (`raw_dataset_root` vs `processed_root`)
- CNN kernel size dependency on actual PSD bin counts

---

## Citation

```bibtex
@article{liu2025mfefcgcn,
  title   = {Multi-frequency {EEG} and multi-functional connectivity graph convolutional network
             based detection method of patients with {Alzheimer's} disease},
  author  = {Liu, Yujian and An, Libing and Yang, Haiqiang and Ge, Shuzhi Sam},
  journal = {Complex \& Intelligent Systems},
  volume  = {11},
  pages   = {366},
  year    = {2025},
  doi     = {10.1007/s40747-025-01974-x}
}
```
