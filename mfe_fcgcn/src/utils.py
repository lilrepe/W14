"""
MFE-FCGCN — Shared Utilities
Paper: Multi-frequency EEG and multi-functional connectivity graph convolutional network
       based detection method of patients with Alzheimer's disease
DOI:   10.1007/s40747-025-01974-x
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = "configs/base.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Subject path helpers
# ---------------------------------------------------------------------------

def subject_id_to_bids_label(sub_id: int) -> str:
    """
    Convert integer subject ID to zero-padded BIDS label string.

    §4.1 — Dataset uses sub-001 … sub-065 naming (three-digit zero-padding).

    Examples:
        1  → "sub-001"
        10 → "sub-010"
        65 → "sub-065"
    """
    return f"sub-{sub_id:03d}"


def raw_eeg_path(raw_root: str, sub_id: int) -> Path:
    """
    Return the path to the preprocessed .set file for a subject.

    §4.1 — Dataset is OpenNeuro ds004504.
    Expected BIDS-derivative structure:
        <raw_root>/derivatives/<bids_label>/eeg/<bids_label>_task-eyesclosed_eeg.set
    """
    label = subject_id_to_bids_label(sub_id)
    return Path(raw_root) / "derivatives" / label / "eeg" / f"{label}_task-eyesclosed_eeg.set"


def processed_subject_root(processed_root: str, sub_id: int) -> Path:
    """
    Return the per-subject directory under processed_root.

    §3.2 / paper implementation — samples are stored per-subject.
    Sub-directory naming follows the BIDS zero-padded format.
    """
    label = subject_id_to_bids_label(sub_id)
    return Path(processed_root) / label


def feature_matrix_path(processed_root: str, sub_id: int, band_name: str, sample_idx: int) -> Path:
    """
    §3.2 — PSD feature matrix CSV for one sample (band, subject, window index).

    Filename convention: sub-<BAND>-<sample_idx>-<sub_id>.csv
    Contains: (n_psd_bins, n_channels) feature rows + one label row.
    """
    sub_root = processed_subject_root(processed_root, sub_id)
    return sub_root / "feature_matrix" / f"sub-{band_name}-{sample_idx}-{sub_id}.csv"


def mi_matrix_path(processed_root: str, sub_id: int, band_name: str, sample_idx: int) -> Path:
    """
    §3.2 — Mutual-information adjacency matrix CSV.

    Filename convention: mul-mat-<BAND>-<sample_idx>-<sub_id>.csv
    Shape: (n_channels, n_channels), row-max normalised.
    """
    sub_root = processed_subject_root(processed_root, sub_id)
    return sub_root / "adjacent_matrix" / f"mul-mat-{band_name}-{sample_idx}-{sub_id}.csv"


def pc_matrix_path(processed_root: str, sub_id: int, band_name: str, sample_idx: int) -> Path:
    """
    §3.2 — Pearson-correlation adjacency matrix CSV.

    Filename convention: person-mat-<BAND>-<sample_idx>-<sub_id>.csv
    Shape: (n_channels, n_channels), row-sum-abs normalised (diagonal preserved).
    """
    sub_root = processed_subject_root(processed_root, sub_id)
    return sub_root / "adjacent_matrix" / f"person-mat-{band_name}-{sample_idx}-{sub_id}.csv"


def pt_graph_path(pt_root: str, sub_id: int) -> Path:
    """
    Path to the serialised PyG graph list for one subject.

    §3.3 / paper implementation — graphs saved as .pt files, one per subject.
    """
    return Path(pt_root) / f"sub-{sub_id}" / f"graph_{sub_id}.pt"


# ---------------------------------------------------------------------------
# Dataset enumeration helpers
# ---------------------------------------------------------------------------

def build_subject_list(cfg: dict) -> List[Tuple[int, int]]:
    """
    Build list of (sub_id, label) tuples from config.

    §4.1 — AD patients: sub_id 1–36 (label=1), HC: sub_id 37–65 (label=0).

    Returns:
        List of (sub_id: int, label: int) tuples.
    """
    subjects = cfg["subjects"]
    ad_lo, ad_hi = subjects["ad_range"]
    hc_lo, hc_hi = subjects["hc_range"]

    result: List[Tuple[int, int]] = []
    for sid in range(ad_lo, ad_hi + 1):
        result.append((sid, 1))  # label=1 → AD
    for sid in range(hc_lo, hc_hi + 1):
        result.append((sid, 0))  # label=0 → HC
    return result


def count_feature_files(processed_root: str, sub_id: int) -> int:
    """
    Count the number of per-band feature matrix CSVs for one subject.

    §3.2 — each sample produces 5 files (one per band); total divided by 5 gives sample count.
    """
    fm_dir = processed_subject_root(processed_root, sub_id) / "feature_matrix"
    if not fm_dir.exists():
        return 0
    total_files = sum(1 for f in fm_dir.iterdir() if f.is_file())
    return total_files // 5  # 5 bands per sample — §3.1


# ---------------------------------------------------------------------------
# Connectivity matrix helpers
# ---------------------------------------------------------------------------

def compute_mi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    §3.2 — "mutual information between each pair of EEG channels, row-max normalised"

    Computes pairwise mutual information using sklearn's mutual_info_regression,
    then normalises each row by its maximum value so entries lie in [0, 1].

    Args:
        df: DataFrame of shape (n_samples, n_channels), where rows = PSD bins,
            columns = channel names.

    Returns:
        DataFrame of shape (n_channels, n_channels) — normalised MI adjacency matrix.

    Note:
        [UNSPECIFIED] Paper does not specify the exact MI estimator (k-NN vs histogram).
        Using: sklearn.feature_selection.mutual_info_regression with discrete_features='auto'.
        Alternatives: histogram-based estimators, MINE.
    """
    from sklearn.feature_selection import mutual_info_regression  # lazy import

    channels = df.columns
    n = len(channels)
    mi_raw = np.zeros((n, n))

    for i, col1 in enumerate(channels):
        for j, col2 in enumerate(channels):
            mi_val = mutual_info_regression(
                df[[col1]], df[col2], discrete_features="auto"
            )
            mi_raw[i, j] = mi_val[0]

    # Row-max normalisation — §3.2
    row_maxes = mi_raw.max(axis=1, keepdims=True)
    mi_norm = np.divide(mi_raw, row_maxes, where=row_maxes != 0)

    return pd.DataFrame(mi_norm, columns=channels, index=channels)


def compute_pc_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    §3.2 — "Pearson correlation adjacency matrix, absolute values,
    row-sum normalised; diagonal preserved from original correlation."

    Args:
        df: DataFrame of shape (n_samples, n_channels).

    Returns:
        DataFrame of shape (n_channels, n_channels) — normalised PC adjacency matrix.
    """
    corr: pd.DataFrame = df.corr()  # Pearson correlation — §3.2

    proc = corr.copy().abs()
    np.fill_diagonal(proc.values, np.nan)

    for i in range(len(proc)):
        row_sum = proc.iloc[i].sum()  # ignores NaN diagonal
        if row_sum != 0:
            proc.iloc[i] = proc.iloc[i] / row_sum

    # Restore original diagonal — §3.2
    for i in range(len(proc)):
        proc.iloc[i, i] = corr.iloc[i, i]

    return proc


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def zscore(arr: np.ndarray) -> np.ndarray:
    """
    §3.2 — "PSD features are z-score normalised per channel per sample."

    Args:
        arr: 1-D or 2-D numpy array.

    Returns:
        Z-scored array (mean=0, std=1). If std==0, returns zeros.
    """
    mu = arr.mean()
    sigma = arr.std()
    if sigma == 0:
        return np.zeros_like(arr)
    return (arr - mu) / sigma
