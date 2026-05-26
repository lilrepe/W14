"""
MFE-FCGCN — Stage 1: Raw EEG Preprocessing
Paper: Multi-frequency EEG and multi-functional connectivity graph convolutional network
       based detection method of patients with Alzheimer's disease
DOI:   10.1007/s40747-025-01974-x

This script corresponds to the paper's preprocessing pipeline described in §3.1–3.2:
  1. Load raw EEG .set files (BIDS derivative format, OpenNeuro ds004504).
  2. For each of the 5 frequency bands, compute PSD via Welch's method.
  3. Construct windowed samples by averaging consecutive PSD segments.
  4. For each sample, compute:
       - z-scored PSD feature matrix (n_psd_bins × n_channels)
       - mutual-information (MI) adjacency matrix (n_channels × n_channels)
       - Pearson-correlation (PC) adjacency matrix (n_channels × n_channels)
  5. Write CSVs for each subject/band/sample to the processed_root directory.

Usage:
    python -m src.preprocessing --config configs/base.yaml

Output directory structure (under processed_root):
    sub-<ID>/
        feature_matrix/
            sub-<BAND>-<sample_idx>-<sub_id>.csv
        adjacent_matrix/
            mul-mat-<BAND>-<sample_idx>-<sub_id>.csv
            person-mat-<BAND>-<sample_idx>-<sub_id>.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd

from src.utils import (
    build_subject_list,
    compute_mi_matrix,
    compute_pc_matrix,
    feature_matrix_path,
    load_config,
    mi_matrix_path,
    pc_matrix_path,
    raw_eeg_path,
    zscore,
)

mne.set_log_level("WARNING")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core per-subject processing
# ---------------------------------------------------------------------------

def process_subject(
    sub_id: int,
    label: int,
    cfg: dict,
) -> Tuple[int, int]:
    """
    §3.1–3.2 — Process one subject: compute PSD features and connectivity matrices
    for all 5 frequency bands, writing one CSV triple per windowed sample.

    Args:
        sub_id:  Integer subject ID (1-indexed, matching BIDS naming).
        label:   Ground-truth label (1=AD, 0=HC).
        cfg:     Loaded YAML configuration dict.

    Returns:
        (n_samples_written, label) — number of windowed samples saved for this subject.
    """
    channels: List[str] = cfg["channels"]  # §3.1 — 19 EEG channels
    n_channels: int = cfg["n_channels"]

    raw_root: str = cfg["raw_dataset_root"]
    processed_root: str = cfg["processed_root"]

    # §4.1 — per-class sample cap
    max_samples: int = (
        cfg["sampling"]["max_samples_ad"] if label == 1
        else cfg["sampling"]["max_samples_hc"]
    )
    window_size: int = cfg["sampling"]["window_size"]   # §3.2 — 4 consecutive segments
    stride: int = cfg["sampling"]["stride"]              # [UNSPECIFIED] see configs/base.yaml

    psd_cfg = cfg["psd"]
    bands: Dict[str, List[float]] = cfg["frequency_bands"]  # §3.1 — 5 bands

    set_path = raw_eeg_path(raw_root, sub_id)
    if not set_path.exists():
        log.warning(f"[sub-{sub_id:03d}] .set file not found at {set_path}. Skipping.")
        return 0, label

    # Load raw EEG — MNE reads EEGLAB .set format
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True)
    sfreq: float = raw.info["sfreq"]

    # Ensure output directories exist
    from src.utils import processed_subject_root
    sub_root = processed_subject_root(processed_root, sub_id)
    (sub_root / "feature_matrix").mkdir(parents=True, exist_ok=True)
    (sub_root / "adjacent_matrix").mkdir(parents=True, exist_ok=True)

    # Per-band PSD computation — §3.1
    # Welch's method (n_fft, n_overlap) applied independently per band.
    band_psds: Dict[str, Dict[str, np.ndarray]] = {}
    for band_name, (fmin, fmax) in bands.items():
        ch_psds: Dict[str, np.ndarray] = {}
        for ch in channels:
            data, _ = raw[ch, :]            # shape: (1, n_times)
            # §3.1 — psd_array_welch with average=None returns per-segment PSDs
            # shape: (1, n_freqs, n_segments)
            psds, _ = mne.time_frequency.psd_array_welch(
                data,
                sfreq,
                fmin=fmin,
                fmax=fmax,
                n_fft=psd_cfg["n_fft"],
                average=psd_cfg["average"],   # None → return all segments
                n_overlap=psd_cfg["n_overlap"],
            )
            # psds[0] shape: (n_freqs, n_segments) — squeeze batch dim
            ch_psds[ch] = psds[0].T  # → (n_segments, n_freqs)
        band_psds[band_name] = ch_psds

    # Label row — same scalar for all 19 channels
    label_row = pd.DataFrame(
        {ch: [label] for ch in channels}
    )

    # Windowed sample construction — §3.2
    # Determine reference number of segments (all channels/bands should agree)
    ref_n_segments = band_psds[list(bands.keys())[0]][channels[0]].shape[0]
    n_samples_written = 0

    sample_idx = 1  # 1-indexed to match paper's file naming convention
    i = 0           # segment pointer

    while i < ref_n_segments - window_size + 1:
        if n_samples_written >= max_samples:
            break

        # §3.2 — average window_size consecutive PSD segments
        for band_name in bands.keys():
            psd_values: Dict[str, np.ndarray] = {}
            for ch in channels:
                seg_window = band_psds[band_name][ch][i : i + window_size]  # (window_size, n_freqs)
                seg_mean = seg_window.mean(axis=0)                           # (n_freqs,)
                psd_values[ch] = zscore(seg_mean)                            # §3.2 — z-score normalise

            # Feature DataFrame: shape (n_freqs, n_channels) — each row = one PSD bin
            df_features = pd.DataFrame(psd_values, columns=channels)

            # Connectivity matrices — §3.2
            # MI: nonlinear, row-max normalised
            df_connectivity = pd.DataFrame(psd_values, columns=channels)
            df_mi = compute_mi_matrix(df_connectivity)

            # PC: linear, row-sum-abs normalised, diagonal preserved
            df_pc_base = pd.DataFrame(psd_values, columns=channels)
            df_pc = compute_pc_matrix(df_pc_base)

            # Append label row to feature matrix (last row)
            all_psd_csv = pd.concat([df_features, label_row], axis=0, ignore_index=True)

            # Write CSVs
            fmat_path = feature_matrix_path(processed_root, sub_id, band_name, sample_idx)
            mi_path = mi_matrix_path(processed_root, sub_id, band_name, sample_idx)
            pc_path = pc_matrix_path(processed_root, sub_id, band_name, sample_idx)

            all_psd_csv.to_csv(str(fmat_path))
            df_mi.to_csv(str(mi_path))
            df_pc.to_csv(str(pc_path))

        n_samples_written += 1
        sample_idx += 1
        i += stride

    log.info(
        f"[sub-{sub_id:03d}] label={label} → {n_samples_written} samples "
        f"(cap={max_samples})"
    )
    return n_samples_written, label


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_preprocessing(cfg: dict) -> None:
    """
    Process all subjects listed in the config and save intermediate CSVs.

    §4.1 — 65 subjects total (36 AD + 29 HC after excluding FTD).
    """
    subject_list = build_subject_list(cfg)

    total_ad, total_hc, total_all = 0, 0, 0
    for sub_id, label in subject_list:
        n, lbl = process_subject(sub_id, label, cfg)
        total_all += n
        if lbl == 1:
            total_ad += n
        else:
            total_hc += n

    log.info(
        f"Preprocessing complete. "
        f"Total samples: {total_all} "
        f"(AD: {total_ad}, HC: {total_hc})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MFE-FCGCN preprocessing")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_preprocessing(cfg)
