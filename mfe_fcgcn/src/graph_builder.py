"""
MFE-FCGCN — Stage 2: Graph Construction
Paper: Multi-frequency EEG and multi-functional connectivity graph convolutional network
       based detection method of patients with Alzheimer's disease
DOI:   10.1007/s40747-025-01974-x

This script converts the per-sample CSV files produced by preprocessing.py into
PyTorch Geometric Data objects and saves them as .pt files (one per subject).

§3.2–3.3 — For each windowed sample, five per-band subgraphs are constructed:
  - Node features: z-scored PSD feature matrix, shape (n_channels, n_psd_bins)
  - Edge set 1 (MI):  mutual-information adjacency → (edge_index, edge_attr)
  - Edge set 2 (PC):  Pearson-correlation adjacency → (edge_index_two, edge_weight_two)
  - Label: y (0 or 1)

The five subgraphs for one sample are stored as a Python list of Data objects
and saved to: <pt_root>/sub-<sub_id>/graph_<sub_id>.pt

Usage:
    python -m src.graph_builder --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.utils import (
    build_subject_list,
    count_feature_files,
    feature_matrix_path,
    load_config,
    mi_matrix_path,
    pc_matrix_path,
    pt_graph_path,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BAND_ORDER = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]  # §3.1 — fixed band ordering


# ---------------------------------------------------------------------------
# Adjacency matrix → sparse edge representation
# ---------------------------------------------------------------------------

def adj_to_edge_index_and_weight(
    adj_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    §3.3 — Convert a dense adjacency matrix DataFrame to COO edge representation.

    All non-zero entries become edges (dense graph).
    Zero entries in the normalised matrices arise only when connectivity is truly 0.

    Args:
        adj_df: Square DataFrame of shape (n_channels, n_channels).

    Returns:
        edge_index: LongTensor of shape (2, n_edges).
        edge_weight: FloatTensor of shape (n_edges, 1).
    """
    adj = torch.tensor(adj_df.values, dtype=torch.float64)

    # COO: indices of all non-zero entries — §3.3
    nonzero_idx = torch.tensor(
        np.transpose(np.nonzero(adj.numpy())), dtype=torch.long
    )  # shape: (n_edges, 2)

    if nonzero_idx.numel() == 0:
        # Degenerate case: completely disconnected graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0, 1), dtype=torch.float64)
        return edge_index, edge_weight

    edge_index = nonzero_idx.T  # shape: (2, n_edges)
    edge_weight = adj[edge_index[0], edge_index[1]].unsqueeze(-1)  # (n_edges, 1)

    return edge_index, edge_weight


# ---------------------------------------------------------------------------
# Single-sample graph construction
# ---------------------------------------------------------------------------

def build_sample_graphs(
    sub_id: int,
    sample_idx: int,
    processed_root: str,
    label_value: int,
) -> List[Data]:
    """
    §3.2–3.3 — Build five per-band PyG Data objects for one windowed sample.

    Each Data object has:
        x              : FloatTensor (n_channels, n_psd_bins) — PSD node features
        edge_index     : LongTensor  (2, n_edges_mi)          — MI edge set (nonlinear)
        edge_attr      : FloatTensor (n_edges_mi, 1)          — MI edge weights
        edge_index_two : LongTensor  (2, n_edges_pc)          — PC edge set (linear)
        edge_weight_two: FloatTensor (n_edges_pc, 1)          — PC edge weights
        y              : LongTensor  (scalar)                  — class label

    Args:
        sub_id:         Integer subject ID.
        sample_idx:     1-indexed window number for this subject.
        processed_root: Root directory of CSV outputs from preprocessing.py.
        label_value:    Ground-truth label (1=AD, 0=HC).

    Returns:
        List of 5 Data objects, one per frequency band (Delta→Gamma order).
    """
    per_band_graphs: List[Data] = []

    for band_name in BAND_ORDER:
        # --- Load feature matrix ---
        fmat_path = feature_matrix_path(processed_root, sub_id, band_name, sample_idx)
        all_cols = pd.read_csv(fmat_path, nrows=0).columns.tolist()
        use_cols = all_cols[1:]  # skip index column

        # §3.2 — feature rows = PSD bins (all rows except last label row)
        n_rows_total = len(pd.read_csv(fmat_path))
        n_feature_rows = n_rows_total - 1  # last row = label

        df_features = pd.read_csv(
            fmat_path, nrows=n_feature_rows, usecols=use_cols
        )
        # x shape: (n_channels, n_psd_bins) — transpose since df is (n_bins, n_channels)
        x = torch.tensor(df_features.values.T, dtype=torch.float32)  # (19, n_bins)

        # --- Read label (from last row, col index 1) ---
        y_val = pd.read_csv(
            fmat_path, skiprows=n_feature_rows, nrows=1, usecols=[1]
        ).values[0, 0]
        y = torch.tensor(int(y_val), dtype=torch.long)

        # --- Load MI adjacency matrix (nonlinear) ---
        mi_path = mi_matrix_path(processed_root, sub_id, band_name, sample_idx)
        mi_cols = pd.read_csv(mi_path, nrows=0).columns.tolist()[1:]
        df_mi = pd.read_csv(mi_path, usecols=mi_cols)
        edge_index_mi, edge_weight_mi = adj_to_edge_index_and_weight(df_mi)

        # --- Load PC adjacency matrix (linear) ---
        pc_path = pc_matrix_path(processed_root, sub_id, band_name, sample_idx)
        pc_cols = pd.read_csv(pc_path, nrows=0).columns.tolist()[1:]
        df_pc = pd.read_csv(pc_path, usecols=pc_cols)
        edge_index_pc, edge_weight_pc = adj_to_edge_index_and_weight(df_pc)

        # --- Assemble PyG Data object ---
        # §3.3 — store both edge sets in a single Data object using custom attributes
        data = Data(
            x=x,
            edge_index=edge_index_mi,
            edge_attr=edge_weight_mi,  # MI weights on primary edge_index
            y=y,
        )
        # Custom attributes for the PC edge set — §3.3
        data.edge_index_two = edge_index_pc
        data.edge_weight_two = edge_weight_pc

        per_band_graphs.append(data)

    return per_band_graphs  # list of 5 Data objects


# ---------------------------------------------------------------------------
# Per-subject graph serialisation
# ---------------------------------------------------------------------------

def build_and_save_subject_graphs(sub_id: int, label: int, cfg: dict) -> int:
    """
    Build graphs for all samples of one subject and save to a .pt file.

    Returns:
        Number of samples (graph-lists) saved.
    """
    processed_root: str = cfg["processed_root"]
    pt_root: str = cfg["pt_root"]

    n_samples = count_feature_files(processed_root, sub_id)
    if n_samples == 0:
        log.warning(f"[sub-{sub_id:03d}] No feature files found. Run preprocessing first.")
        return 0

    all_sample_graphs: List[List[Data]] = []
    for sample_idx in range(1, n_samples + 1):
        graphs = build_sample_graphs(sub_id, sample_idx, processed_root, label)
        all_sample_graphs.append(graphs)

    # Ensure output directory exists
    out_path = pt_graph_path(pt_root, sub_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(all_sample_graphs, str(out_path))
    log.info(f"[sub-{sub_id:03d}] saved {n_samples} samples → {out_path}")
    return n_samples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_graph_building(cfg: dict) -> None:
    """Build and save graphs for all subjects."""
    subject_list = build_subject_list(cfg)
    for sub_id, label in subject_list:
        build_and_save_subject_graphs(sub_id, label, cfg)
    log.info("Graph construction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MFE-FCGCN graph builder")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_graph_building(cfg)
