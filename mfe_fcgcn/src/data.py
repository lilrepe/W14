"""
MFE-FCGCN — Dataset and Data Loading
Paper: Multi-frequency EEG and multi-functional connectivity graph convolutional network
       based detection method of patients with Alzheimer's disease
DOI:   10.1007/s40747-025-01974-x

§4.1 — Dataset is OpenNeuro ds004504 (Alzheimer's disease, Frontotemporal Dementia,
and Healthy Subjects EEG recordings). Only AD (subjects 1–36) and HC (subjects 37–65)
are used. FTD subjects are excluded.

Download instructions:
    https://openneuro.org/datasets/ds004504/versions/1.0.8
    Place the downloaded BIDS dataset at the path specified by raw_dataset_root in configs/base.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data

from src.utils import build_subject_list, load_config, pt_graph_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyG-compatible Dataset wrapper
# ---------------------------------------------------------------------------

class MFEFCGCNDataset(Dataset):
    """
    §3.3 / paper implementation — Dataset wrapping a list of 5-graph samples.

    Each item is a list of 5 PyG Data objects (one per frequency band),
    representing one windowed EEG sample from one subject.

    Attributes:
        data_list: List[List[Data]] — all samples (each = list of 5 Data objects)
        labels:    List[int]        — class labels extracted from data_list[i][0].y
    """

    def __init__(
        self,
        data_list: List[List[Data]],
        transform=None,
    ) -> None:
        super().__init__()
        self.data_list = data_list
        # Extract labels from the first band's Data object (all bands share label)
        self.labels: List[int] = [item[0].y.item() for item in data_list]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> List[Data]:
        sample = self.data_list[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# .pt file loading
# ---------------------------------------------------------------------------

def load_pt_file(pt_path: str) -> List[List[Data]]:
    """
    Load a serialised list of 5-graph samples from a .pt file.

    §3.3 / paper implementation — each .pt file stores all samples for one subject
    as a Python list: [sample_1, sample_2, ...] where each sample_i is [Data_Delta,
    Data_Theta, Data_Alpha, Data_Beta, Data_Gamma].

    Args:
        pt_path: Absolute path to the .pt file.

    Returns:
        List of samples, each a list of 5 PyG Data objects.
    """
    return torch.load(pt_path, weights_only=False)


def load_subjects_dataset(
    sub_ids: List[int],
    pt_root: str,
) -> MFEFCGCNDataset:
    """
    Load and concatenate .pt graph files for a list of subjects into one Dataset.

    §4.1 — subject-level splits are performed first; samples from all selected
    subjects are then pooled for training / validation / testing.

    Args:
        sub_ids: List of integer subject IDs to load.
        pt_root: Root directory containing per-subject .pt files.

    Returns:
        MFEFCGCNDataset containing all samples from the given subjects.
    """
    all_samples: List[List[Data]] = []
    for sid in sub_ids:
        pt_path = pt_graph_path(pt_root, sid)
        if not Path(str(pt_path)).exists():
            log.warning(f"[sub-{sid:03d}] .pt file not found at {pt_path}. Skipping.")
            continue
        subject_samples = load_pt_file(str(pt_path))
        all_samples.extend(subject_samples)
    return MFEFCGCNDataset(all_samples)


# ---------------------------------------------------------------------------
# Train/test split (outer fold)
# ---------------------------------------------------------------------------

def outer_train_test_split(
    subject_list: List[Tuple[int, int]],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    §4.1 — "80/20 stratified train/test split at the subject level."

    Stratification is on label to preserve AD/HC ratio in both splits.

    [UNSPECIFIED] Paper does not state a random seed for the outer split;
    leaving as caller-provided (None = non-deterministic).
    Alternatives: fix a seed (e.g. 42) for full reproducibility.

    Args:
        subject_list: List of (sub_id, label) tuples.
        test_size:    Fraction of subjects for the test set (default 0.2 — §4.1).
        random_state: Optional seed for sklearn train_test_split.

    Returns:
        (train_subjects, test_subjects) — each a list of (sub_id, label) tuples.
    """
    df = pd.DataFrame(subject_list, columns=["sub_id", "label"])
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    train_subjects = list(zip(train_df["sub_id"], train_df["label"]))
    test_subjects  = list(zip(test_df["sub_id"],  test_df["label"]))
    return train_subjects, test_subjects


# ---------------------------------------------------------------------------
# Inner k-fold cross-validation
# ---------------------------------------------------------------------------

def inner_kfold_splits(
    dataset: MFEFCGCNDataset,
    k: int = 15,
    random_state: int = 1,
) -> List[Tuple[Subset, Subset]]:
    """
    §4.1 — "inner 15-fold stratified cross-validation on the training set."

    Stratification is on sample-level labels.

    [UNSPECIFIED] Paper does not state random_state for inner folds;
    using 1 to match paper implementation.
    Alternatives: any integer seed.

    Args:
        dataset:      MFEFCGCNDataset of training samples.
        k:            Number of inner folds (default 15 — §4.1).
        random_state: Seed for StratifiedKFold shuffling.

    Returns:
        List of (train_subset, val_subset) pairs, one per fold.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    folds: List[Tuple[Subset, Subset]] = []
    for train_idx, val_idx in skf.split(dataset, dataset.labels):
        folds.append((Subset(dataset, train_idx), Subset(dataset, val_idx)))
    return folds
