"""
MFE-FCGCN — Training Loop
Paper: Multi-frequency EEG and multi-functional connectivity graph convolutional network
       based detection method of patients with Alzheimer's disease
DOI:   10.1007/s40747-025-01974-x

§4.1 — Training procedure:
  - Outer 80/20 stratified split at subject level
  - Inner 15-fold stratified cross-validation on training subjects
  - Adam optimiser, lr=0.00012, weight_decay=1e-4
  - ReduceLROnPlateau scheduler (mode=min, factor=0.1, patience=5)
  - 200 epochs per inner fold
  - Best model selected by validation AUC
  - Binary cross-entropy loss (CrossEntropyLoss over 2 classes)

Usage:
    python -m src.train --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data import (
    MFEFCGCNDataset,
    inner_kfold_splits,
    load_subjects_dataset,
    outer_train_test_split,
)
from src.evaluate import EvalMetrics, compute_metrics
from src.model import FiveGraphsModel, build_model
from src.utils import build_subject_list, load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_five_graphs(batch: List[List]) -> List:
    """
    §3.3 — Custom collate function for batches of 5-graph samples.

    Each item in `batch` is a list of 5 PyG Data objects.
    Returns a list of 5 mini-batched Data objects (one per band).

    PyG's DataLoader expects a single Data object per item, but our samples
    are lists of 5. This collator transposes [sample × band] → [band × batch].
    """
    from torch_geometric.data import Batch

    band_batches = []
    for band_idx in range(5):  # §3.1 — 5 bands
        band_data = [sample[band_idx] for sample in batch]
        band_batches.append(Batch.from_data_list(band_data))
    return band_batches


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: FiveGraphsModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    §4.1 — Single training epoch.

    Returns:
        (total_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_total = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        # Move all 5 band-batches to device
        band_batches = [b.to(device) for b in batch]

        optimizer.zero_grad()
        logits = model(band_batches)                        # (B, 2)
        labels = band_batches[0].y                          # (B,) — same label across bands

        loss = criterion(logits, labels)                    # §4.1 — CrossEntropyLoss
        loss.backward()
        optimizer.step()

        preds = logits.softmax(dim=1).argmax(dim=1)         # hard predictions
        n_correct += (preds == labels).sum().item()
        n_total += labels.size(0)
        total_loss += loss.item()

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return total_loss, accuracy


# ---------------------------------------------------------------------------
# Evaluation (val / test)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: FiveGraphsModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    n_samples: int,
) -> EvalMetrics:
    """
    §4.2 — Evaluate model on a DataLoader, computing all reported metrics.

    Args:
        n_samples: Total sample count in the dataset (for accuracy denominator).
    """
    model.eval()

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_scores_all: List[float] = []
    total_loss = 0.0
    n_correct = 0

    for batch in tqdm(loader, desc="  eval ", leave=False):
        band_batches = [b.to(device) for b in batch]
        logits = model(band_batches)                         # (B, 2)
        labels = band_batches[0].y                           # (B,)

        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)

        # §4.2 — AUC uses the score for the positive class (AD=1)
        y_scores_all.extend(probs[:, 1].cpu().numpy().tolist())
        y_pred_all.extend(preds.cpu().numpy().tolist())
        y_true_all.extend(labels.cpu().numpy().tolist())

        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_correct += (preds == labels).sum().item()

    return compute_metrics(
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_scores=y_scores_all,
        n_correct=n_correct,
        n_total=n_samples,
        total_loss=total_loss,
    )


# ---------------------------------------------------------------------------
# Save / load checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model: FiveGraphsModel, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    log.info(f"Checkpoint saved → {path}")


def load_checkpoint(model: FiveGraphsModel, path: str, device: torch.device) -> FiveGraphsModel:
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    log.info(f"Checkpoint loaded ← {path}")
    return model


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def run_training(cfg: dict) -> None:
    """
    §4.1 — Full nested cross-validation training procedure.

    Outer loop: one 80/20 subject-level split.
    Inner loop: 15-fold stratified CV on training subjects.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    subject_list = build_subject_list(cfg)  # §4.1 — 65 subjects

    # §4.1 — outer 80/20 stratified split at subject level
    train_subjects, test_subjects = outer_train_test_split(
        subject_list,
        test_size=cfg["cv"]["test_size"],   # 0.2 — §4.1
        # [UNSPECIFIED] random_state not stated in paper; passing None
        random_state=None,
    )

    pt_root = cfg["pt_root"]

    # Log split CSVs — §4.1 / paper implementation
    kfold_raw_root = Path(cfg["kfold_raw_root"]) / "fold_1"
    kfold_raw_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_subjects, columns=["sub_id", "label"]).to_csv(
        kfold_raw_root / "train_fold_1.csv"
    )
    pd.DataFrame(test_subjects, columns=["sub_id", "label"]).to_csv(
        kfold_raw_root / "test_fold_1.csv"
    )

    # Load datasets
    train_ids = [s[0] for s in train_subjects]
    test_ids  = [s[0] for s in test_subjects]

    train_dataset = load_subjects_dataset(train_ids, pt_root)
    test_dataset  = load_subjects_dataset(test_ids,  pt_root)

    # §4.1 — test DataLoader (shared across all inner folds)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size_test"],  # paper implementation
        shuffle=True,
        collate_fn=collate_five_graphs,
    )

    best_model_path = cfg["best_model_path"]
    evaluate_root = Path(cfg["evaluate_root"]) / "fold_1"
    evaluate_root.mkdir(parents=True, exist_ok=True)

    # §4.1 — inner 15-fold stratified cross-validation
    inner_folds = inner_kfold_splits(
        train_dataset,
        k=cfg["cv"]["inner_k"],                  # 15 — §4.1
        random_state=cfg["cv"]["inner_random_state"],
    )

    val_best_auc_global = 0.0

    for fold_idx, (inner_train, val) in enumerate(inner_folds):
        log.info(
            f"\n=== Inner fold {fold_idx + 1}/{cfg['cv']['inner_k']} "
            f"| train={len(inner_train)} val={len(val)} ==="
        )

        train_loader = DataLoader(
            inner_train,
            batch_size=cfg["training"]["batch_size_train"],  # 64 — paper implementation
            shuffle=True,
            collate_fn=collate_five_graphs,
        )
        val_loader = DataLoader(
            val,
            batch_size=cfg["training"]["batch_size_val"],    # 128 — paper implementation
            shuffle=False,
            collate_fn=collate_five_graphs,
        )

        # §4.1 — model and optimiser
        model = build_model(cfg, device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["lr"],                  # 0.00012 — §4.1
            weight_decay=cfg["training"]["weight_decay"],  # 1e-4 — §4.1
        )
        # [PARTIALLY_SPECIFIED] ReduceLROnPlateau used; factor/patience from paper implementation
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=cfg["training"]["scheduler"]["mode"],       # "min"
            factor=cfg["training"]["scheduler"]["factor"],   # 0.1
            patience=cfg["training"]["scheduler"]["patience"],  # 5
        )
        criterion = torch.nn.CrossEntropyLoss()  # §4.1

        # Per-fold metric history
        train_history = {"train_loss": [], "train_accuracy": []}
        val_history   = {"val_loss": [], "val_accuracy": [], "val_sensitivity": [],
                         "val_specificity": [], "val_f1": [], "val_auc": []}
        test_history  = {"test_loss": [], "test_accuracy": [], "test_sensitivity": [],
                         "test_specificity": [], "test_f1": [], "test_auc": []}

        # §4.1 — 200 epochs per inner fold
        for epoch in range(cfg["training"]["epochs"]):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_metrics = evaluate(model, val_loader, criterion, device, len(val))
            test_metrics = evaluate(model, test_loader, criterion, device, len(test_dataset))

            # Scheduler steps on test loss — §4.1 / paper implementation
            scheduler.step(test_metrics.loss)

            # Best model by validation AUC — §4.1 / paper implementation
            if val_metrics.auc > val_best_auc_global:
                val_best_auc_global = val_metrics.auc
                save_checkpoint(model, best_model_path)

            log.info(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_acc={val_metrics.accuracy:.4f} val_auc={val_metrics.auc:.4f} | "
                f"test_acc={test_metrics.accuracy:.4f} test_auc={test_metrics.auc:.4f}"
            )

            # Accumulate history
            train_history["train_loss"].append(train_loss)
            train_history["train_accuracy"].append(train_acc)
            for k, v in val_metrics.as_dict().items():
                if f"val_{k}" in val_history:
                    val_history[f"val_{k}"].append(v)
            for k, v in test_metrics.as_dict().items():
                if f"test_{k}" in test_history:
                    test_history[f"test_{k}"].append(v)

        # Save per-fold CSVs — §4.1 / paper implementation
        pd.DataFrame(train_history).to_csv(
            evaluate_root / f"train_evaluate_{fold_idx + 1}.csv"
        )
        pd.DataFrame(val_history).to_csv(
            evaluate_root / f"val_evaluate_{fold_idx + 1}.csv"
        )
        pd.DataFrame(test_history).to_csv(
            evaluate_root / f"test_evaluate_{fold_idx + 1}.csv"
        )

    # Final test evaluation on best checkpoint
    log.info("\n=== Final evaluation with best-AUC checkpoint ===")
    final_model = build_model(cfg, device)
    final_model = load_checkpoint(final_model, best_model_path, device)
    final_metrics = evaluate(
        final_model, test_loader, torch.nn.CrossEntropyLoss(), device, len(test_dataset)
    )
    log.info(
        f"Best model | "
        f"acc={final_metrics.accuracy:.4f} "
        f"sensitivity={final_metrics.sensitivity:.4f} "
        f"specificity={final_metrics.specificity:.4f} "
        f"f1={final_metrics.f1:.4f} "
        f"auc={final_metrics.auc:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MFE-FCGCN training")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_training(cfg)
