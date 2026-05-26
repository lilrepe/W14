"""
MFE-FCGCN — Model Architecture
Paper: Multi-frequency EEG and multi-functional connectivity graph convolutional network
       based detection method of patients with Alzheimer's disease
DOI:   10.1007/s40747-025-01974-x

Implements §3.3 — the full MFE-FCGCN architecture, consisting of:
  1. SimpleCNN     — per-band 1-D convolutional feature extractor
  2. GCN           — two-layer GCNConv with dual edge sets (MI then PC)
  3. FiveGraphsModel — multi-band feature fusion + MLP classifier

Architecture overview (§3.3):
  For each of 5 frequency-band subgraphs:
    PSD node features (n_channels, n_psd_bins)
    → SimpleCNN (kernel squeezes bins → 1 scalar per channel)
    → GCNConv-1 (MI edges, nonlinear connectivity)
    → GCNConv-2 (PC edges, linear connectivity)
    → Flatten: (n_channels × out_channels,)
  Concatenate 5 bands: (5 × n_channels × out_channels,)
  → Linear → ReLU → Dropout
  → Linear → ReLU → Dropout
  → Linear → logits (2 classes: AD / HC)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    All model hyperparameters — see configs/base.yaml for sources.

    §3.3 / paper implementation values used as defaults.
    """
    # §3.3 / paper implementation — input feature size per node after CNN squeeze
    max_features: int = 16
    # [UNSPECIFIED] Paper does not explicitly state this number; value from paper implementation.
    # The CNN squeezes (n_channels, n_psd_bins) → (n_channels, 1); then max_features
    # controls the GCN input dimension. The paper implementation uses 16.

    # §3.3 / paper implementation — GCN hidden channels
    gcn1_channels: int = 16

    # §3.3 / paper implementation — GCN output channels per node
    gcn_out_channels: int = 5

    # §3.3 / paper implementation — MLP layer widths
    lin1_channels: int = 30
    lin2_channels: int = 14

    # §4.1 — binary classification: AD (1) vs HC (0)
    fc_out_channels: int = 2

    # §3.3 / paper implementation — dropout rates
    dropout_lin1_rate: float = 0.8
    dropout_lin2_rate: float = 0.5

    # §3.1 — number of EEG channels (nodes in graph)
    n_channels: int = 19

    # §3.1 — number of frequency bands
    n_bands: int = 5


# ---------------------------------------------------------------------------
# SimpleCNN — per-band PSD bin compressor
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    §3.3 — 1-D convolutional layer that collapses PSD frequency-bin dimension
    into a single scalar feature per EEG channel.

    Input to the model's forward: a batch of PyG Data objects.
    Each node's feature vector has dimension n_psd_bins (varies by band:
    Delta has fewer bins than Gamma because the frequency range is narrower).

    Design: Conv2d with kernel (1, n_psd_bins) — this sweeps across the bin
    dimension while treating channels independently.

    Because n_psd_bins differs across bands, the kernel size must match.
    The forward method dispatches on feature_dim.

    §3.3 / paper implementation — kernel sizes observed:
        feature_dim == 18 → kernel (1, 3)
        feature_dim == 21 → kernel (1, 6)
        feature_dim == 26 → kernel (1, 11)
        feature_dim == 76 → kernel (1, 61)
        feature_dim == 86 → kernel (1, 71)

    [UNSPECIFIED] Paper does not state these exact feature_dim values;
    they are derived from the Welch parameters (n_fft=2500, n_overlap=500,
    window_size=4) applied to each band's frequency range.
    Alternatives: confirm by running preprocessing on your data and checking
    the actual n_psd_bins per band.
    """

    def __init__(self) -> None:
        super().__init__()
        # §3.3 / paper implementation — one conv per possible bin count
        # in_channels=1, out_channels=1 (single feature map) — §3.3
        self.cnn_18 = nn.Conv2d(1, 1, kernel_size=(1, 3),  padding=(0, 0), stride=1)
        self.cnn_21 = nn.Conv2d(1, 1, kernel_size=(1, 6),  padding=(0, 0), stride=1)
        self.cnn_26 = nn.Conv2d(1, 1, kernel_size=(1, 11), padding=(0, 0), stride=1)
        self.cnn_76 = nn.Conv2d(1, 1, kernel_size=(1, 61), padding=(0, 0), stride=1)
        self.cnn_86 = nn.Conv2d(1, 1, kernel_size=(1, 71), padding=(0, 0), stride=1)

    def forward(self, data: Data) -> Data:
        """
        §3.3 — Extract single node feature from raw PSD bins via 1-D conv.

        Input tensor shape:  data.x — (batch_nodes, n_psd_bins)
            where batch_nodes = batch_size × n_channels
        Output: data.x updated to shape (batch_nodes, 1)

        Shape trace (assuming batch_size=B, n_channels=19, feature_dim=F):
            data.x: (B*19, F)
            → view:  (B, 1, 19, F)   — treat as single-channel 2-D image
            → conv:  (B, 1, 19, 1)   — kernel sweeps F dimension
            → view:  (B*19, 1)       — restore node-batch layout
        """
        feature_dim: int = data.x.size(1)      # n_psd_bins — varies by band
        x = data.x                              # (B*19, F)
        x = x.to(torch.float32)
        x = x.view(-1, 1, 19, feature_dim)     # (B, 1, 19, F)

        # Dispatch on feature_dim — §3.3 / paper implementation
        if feature_dim == 18:
            x = self.cnn_18(x)
        elif feature_dim == 21:
            x = self.cnn_21(x)
        elif feature_dim == 26:
            x = self.cnn_26(x)
        elif feature_dim == 76:
            x = self.cnn_76(x)
        elif feature_dim == 86:
            x = self.cnn_86(x)
        else:
            raise ValueError(
                f"Unexpected feature_dim={feature_dim}. "
                "Add a corresponding Conv2d layer or check preprocessing parameters. "
                "[UNSPECIFIED] Paper does not enumerate all valid feature_dim values."
            )
        # x: (B, 1, 19, 1)

        batch_size = len(set(data.batch.tolist()))
        x = x.view(batch_size * 19, -1).to(torch.float32)  # (B*19, 1)

        # Mutate data in-place (matches paper implementation pattern) — §3.3
        data.x = x
        return data


# ---------------------------------------------------------------------------
# GCN — two-layer graph convolutional network with dual edge sets
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    """
    §3.3 — Two-layer GCNConv sub-network applied to one frequency-band subgraph.

    Layer 1: GCNConv over MI (nonlinear) edges — captures nonlinear connectivity.
    Layer 2: GCNConv over PC (linear) edges   — captures linear connectivity.

    This alternating edge-set strategy is the paper's key multi-functional
    connectivity contribution (§3.2, contribution point 3).

    Output per graph: flattened node embeddings of shape (n_channels × out_channels,).
    """

    def __init__(
        self,
        max_features: int,
        gcn1_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        # §3.3 — GCNConv-1: MI edges (nonlinear functional connectivity)
        self.conv1 = GCNConv(max_features, gcn1_channels)
        # §3.3 — GCNConv-2: PC edges (linear functional connectivity)
        self.conv2 = GCNConv(gcn1_channels, out_channels)
        # §3.3 — per-band PSD feature extractor (shared across bands in FiveGraphsModel)
        self.cnn = SimpleCNN()

        self.max_features = max_features
        self.out_channels = out_channels

    def forward(self, data: Data) -> torch.Tensor:
        """
        §3.3 — Forward pass for a single frequency-band graph.

        Args:
            data: PyG Data object with fields:
                x              : (B*19, n_psd_bins) — raw PSD node features
                edge_index     : (2, n_edges_mi)    — MI edge COO indices
                edge_attr      : (n_edges_mi, 1)    — MI edge weights
                edge_index_two : (2, n_edges_pc)    — PC edge COO indices
                edge_weight_two: (n_edges_pc, 1)    — PC edge weights
                batch          : (B*19,)             — batch assignment
                y              : (B,)                — labels

        Returns:
            Tensor of shape (B, 19 × out_channels) — per-graph node embeddings.
        """
        # Step 1: compress PSD bins → 1 feature per node — §3.3
        data = self.cnn(data)
        # data.x: (B*19, 1)

        x: torch.Tensor = data.x.to(torch.float32)         # (B*19, 1)
        edge_index: torch.Tensor = data.edge_index.to(torch.long)
        edge_weight: torch.Tensor = data.edge_attr.to(torch.float32).squeeze(-1)  # (n_edges_mi,)
        edge_index_two: torch.Tensor = data.edge_index_two.to(torch.long)
        edge_weight_two: torch.Tensor = data.edge_weight_two.to(torch.float32).squeeze(-1)  # (n_edges_pc,)

        # Step 2: GCNConv-1 over MI (nonlinear) edge set — §3.3
        x = F.relu(self.conv1(x, edge_index, edge_weight))  # (B*19, gcn1_channels)

        # Step 3: GCNConv-2 over PC (linear) edge set — §3.3
        x = self.conv2(x, edge_index_two, edge_weight_two)  # (B*19, out_channels)

        # Step 4: flatten per-graph node embeddings — §3.3
        batch_size = x.size(0) // 19
        x = x.view(batch_size, 19 * self.out_channels)      # (B, 19 * out_channels)

        return x.to(torch.float32)


# ---------------------------------------------------------------------------
# FiveGraphsModel — multi-band fusion + MLP classifier
# ---------------------------------------------------------------------------

class FiveGraphsModel(nn.Module):
    """
    §3.3 — Full MFE-FCGCN model.

    Processes 5 frequency-band subgraphs through a shared GCN, concatenates
    their embeddings, and classifies via a 3-layer MLP.

    Input:  List of 5 PyG Data objects (one per band, ordered Delta→Gamma).
    Output: Logits of shape (B, 2) — softmax gives AD / HC probabilities.

    §3.3 — "multi-frequency feature extraction" refers to running the same
    GCN on each band's subgraph and then fusing via concatenation.
    This constitutes contribution points 2 and 4 from the paper.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # §3.3 — one GCN sub-network, applied to each of the 5 bands
        self.gcn = GCN(
            max_features=cfg.max_features,
            gcn1_channels=cfg.gcn1_channels,
            out_channels=cfg.gcn_out_channels,
        )

        # §3.3 — MLP fusion classifier
        # Input dim = 5 bands × 19 channels × out_channels
        fc_input_dim = cfg.n_bands * cfg.n_channels * cfg.gcn_out_channels  # 5 * 19 * 5 = 475

        # §3.3 / paper implementation — three linear layers
        self.fc  = nn.Linear(fc_input_dim,         cfg.lin1_channels)
        self.lin2 = nn.Linear(cfg.lin1_channels,   cfg.lin2_channels)
        self.lin3 = nn.Linear(cfg.lin2_channels,   cfg.fc_out_channels)

    def forward(self, data_list: List[Data]) -> torch.Tensor:
        """
        §3.3 — Multi-band forward pass.

        Args:
            data_list: List of 5 PyG Data objects (Delta, Theta, Alpha, Beta, Gamma).
                Each must be on the same device.

        Returns:
            logits: FloatTensor of shape (B, 2).
        """
        # Apply shared GCN to each band — §3.3 (contribution point 2)
        band_embeddings = [self.gcn(data) for data in data_list]
        # Each: (B, 19 × out_channels)

        # Concatenate across bands — §3.3 (contribution point 4)
        concat = torch.cat(band_embeddings, dim=1)  # (B, 5 × 19 × out_channels)
        concat = concat.to(torch.float32)

        # MLP classifier — §3.3 / paper implementation
        out = self.fc(concat)                                                # (B, lin1_channels)
        out = F.relu(out)
        out = F.dropout(out, p=self.cfg.dropout_lin1_rate, training=self.training)

        out = self.lin2(out)                                                 # (B, lin2_channels)
        out = F.relu(out)
        out = F.dropout(out, p=self.cfg.dropout_lin2_rate, training=self.training)

        out = self.lin3(out)                                                 # (B, 2)
        return out


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> FiveGraphsModel:
    """
    Instantiate FiveGraphsModel from the YAML config dict.

    All hyperparameters sourced from cfg['gcn'], cfg['classifier'], cfg['model'].
    """
    model_cfg = ModelConfig(
        max_features=cfg["gcn"]["input_features"],         # §3.3 / paper implementation
        gcn1_channels=cfg["gcn"]["hidden_channels"],       # §3.3 / paper implementation
        gcn_out_channels=cfg["gcn"]["out_channels"],       # §3.3 / paper implementation
        lin1_channels=cfg["classifier"]["lin1_channels"],  # §3.3 / paper implementation
        lin2_channels=cfg["classifier"]["lin2_channels"],  # §3.3 / paper implementation
        fc_out_channels=cfg["classifier"]["fc_out_channels"],  # §4.1
        dropout_lin1_rate=cfg["classifier"]["dropout_lin1_rate"],  # §3.3
        dropout_lin2_rate=cfg["classifier"]["dropout_lin2_rate"],  # §3.3
        n_channels=cfg["n_channels"],  # §3.1
        n_bands=cfg["n_bands"],        # §3.1
    )
    return FiveGraphsModel(model_cfg).to(device)
