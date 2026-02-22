# data_preprocessor.py
"""
DataPreprocessor
================
Sits between DataLoader.get_graph_data_list() and RelationalMotifCounter.

Responsibilities
----------------
1. GLOBAL PADDING
   Computes N_max over ALL graphs in the dataset, then pads every graph's
   features, adjacency matrices, and edge-feature tensors to (N_max, *).

2. ONE-HOT NODE FEATURES
   For every node-feature column (by index), collects all unique integer
   values seen across the entire dataset and assigns each (col, value) pair
   a column index in a flat one-hot matrix of shape (N_max, D).

   The mapping  feature_onehot_mapping  is:
       {col_idx: {value_int: onehot_col_idx}}

3. PRE-STACKED DATASET TENSORS
   After padding and one-hot encoding, all per-graph tensors are stacked
   into large dataset-level tensors stored on the target device:

       all_features   : (num_graphs, N_max, F)
       all_feat_onehot: (num_graphs, N_max, D)
       all_adj        : {rel_key: (num_graphs, N_max, N_max)}
       all_edge       : list of (num_graphs, C, N_max, N_max) or None

   During counting, _build_batch_tensors() retrieves a batch with a pure
   O(1) tensor slice — no Python loops, no .to(device) calls, no padding.

Usage
-----
    preprocessor = DataPreprocessor(device=args.device)
    preprocessor.preprocess(graph_data_list)          # builds dataset tensors

    # Pass preprocessor to motif_counter — it reads slices directly.
    motif_counter.count_batch(preprocessor, batch_size=batch_size)
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple


class DataPreprocessor:
    """
    Pre-pads all graphs to global N_max, builds one-hot feature matrices,
    and pre-stacks everything into dataset-level GPU tensors for O(1) batching.

    Parameters
    ----------
    device : str
        'cuda' or 'cpu'.  All output tensors are moved to this device.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

        # Set by preprocess()
        self.N_max:                  int = 0
        self.num_graphs:             int = 0
        self.feature_onehot_mapping: Dict[int, Dict[int, int]] = {}
        self.total_onehot_dim:       int = 0
        self.relation_keys:          List[str] = []
        self.has_edge_features:      bool = False

        # Pre-stacked dataset tensors (set by preprocess)
        self.all_features:    Optional[torch.Tensor] = None   # (G, N_max, F)
        self.all_feat_onehot: Optional[torch.Tensor] = None   # (G, N_max, D)
        self.all_adj:         Dict[str, torch.Tensor] = {}    # {rel: (G, N_max, N_max)}
        self.all_edge:        Optional[List[torch.Tensor]] = None  # list[(G,C,N_max,N_max)]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def preprocess(self, graph_data_list: List[Dict]) -> 'DataPreprocessor':
        """
        One-pass preprocessing over all graphs.

        Builds:
          self.all_features    (num_graphs, N_max, F)
          self.all_feat_onehot (num_graphs, N_max, D)
          self.all_adj         {rel: (num_graphs, N_max, N_max)}
          self.all_edge        list of (num_graphs, C, N_max, N_max) or None

        Also stores feature_onehot_mapping and N_max metadata into each
        graph_data dict for compatibility with the single-graph count() path.

        Returns self so callers can chain:
            preprocessor = DataPreprocessor(device).preprocess(graph_data_list)
        """
        if not graph_data_list:
            return self

        G = len(graph_data_list)
        self.num_graphs = G

        # ── Step 1: global N_max ──────────────────────────────────────
        self.N_max = max(
            gd['features'].shape[0] if torch.is_tensor(gd['features'])
            else len(gd['features'])
            for gd in graph_data_list
        )
        N_max = self.N_max
        print(f"  [DataPreprocessor] Global N_max = {N_max}  ({G} graphs)")

        # ── Step 2: one-hot mapping ───────────────────────────────────
        self._build_onehot_mapping(graph_data_list)
        D = self.total_onehot_dim
        print(f"  [DataPreprocessor] One-hot feature dim = {D}")

        # Determine relation keys and edge feature count
        self.relation_keys     = list(graph_data_list[0]['matrices'].keys())
        sample_labels          = graph_data_list[0].get('labels')
        self.has_edge_features = sample_labels is not None
        num_edge_feat_types    = len(sample_labels) if sample_labels is not None else 0

        # ── Step 3: infer feature dim F ───────────────────────────────
        f0 = graph_data_list[0]['features']
        if not torch.is_tensor(f0):
            f0 = torch.tensor(f0, dtype=torch.float32)
        F = f0.shape[1]

        # ── Step 4: allocate CPU accumulators ─────────────────────────
        feat_acc   = torch.zeros(G, N_max, F,  dtype=torch.float32)
        onehot_acc = torch.zeros(G, N_max, D,  dtype=torch.float32)
        adj_acc    = {rel: torch.zeros(G, N_max, N_max, dtype=torch.float32)
                      for rel in self.relation_keys}
        edge_acc: Optional[List[torch.Tensor]] = None
        if self.has_edge_features:
            edge_acc = []
            for feat_idx in range(num_edge_feat_types):
                ef = graph_data_list[0]['labels'][feat_idx]
                if not torch.is_tensor(ef):
                    ef = torch.tensor(ef, dtype=torch.float32)
                C = ef.shape[0]
                edge_acc.append(torch.zeros(G, C, N_max, N_max, dtype=torch.float32))

        # ── Step 5: fill accumulators in one Python loop ──────────────
        for g, gd in enumerate(graph_data_list):
            feats = gd['features']
            if not torch.is_tensor(feats):
                feats = torch.tensor(feats, dtype=torch.float32)
            else:
                feats = feats.float().cpu()

            N = feats.shape[0]
            feat_acc[g, :N]   = feats
            onehot_acc[g, :N] = self._build_onehot(feats, N)

            for rel in self.relation_keys:
                a = gd['matrices'][rel]
                if not torch.is_tensor(a):
                    a = torch.tensor(a, dtype=torch.float32)
                else:
                    a = a.float().cpu()
                Na = a.shape[0]
                adj_acc[rel][g, :Na, :Na] = a

            if self.has_edge_features and edge_acc is not None:
                for feat_idx in range(num_edge_feat_types):
                    ef = gd['labels'][feat_idx]
                    if not torch.is_tensor(ef):
                        ef = torch.tensor(ef, dtype=torch.float32)
                    else:
                        ef = ef.float().cpu()
                    C, Ne, _ = ef.shape
                    edge_acc[feat_idx][g, :C, :Ne, :Ne] = ef

            # Write padded slices + metadata back into each graph_data dict
            # so the single-graph count() path still works without changes.
            gd['features']               = feat_acc[g]
            gd['feat_onehot']            = onehot_acc[g]
            gd['feature_onehot_mapping'] = self.feature_onehot_mapping
            gd['N_max']                  = N_max
            for rel in self.relation_keys:
                gd['matrices'][rel]      = adj_acc[rel][g]
            if self.has_edge_features and edge_acc is not None:
                gd['labels'] = [edge_acc[fi][g] for fi in range(num_edge_feat_types)]

        # ── Step 6: pin CPU tensors for fast async host→device transfer ─
        # Tensors stay on CPU. get_batch() moves one contiguous slice to GPU
        # per batch — one transfer instead of 50k individual .to() calls,
        # and VRAM is only occupied by one batch at a time.
        print(f"  [DataPreprocessor] Pinning stacked tensors in CPU memory...")
        self.all_features    = feat_acc.pin_memory()
        self.all_feat_onehot = onehot_acc.pin_memory()
        self.all_adj         = {rel: adj_acc[rel].pin_memory() for rel in self.relation_keys}
        if self.has_edge_features and edge_acc is not None:
            self.all_edge = [e.pin_memory() for e in edge_acc]

        print(f"  [DataPreprocessor] Done. "
              f"features={tuple(self.all_features.shape)}, "
              f"onehot={tuple(self.all_feat_onehot.shape)}"
              f"  [pinned CPU → GPU per batch]")

        return self

    # ------------------------------------------------------------------
    # Batch slice — O(1), called by motif_counter._build_batch_tensors
    # ------------------------------------------------------------------

    def get_batch(
        self,
        start: int,
        end:   int,   # exclusive
    ) -> Tuple[
        torch.Tensor,                    # feat_b        (B, N_max, F)
        torch.Tensor,                    # feat_onehot_b (B, N_max, D)
        Dict[str, torch.Tensor],         # adj_b         {rel: (B, N_max, N_max)}
        Optional[List[torch.Tensor]],    # edge_b        list[(B,C,N_max,N_max)] or None
    ]:
        """
        Slice the pre-stacked CPU tensors and move to GPU in one transfer.

        Why not pre-move to GPU?
        -----------------------
        Keeping all 130k graphs on GPU simultaneously saturates VRAM, which
        causes memory pressure when _iteration_function_batched allocates its
        intermediate tensors (unmasked, masked, sorted, stacked).  Moving one
        batch slice at a time keeps VRAM usage bounded to one batch.

        Why faster than the old per-graph .to() approach?
        --------------------------------------------------
        One contiguous slice→device copy (e.g. 50000×29×29 floats in one DMA
        transfer) is orders of magnitude faster than 50000 separate .to()
        calls that each incur Python overhead + a tiny DMA descriptor.
        Pin-memory ensures the transfer goes directly from RAM to VRAM without
        an intermediate OS copy.
        """
        dev = self.device
        feat_b        = self.all_features[start:end].to(dev, non_blocking=True)
        feat_onehot_b = self.all_feat_onehot[start:end].to(dev, non_blocking=True)
        adj_b         = {rel: self.all_adj[rel][start:end].to(dev, non_blocking=True)
                         for rel in self.relation_keys}
        edge_b        = ([e[start:end].to(dev, non_blocking=True) for e in self.all_edge]
                         if self.all_edge is not None else None)
        return feat_b, feat_onehot_b, adj_b, edge_b

    # ------------------------------------------------------------------
    # Private: one-hot mapping construction
    # ------------------------------------------------------------------

    def _build_onehot_mapping(self, graph_data_list: List[Dict]):
        f0 = graph_data_list[0]['features']
        if not torch.is_tensor(f0):
            f0 = torch.tensor(f0, dtype=torch.float32)
        num_feats = f0.shape[1]

        col_unique: Dict[int, set] = {col: set() for col in range(num_feats)}
        for gd in graph_data_list:
            feats = gd['features']
            if not torch.is_tensor(feats):
                feats = torch.tensor(feats, dtype=torch.float32)
            for col in range(num_feats):
                for v in feats[:, col].unique().tolist():
                    col_unique[col].add(int(round(v)))

        mapping: Dict[int, Dict[int, int]] = {}
        onehot_col = 0
        for col in range(num_feats):
            mapping[col] = {}
            for val in sorted(col_unique[col]):
                mapping[col][val] = onehot_col
                onehot_col += 1

        self.feature_onehot_mapping = mapping
        self.total_onehot_dim = onehot_col

    # ------------------------------------------------------------------
    # Private: one-hot construction for a single feature matrix
    # ------------------------------------------------------------------

    def _build_onehot(self, feats: torch.Tensor, N: int) -> torch.Tensor:
        """Build (N, D) one-hot tensor from (N, F) feature matrix (CPU)."""
        D      = self.total_onehot_dim
        result = torch.zeros(N, D, dtype=torch.float32)
        for col, col_map in self.feature_onehot_mapping.items():
            for val, oh_col in col_map.items():
                mask = (feats[:, col] == val)
                result[mask, oh_col] = 1.0
        return result
