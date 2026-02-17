# data.py

import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from dgl.data import CoraGraphDataset
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import scipy.sparse as sparse
import copy
from torch_geometric.datasets import QM9
import pickle
from pathlib import Path


def reduce_node_features(x, y, random_seed, n_components=5) -> Tuple[np.ndarray, np.ndarray]:
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    np.random.seed(random_seed)
    model = ExtraTreesClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_)
    important_feats = np.array(feat_importances.nlargest(n_components).index)
    return x[:, important_feats], important_feats


# ============================================================================
# QM9 Helper Functions
# ============================================================================

def edge_feature_info(dataset, relation_name="edges"):
    total_num_nodes = 0
    all_vals = set()
    for g in dataset:
        total_num_nodes += int(g.num_nodes)
        if getattr(g, "edge_attr", None) is None or g.edge_attr.numel() == 0:
            continue
        all_vals.update(int(v) for v in torch.argmax(g.edge_attr, dim=1).tolist())

    unique_values = sorted(all_vals)
    return {
        0: {
            "relation_name":    relation_name,
            "feature_name":     "bond_type",
            "num_nodes":        int(total_num_nodes),
            "num_unique_values": len(unique_values),
            "unique_values":    unique_values,
            "value_to_index":   {int(v): i for i, v in enumerate(unique_values)},
            "index_to_value":   {i: int(v) for i, v in enumerate(unique_values)},
        }
    }


def scipy_adj_to_torch(adj_csr, device=None, dtype=torch.float32):
    A = adj_csr.toarray() if sparse.issparse(adj_csr) else np.asarray(adj_csr)
    return torch.tensor(A, dtype=dtype, device=device)


def nodefeat_to_torch(x_np, device=None, dtype=torch.long):
    return torch.tensor(np.asarray(x_np), dtype=dtype, device=device)


def build_edge_feature_tensors_for_graph(edge_array, n_nodes, feature_info_mapping,
                                          device=None, dtype=torch.float32):
    per_graph_fim = copy.deepcopy(feature_info_mapping)
    for f in per_graph_fim:
        per_graph_fim[f]["num_nodes"] = int(n_nodes)

    edge_array = np.zeros((0, 3), dtype=int) if edge_array is None else np.asarray(edge_array)
    edge_tensors = []

    for f_idx in sorted(per_graph_fim.keys()):
        info = per_graph_fim[f_idx]
        C, v2i = int(info["num_unique_values"]), info["value_to_index"]
        T = torch.zeros((C, n_nodes, n_nodes), dtype=dtype, device=device)
        if edge_array.shape[0] > 0:
            for s, d, v in zip(edge_array[:, 0].astype(int),
                               edge_array[:, 1].astype(int),
                               edge_array[:, 2]):
                if 0 <= s < n_nodes and 0 <= d < n_nodes and v in v2i:
                    T[int(v2i[v]), s, d] = 1.0
        edge_tensors.append(T)

    return edge_tensors, per_graph_fim


# ============================================================================
# DataLoader
# ============================================================================

class DataLoader:
    """
    Loads Cora or QM9 and builds graph_data dicts for motif counting.

    graph_data structure  (from get_graph_data_list)
    -------------------------------------------------
    {
        'matrices': {<relation_name>: (N, N) tensor},   <- keys match motif_counter.relation_keys
        'features': (N, F) tensor / numpy array,
        'labels':   edge-feature tensors | None,
        'feature_info_mapping': dict | None,             <- QM9 only
    }

    Usage
    -----
    # Step 1: create the counter so its relation keys are known
    motif_counter = RelationalMotifCounter(database_name, args)

    # Step 2: pass those keys to get_graph_data_list so 'matrices' dict is built correctly
    graph_data_list = data_loader.get_graph_data_list(
        relation_keys=motif_counter.relation_keys   # e.g. ['citations'] or ['edges']
    )

    # Step 3: count — counter reads graph_data['matrices'] directly, no internal copy
    motif_counts = motif_counter.count(graph_data_list[0])
    """

    def __init__(self, dataset_type: str = 'cora', n_components: int = 5,
                 random_seed: int = 0, max_graphs: int = None, device: str = 'cpu'):
        self.dataset_type = dataset_type.lower()
        self.n_components = n_components
        self.random_seed  = random_seed
        self.max_graphs   = max_graphs
        self.device       = device

        self.cache_dir = Path('./preprocessed_data')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._graph_data_list_cache = None

        # Cora
        self.graph                     = None
        self.features                  = None
        self.features_raw              = None
        self.labels                    = None
        self.adjacency_matrix          = None
        self.num_classes               = None
        self.edges                     = None
        self.num_nodes                 = None
        self.num_features              = None
        self.train_mask                = None
        self.val_mask                  = None
        self.test_mask                 = None
        self.important_feature_indices = None

        # QM9
        self.list_adj             = []
        self.list_node_feature    = []
        self.list_edge_feature    = []
        self.feature_info_mapping = None

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #

    def _get_cache_filepath(self) -> Path:
        if self.dataset_type == 'cora':
            return self.cache_dir / "cora_graph_data.pkl"
        elif self.dataset_type == 'qm9':
            suffix = "all" if self.max_graphs is None else str(self.max_graphs)
            return self.cache_dir / f"qm9_{suffix}_graph_data.pkl"
        return self.cache_dir / f"{self.dataset_type}_graph_data.pkl"

    def _load_from_cache(self) -> bool:
        cache_file = self._get_cache_filepath()
        if not cache_file.exists():
            return False
        try:
            print(f"  Loading preprocessed data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            self._graph_data_list_cache = cached_data['graph_data_list']
            if self.dataset_type == 'cora':
                self.num_nodes    = cached_data.get('num_nodes')
                self.num_features = cached_data.get('num_features')
            elif self.dataset_type == 'qm9':
                self.feature_info_mapping = cached_data.get('feature_info_mapping')
            print(f"  Successfully loaded from cache")
            return True
        except Exception as e:
            print(f"  Failed to load from cache: {e}")
            return False

    def _save_to_cache(self, graph_data_list: List[Dict]):
        cache_file = self._get_cache_filepath()
        try:
            print(f"  Saving preprocessed data to cache: {cache_file}")
            cache_data = {
                'graph_data_list': graph_data_list,
                'dataset_type':    self.dataset_type,
                'device':          self.device,
            }
            if self.dataset_type == 'cora':
                cache_data['num_nodes']    = self.num_nodes
                cache_data['num_features'] = self.num_features
            elif self.dataset_type == 'qm9':
                cache_data['feature_info_mapping'] = self.feature_info_mapping
                cache_data['max_graphs']           = self.max_graphs
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"  Saved to cache")
        except Exception as e:
            print(f"  Failed to save to cache: {e}")

    # ------------------------------------------------------------------ #
    # Load raw dataset
    # ------------------------------------------------------------------ #

    def load_data(self) -> None:
        if self._load_from_cache():
            return
        if self.dataset_type == 'cora':
            self._load_cora()
        elif self.dataset_type == 'qm9':
            self._load_qm9()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}. Use 'cora' or 'qm9'.")

    def _load_cora(self) -> None:
        dataset = CoraGraphDataset()
        self.graph = dataset[0]

        features = self.graph.ndata['feat']
        self.features_raw = torch.where(
            features > 0, torch.ones_like(features), torch.zeros_like(features)
        ).float()
        self.labels      = self.graph.ndata['label']
        self.num_classes = dataset.num_classes

        self._reduce_features()

        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask   = self.graph.ndata['val_mask']
        self.test_mask  = self.graph.ndata['test_mask']

        src, dst    = self.graph.edges()
        self.edges  = torch.stack([src, dst], dim=1)
        self.num_nodes = self.graph.num_nodes()
        self.adjacency_matrix = self.graph.adjacency_matrix().to_dense().float()

    def _load_qm9(self) -> None:
        print(f"  Loading QM9 dataset from ./data/QM9...")
        data = QM9(root="./data/QM9")

        print(f"  Building edge feature mapping...")
        self.feature_info_mapping = edge_feature_info(data)

        num_graphs = len(data) if self.max_graphs is None else min(self.max_graphs, len(data))
        print(f"  Processing {num_graphs} molecules...")

        for i in range(num_graphs):
            mol = data[i]
            N   = mol.num_nodes
            edge_index = mol.edge_index

            self.list_adj.append(sparse.csr_matrix(
                (np.ones(edge_index.size(1)),
                 (edge_index[0].numpy(), edge_index[1].numpy())),
                shape=(N, N)
            ))

            X = mol.x
            atom_type = torch.argmax(X[:, 0:5], dim=1)
            num_h     = torch.clamp(X[:, 10].long(), max=3)
            self.list_node_feature.append(torch.stack([atom_type, num_h], dim=1).numpy())

            if mol.edge_attr is not None and mol.edge_attr.size(0) > 0:
                bond_type = torch.argmax(mol.edge_attr, dim=1)
                self.list_edge_feature.append(
                    torch.stack([edge_index[0], edge_index[1], bond_type], dim=1).numpy()
                )
            else:
                self.list_edge_feature.append(None)

            if (i + 1) % 1000 == 0:
                print(f"    Processed {i + 1}/{num_graphs} molecules...")

        print(f"  Loaded {len(self.list_adj)} molecules")

    def _reduce_features(self) -> None:
        x_reduced, important_feats = reduce_node_features(
            self.features_raw, self.labels, self.random_seed, self.n_components
        )
        self.important_feature_indices = important_feats
        self.features     = np.concatenate(
            [x_reduced, self.labels.numpy().reshape(-1, 1)], axis=1
        )
        self.num_features = self.features.shape[1]

    # ------------------------------------------------------------------ #
    # get_data  (unchanged public API)
    # ------------------------------------------------------------------ #

    def get_data(self) -> Dict:
        if self.dataset_type == 'cora':
            return {
                'graph':                    self.graph,
                'features':                 self.features,
                'features_raw':             self.features_raw,
                'labels':                   self.labels,
                'num_classes':              self.num_classes,
                'adjacency_matrix':         self.adjacency_matrix,
                'edges':                    self.edges,
                'num_nodes':                self.num_nodes,
                'num_features':             self.num_features,
                'train_mask':               self.train_mask,
                'val_mask':                 self.val_mask,
                'test_mask':                self.test_mask,
                'important_feature_indices': self.important_feature_indices,
            }
        elif self.dataset_type == 'qm9':
            return {
                'list_adj':             self.list_adj,
                'list_node_feature':    self.list_node_feature,
                'list_edge_feature':    self.list_edge_feature,
                'feature_info_mapping': self.feature_info_mapping,
                'num_graphs': (len(self.list_adj) if self.list_adj
                               else len(self._graph_data_list_cache or [])),
            }

    # ------------------------------------------------------------------ #
    # get_graph_data_list  ← THE KEY METHOD
    # ------------------------------------------------------------------ #

    def get_graph_data_list(
        self, relation_keys: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Build and return a list of graph_data dicts for motif counting.

        Each dict has the structure:
            {
                'matrices': {<relation_name>: (N, N) tensor},
                'features': (N, F) tensor / numpy array,
                'labels':   edge-feature tensors | None,
                'feature_info_mapping': dict | None,
            }

        Parameters
        ----------
        relation_keys : list[str]
            Relation names to use as keys inside 'matrices'.
            Get from motif_counter.relation_keys BEFORE calling this method.

            Example:
                motif_counter = RelationalMotifCounter(database_name, args)
                graph_data_list = data_loader.get_graph_data_list(
                    relation_keys=motif_counter.relation_keys
                )

            - Cora: ['citations']
            - QM9:  ['edges']   (or whatever the DB relation is named)
        """
        if self._graph_data_list_cache is not None:
            print(f"  Using cached graph_data_list ({len(self._graph_data_list_cache)} graphs)")
            return self._graph_data_list_cache

        if relation_keys is None:
            relation_keys = ['adjacency']   # generic fallback
            print(f"  Warning: relation_keys not provided, using fallback: {relation_keys}")
            print(f"  Pass motif_counter.relation_keys to get_graph_data_list() for correct keys.")

        print(f"  Building graph_data_list (matrices keys: {relation_keys})...")

        if self.dataset_type == 'cora':
            graph_data_list = self._build_cora_list(relation_keys)
        elif self.dataset_type == 'qm9':
            graph_data_list = self._build_qm9_list(relation_keys)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        self._graph_data_list_cache = graph_data_list
        self._save_to_cache(graph_data_list)
        return graph_data_list

    # ------------------------------------------------------------------ #
    # Per-dataset builders
    # ------------------------------------------------------------------ #

    def _build_cora_list(self, relation_keys: List[str]) -> List[Dict]:
        """
        Cora is one graph.
        'matrices' = {key: adjacency_tensor} for every key in relation_keys.
        """
        adj = self.adjacency_matrix.to(self.device)  # (N, N) float tensor

        graph_data = {
            'matrices': {key: adj for key in relation_keys},
            'features': self.features,    # numpy (N, F) with labels column
            'labels':   None,             # no edge features in Cora
            'feature_info_mapping': None,
        }
        return [graph_data]

    def _build_qm9_list(self, relation_keys: List[str]) -> List[Dict]:
        """
        QM9 has one graph_data per molecule.
        'matrices' = {key: adj_tensor} for every key in relation_keys.
        """
        graph_data_list = []
        num_graphs = len(self.list_adj)
        print(f"  Converting {num_graphs} molecules to graph_data format...")

        for g in range(num_graphs):
            adj     = scipy_adj_to_torch(self.list_adj[g], device=self.device)
            n_nodes = adj.shape[0]

            node_features = nodefeat_to_torch(self.list_node_feature[g], device=self.device)

            edge_tensors, per_graph_fim = build_edge_feature_tensors_for_graph(
                self.list_edge_feature[g], n_nodes,
                self.feature_info_mapping, device=self.device,
            )

            graph_data = {
                'matrices': {key: adj for key in relation_keys},
                'features': node_features,
                'labels':   edge_tensors,        # list of (C, N, N) tensors
                'feature_info_mapping': per_graph_fim,
            }
            graph_data_list.append(graph_data)

            if (g + 1) % 1000 == 0:
                print(f"    Converted {g + 1}/{num_graphs} molecules...")

        print(f"  Converted all {num_graphs} molecules")
        return graph_data_list
