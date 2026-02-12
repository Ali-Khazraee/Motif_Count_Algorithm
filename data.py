# data.py

import numpy as np
import torch
from typing import Dict, Tuple, List
import dgl
from dgl.data import CoraGraphDataset
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import scipy.sparse as sparse
import copy
from torch_geometric.datasets import QM9
import pickle
from pathlib import Path
import os


def reduce_node_features(x, y, random_seed, n_components=5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce node features using ExtraTreesClassifier feature importance.
    
    Args:
        x: Node features (numpy array or torch tensor)
        y: Node labels (numpy array or torch tensor)
        random_seed: Random seed for reproducibility
        n_components: Number of top features to keep
        
    Returns:
        Tuple of (reduced features, indices of important features)
    """
    # Convert to numpy if needed
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    
    np.random.seed(random_seed)
    model = ExtraTreesClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_)
    important_feats = np.array(feat_importances.nlargest(n_components).index)
    x_reduced = x[:, important_feats]
    return x_reduced, important_feats


# ============================================================================
# QM9 Helper Functions
# ============================================================================

def edge_feature_info(dataset, relation_name="edges"):
    """
    Build ONE global feature_info_mapping entry (key=0) by scanning all graphs.

    Assumptions for QM9 in PyG:
      - data.edge_attr is shape (E, 4) one-hot bond type -> bond_type = argmax(edge_attr)
    """
    total_num_nodes = 0
    all_vals = set()

    for g in dataset:
        # total nodes across all graphs
        total_num_nodes += int(g.num_nodes)

        if getattr(g, "edge_attr", None) is None or g.edge_attr.numel() == 0:
            continue

        # bond type from one-hot edge_attr
        vals = torch.argmax(g.edge_attr, dim=1).detach().cpu().tolist()
        all_vals.update(int(v) for v in vals)

    unique_values = sorted(all_vals)
    value_to_index = {int(v): i for i, v in enumerate(unique_values)}
    index_to_value = {i: int(v) for i, v in enumerate(unique_values)}

    feature_info_mapping = {
        0: {
            "relation_name": relation_name,
            "feature_name": "bond_type",
            "num_nodes": int(total_num_nodes),
            "num_unique_values": len(unique_values),
            "unique_values": unique_values,
            "value_to_index": value_to_index,
            "index_to_value": index_to_value,
        }
    }

    return feature_info_mapping


def scipy_adj_to_torch(adj_csr, device=None, dtype=torch.float32):
    """scipy CSR -> torch dense (NxN)"""
    if sparse.issparse(adj_csr):
        A = adj_csr.toarray()
    else:
        A = np.asarray(adj_csr)
    return torch.tensor(A, dtype=dtype, device=device)


def nodefeat_to_torch(x_np, device=None, dtype=torch.long):
    """numpy (NxF) -> torch (NxF)"""
    return torch.tensor(np.asarray(x_np), dtype=dtype, device=device)


def build_edge_feature_tensors_for_graph(edge_array, n_nodes, feature_info_mapping, device=None, dtype=torch.float32):
    """
    edge_array: numpy array shape (E, 3) where columns are [src, dst, feat_value]
    feature_info_mapping: dict like {0: {'num_unique_values':3, 'value_to_index':{0:0,1:1,2:2}, ...}, ...}
    
    Returns:
        edge_tensors: list where edge_tensors[f] is a torch tensor (C_f, N, N) one-hot adjacency per value
        per_graph_feature_info_mapping: same mapping but with num_nodes updated to n_nodes
    """
    # make a per-graph copy (so num_nodes matches this graph)
    per_graph_feature_info_mapping = copy.deepcopy(feature_info_mapping)
    for f in per_graph_feature_info_mapping:
        per_graph_feature_info_mapping[f]["num_nodes"] = int(n_nodes)

    # edge_array could be empty
    if edge_array is None:
        edge_array = np.zeros((0, 3), dtype=int)
    else:
        edge_array = np.asarray(edge_array)

    edge_tensors = []

    # Build for each feature key using the edge_array (feature value is in col 2)
    for f_idx in sorted(per_graph_feature_info_mapping.keys()):
        info = per_graph_feature_info_mapping[f_idx]
        C = int(info["num_unique_values"])
        v2i = info["value_to_index"]

        # (C, N, N)
        T = torch.zeros((C, n_nodes, n_nodes), dtype=dtype, device=device)

        if edge_array.shape[0] > 0:
            src = edge_array[:, 0].astype(int)
            dst = edge_array[:, 1].astype(int)
            val = edge_array[:, 2]

            # fill one-hot adjacency
            for s, d, v in zip(src, dst, val):
                if 0 <= s < n_nodes and 0 <= d < n_nodes:
                    # map raw value -> channel index
                    if v in v2i:
                        c = int(v2i[v])
                        T[c, s, d] = 1.0

        edge_tensors.append(T)

    return edge_tensors, per_graph_feature_info_mapping


class DataLoader:
    """
    A class to load datasets (Cora or QM9).
    
    For Cora:
    - Citation network: nodes=papers, edges=citations
    - Node features are bag-of-words representations
    - Node labels are paper categories
    
    For QM9:
    - Molecular graphs: nodes=atoms, edges=bonds
    - Node features: atom type, number of hydrogens
    - Edge features: bond type
    - Multiple graphs (molecules)
    
    This class handles all data preprocessing including feature reduction.
    """
    
    def __init__(self, dataset_type: str = 'cora', n_components: int = 5, random_seed: int = 0, 
                 max_graphs: int = None, device: str = 'cpu'):
        """
        Initialize the DataLoader.
        
        Args:
            dataset_type: 'cora' or 'qm9'
            n_components: Number of top features to keep after reduction (for Cora)
            random_seed: Random seed for reproducibility in feature reduction
            max_graphs: Maximum number of graphs to load (for QM9, None = all)
            device: Device for tensors ('cpu' or 'cuda')
        """
        self.dataset_type = dataset_type.lower()
        self.n_components = n_components
        self.random_seed = random_seed
        self.max_graphs = max_graphs
        self.device = device
        
        # Cache management
        self.cache_dir = Path('./preprocessed_data')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._graph_data_list_cache = None  # Cache for graph_data_list
        
        # Common data variables
        self.features = None
        self.labels = None
        self.adjacency_matrix = None
        
        # Cora-specific
        self.graph = None
        self.num_classes = None
        self.edges = None
        self.num_nodes = None
        self.num_features = None
        self.features_raw = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.important_feature_indices = None
        
        # QM9-specific
        self.list_adj = []
        self.list_node_feature = []
        self.list_edge_feature = []
        self.feature_info_mapping = None
    
    def _get_cache_filepath(self) -> Path:
        """
        Get the cache file path for preprocessed graph data list.
        
        Returns:
            Path to the cache pickle file
        """
        if self.dataset_type == 'cora':
            filename = f"cora_graph_data.pkl"
        elif self.dataset_type == 'qm9':
            if self.max_graphs is None:
                filename = f"qm9_all_graph_data.pkl"
            else:
                filename = f"qm9_{self.max_graphs}_graph_data.pkl"
        else:
            filename = f"{self.dataset_type}_graph_data.pkl"
        
        return self.cache_dir / filename
    
    def _load_from_cache(self) -> bool:
        """
        Try to load preprocessed graph_data_list from cache.
        
        Returns:
            True if successfully loaded from cache, False otherwise
        """
        cache_file = self._get_cache_filepath()
        
        if not cache_file.exists():
            return False
        
        try:
            print(f"  📦 Loading preprocessed data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self._graph_data_list_cache = cached_data['graph_data_list']
            
            # Also restore metadata for get_data()
            if self.dataset_type == 'cora':
                self.num_nodes = cached_data.get('num_nodes')
                self.num_features = cached_data.get('num_features')
            elif self.dataset_type == 'qm9':
                self.feature_info_mapping = cached_data.get('feature_info_mapping')
            
            print(f"  ✓ Successfully loaded from cache")
            return True
            
        except Exception as e:
            print(f"  ⚠ Failed to load from cache: {e}")
            print(f"  → Will reprocess dataset")
            return False
    
    def _save_to_cache(self, graph_data_list: List[Dict]):
        """
        Save preprocessed graph_data_list to cache.
        
        Args:
            graph_data_list: List of graph data dictionaries
        """
        cache_file = self._get_cache_filepath()
        
        try:
            print(f"  💾 Saving preprocessed data to cache: {cache_file}")
            
            # Prepare data to cache
            cache_data = {
                'graph_data_list': graph_data_list,
                'dataset_type': self.dataset_type,
                'device': self.device,
            }
            
            # Add metadata
            if self.dataset_type == 'cora':
                cache_data['num_nodes'] = self.num_nodes
                cache_data['num_features'] = self.num_features
            elif self.dataset_type == 'qm9':
                cache_data['feature_info_mapping'] = self.feature_info_mapping
                cache_data['max_graphs'] = self.max_graphs
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"  ✓ Saved to cache")
            
        except Exception as e:
            print(f"  ⚠ Failed to save to cache: {e}")
            # Non-fatal, continue execution
        
    def load_data(self) -> None:
        """
        Load the dataset and perform all preprocessing.
        Checks cache first, loads from scratch if cache not found.
        """
        # Try to load from cache first
        if self._load_from_cache():
            return
        
        # Cache not found or failed to load, process from scratch
        if self.dataset_type == 'cora':
            self._load_cora()
        elif self.dataset_type == 'qm9':
            self._load_qm9()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}. Use 'cora' or 'qm9'.")
    
    def _load_cora(self) -> None:
        """Load Cora dataset using DGL and perform all preprocessing."""
        # Load the dataset
        dataset = CoraGraphDataset()
        self.graph = dataset[0]
        
        # Extract features
        features = self.graph.ndata['feat']
        
        # Binarize features and keep as tensor
        features_binary = torch.where(features > 0, 
                                      torch.ones_like(features), 
                                      torch.zeros_like(features)).float()
        self.features_raw = features_binary
        
        # Extract labels as tensor
        self.labels = self.graph.ndata['label']
        self.num_classes = dataset.num_classes
        
        # Perform feature reduction
        self._reduce_features()
        
        # Extract masks as tensors
        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']
        
        # Extract edges as tensor
        src, dst = self.graph.edges()
        self.edges = torch.stack([src, dst], dim=1)
        
        # Get number of nodes
        self.num_nodes = self.graph.num_nodes()
        
        # Create adjacency matrix
        self._create_adjacency_matrix()
    
    def _load_qm9(self) -> None:
        """Load QM9 dataset and preprocess all molecules."""
        print(f"  Loading QM9 dataset from ./data/QM9...")
        data = QM9(root="./data/QM9")
        
        # Build global feature info mapping
        print(f"  Building edge feature mapping...")
        self.feature_info_mapping = edge_feature_info(data)
        
        # Process each molecule
        num_graphs = len(data) if self.max_graphs is None else min(self.max_graphs, len(data))
        print(f"  Processing {num_graphs} molecules...")
        
        for i in range(num_graphs):
            mol = data[i]
            
            # -------------------------
            # Adjacency
            # -------------------------
            N = mol.num_nodes
            edge_index = mol.edge_index
            
            adj = sparse.csr_matrix(
                (np.ones(edge_index.size(1)),
                 (edge_index[0].numpy(), edge_index[1].numpy())),
                shape=(N, N)
            )
            self.list_adj.append(adj)
            
            # -------------------------
            # Node features (DB-style)
            # -------------------------
            X = mol.x  # (N, 11)
            atom_type = torch.argmax(X[:, 0:5], dim=1)        # 0–4
            num_h = torch.clamp(X[:, 10].long(), max=3)       # 0–3
            
            node_feats = torch.stack([atom_type, num_h], dim=1)  # (N,2)
            self.list_node_feature.append(node_feats.numpy())
            
            # -------------------------
            # Edge features (bond_type)
            # -------------------------
            if mol.edge_attr is not None and mol.edge_attr.size(0) > 0:
                bond_type = torch.argmax(mol.edge_attr, dim=1)  # 0–3
                src = edge_index[0]
                dst = edge_index[1]
                
                edge_feats = torch.stack([src, dst, bond_type], dim=1)  # (E_dir,3)
                self.list_edge_feature.append(edge_feats.numpy())
            else:
                self.list_edge_feature.append(None)
            
            if (i + 1) % 1000 == 0:
                print(f"    Processed {i + 1}/{num_graphs} molecules...")
        
        print(f"  ✓ Loaded {len(self.list_adj)} molecules")
    
    def _reduce_features(self) -> None:
        """
        Reduce features using ExtraTreesClassifier feature importance,
        then concatenate with labels column (Cora only).
        """
        # Reduce features
        x_reduced, important_feats = reduce_node_features(
            self.features_raw,
            self.labels,
            self.random_seed,
            self.n_components
        )
        
        # Store important feature indices
        self.important_feature_indices = important_feats
        
        # Add labels as a column (convert to numpy for concatenation)
        labels_column = self.labels.numpy().reshape(-1, 1)
        x_with_labels = np.concatenate([x_reduced, labels_column], axis=1)
        
        # Store as numpy array (for compatibility with motif counter)
        self.features = x_with_labels
        self.num_features = self.features.shape[1]
        
    def _create_adjacency_matrix(self) -> None:
        """Create adjacency matrix from the graph as tensor (Cora only)."""
        self.adjacency_matrix = self.graph.adjacency_matrix().to_dense().float()
            
    def get_data(self) -> Dict:
        """
        Get all data as a dictionary.
        Note: Some fields may be None if data was loaded from cache.
        
        Returns:
            Dictionary containing all dataset components
        """
        if self.dataset_type == 'cora':
            return {
                'graph': self.graph,
                'features': self.features,  # Reduced features with labels column
                'features_raw': self.features_raw,  # Original binary features
                'labels': self.labels,
                'num_classes': self.num_classes,
                'adjacency_matrix': self.adjacency_matrix,
                'edges': self.edges,
                'num_nodes': self.num_nodes,
                'num_features': self.num_features,
                'train_mask': self.train_mask,
                'val_mask': self.val_mask,
                'test_mask': self.test_mask,
                'important_feature_indices': self.important_feature_indices
            }
        elif self.dataset_type == 'qm9':
            return {
                'list_adj': self.list_adj,
                'list_node_feature': self.list_node_feature,
                'list_edge_feature': self.list_edge_feature,
                'feature_info_mapping': self.feature_info_mapping,
                'num_graphs': len(self.list_adj) if self.list_adj else (len(self._graph_data_list_cache) if self._graph_data_list_cache else 0)
            }
    
    def get_graph_data_list(self) -> list:
        """
        Get graph data formatted for motif counting as a list.
        Uses cache if available, creates and caches if not.
        
        Returns:
            List containing graph_data dictionaries ready for motif counting.
            - For Cora: single-item list
            - For QM9: multiple items (one per molecule)
        """
        # Check if already cached in memory
        if self._graph_data_list_cache is not None:
            print(f"  ✓ Using cached graph_data_list ({len(self._graph_data_list_cache)} graphs)")
            return self._graph_data_list_cache
        
        # Build graph_data_list from scratch
        print(f"  Building graph_data_list...")
        
        if self.dataset_type == 'cora':
            graph_data = {
                'adjacency_matrix': self.adjacency_matrix,
                'features': self.features,  # Already reduced with labels
                'labels': None
            }
            graph_data_list = [graph_data]
        
        elif self.dataset_type == 'qm9':
            graph_data_list = []
            
            num_graphs = len(self.list_adj)
            print(f"  Converting {num_graphs} molecules to graph_data format...")
            
            for g in range(num_graphs):
                # Build per-graph adjacency + node features
                big_adj = scipy_adj_to_torch(self.list_adj[g], device=self.device)
                big_node_features = nodefeat_to_torch(self.list_node_feature[g], device=self.device)
                n_nodes = big_adj.shape[0]
                
                # Build per-graph edge features
                big_edge_features_tensor, per_graph_fim = build_edge_feature_tensors_for_graph(
                    self.list_edge_feature[g],
                    n_nodes,
                    self.feature_info_mapping,
                    device=self.device
                )
                
                graph_data = {
                    'adjacency_matrix': big_adj,
                    'features': big_node_features,
                    'labels': big_edge_features_tensor,
                    'feature_info_mapping': per_graph_fim
                }
                
                graph_data_list.append(graph_data)
                
                if (g + 1) % 1000 == 0:
                    print(f"    Converted {g + 1}/{num_graphs} molecules...")
            
            print(f"  ✓ Converted all {num_graphs} molecules")
        
        # Cache the result
        self._graph_data_list_cache = graph_data_list
        self._save_to_cache(graph_data_list)
        
        return graph_data_list
