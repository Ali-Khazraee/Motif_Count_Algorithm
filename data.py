# data.py

import numpy as np
import torch
from typing import Dict, Tuple
import dgl
from dgl.data import CoraGraphDataset
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd


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


class DataLoader:
    """
    A class to load the Cora dataset using DGL.
    
    The Cora dataset is a citation network where:
    - Nodes represent papers
    - Edges represent citations
    - Node features are bag-of-words representations
    - Node labels are paper categories
    
    This class handles all data preprocessing including feature reduction.
    """
    
    def __init__(self, n_components: int = 5, random_seed: int = 0):
        """
        Initialize the DataLoader.
        
        Args:
            n_components: Number of top features to keep after reduction
            random_seed: Random seed for reproducibility in feature reduction
        """
        # Data variables
        self.graph = None
        self.features = None
        self.features_raw = None  # Original features before reduction
        self.labels = None
        self.num_classes = None
        self.adjacency_matrix = None
        self.edges = None
        self.num_nodes = None
        self.num_features = None
        
        # Train/val/test masks (provided by DGL)
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        
        # Feature reduction parameters
        self.n_components = n_components
        self.random_seed = random_seed
        self.important_feature_indices = None
        
    def load_data(self) -> None:
        """Load the Cora dataset using DGL and perform all preprocessing."""
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
    
    def _reduce_features(self) -> None:
        """
        Reduce features using ExtraTreesClassifier feature importance,
        then concatenate with labels column.
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
        """Create adjacency matrix from the graph as tensor."""
        self.adjacency_matrix = self.graph.adjacency_matrix().to_dense().float()
            
    def get_data(self) -> Dict:
        """
        Get all data as a dictionary.
        
        Returns:
            Dictionary containing all dataset components
        """
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
    
    def get_graph_data_list(self) -> list:
        """
        Get graph data formatted for motif counting as a list.
        
        Returns:
            List containing graph_data dictionaries ready for motif counting.
            Currently returns a single-item list, but structured for future expansion.
        """
        graph_data = {
            'adjacency_matrix': self.adjacency_matrix,
            'features': self.features,  # Already reduced with labels
            'labels': None
        }
        
        return [graph_data]
