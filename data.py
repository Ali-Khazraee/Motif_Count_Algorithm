import numpy as np
import torch
from typing import Dict
import dgl
from dgl.data import CoraGraphDataset


class DataLoader:
    """
    A class to load the Cora dataset using DGL.
    
    The Cora dataset is a citation network where:
    - Nodes represent papers
    - Edges represent citations
    - Node features are bag-of-words representations
    - Node labels are paper categories
    """
    
    def __init__(self):
        """Initialize the DataLoader."""
        # Data variables
        self.graph = None
        self.features = None
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
        
    def load_data(self) -> None:
        """Load the Cora dataset using DGL."""
        # print("Loading Cora dataset from DGL...")
        
        # Load the dataset
        dataset = CoraGraphDataset()
        self.graph = dataset[0]
        
        # Extract features
        features = self.graph.ndata['feat']
        
        # Binarize features and keep as tensor
        self.features = torch.where(features > 0, 
                                    torch.ones_like(features), 
                                    torch.zeros_like(features)).float()
        self.num_features = self.features.shape[1]
        
        # Extract labels as tensor
        self.labels = self.graph.ndata['label']
        self.num_classes = dataset.num_classes
        
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
        
        # print(f"Dataset loaded successfully!")
        # print(f"Number of nodes: {self.num_nodes}")
        # print(f"Number of features: {self.num_features}")
        # print(f"Number of classes: {self.num_classes}")
        # print(f"Number of edges: {len(self.edges)}")
        # print(f"Train/Val/Test: {self.train_mask.sum()}/{self.val_mask.sum()}/{self.test_mask.sum()}")
        
    def _create_adjacency_matrix(self) -> None:
        """Create adjacency matrix from the graph as tensor."""
        self.adjacency_matrix = self.graph.adjacency_matrix().to_dense().float()
            
    def get_data(self) -> Dict:
        """
        Get all data as a dictionary.
        
        Returns:
            Dictionary containing all dataset components (all tensors)
        """
        return {
            'graph': self.graph,
            'features': self.features,
            'labels': self.labels,
            'num_classes': self.num_classes,
            'adjacency_matrix': self.adjacency_matrix,
            'edges': self.edges,
            'num_nodes': self.num_nodes,
            'num_features': self.num_features,
            'train_mask': self.train_mask,
            'val_mask': self.val_mask,
            'test_mask': self.test_mask
        }