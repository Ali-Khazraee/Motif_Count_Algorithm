# motif_store.py

import torch
import pickle
from typing import Dict, List, Any
from pathlib import Path


class RuleBasedMotifStore:
    """
    Container for motif definitions stored as relational database rules.
    Holds all rules, matrices, indices, and metadata needed for motif counting.
    """
    
    def __init__(self):
        # Rule-related data structures
        self.rules: List = []
        self.multiples: List = []
        self.states: List = []
        self.values: List = []
        self.prunes: List = []
        
        # Structural metadata for rules
        self.functors: Dict = {}
        self.variables: Dict = {}
        self.nodes: Dict = {}
        self.masks: Dict = {}
        
        # Index structures for efficient computation
        self.base_indices: List = []
        self.mask_indices: List = []
        self.sort_indices: List = []
        self.stack_indices: List = []
        
        # Database entities and relations
        self.entities: Dict = {}
        self.relations: Dict = {}
        self.attributes: Dict = {}
        self.keys: Dict = {}
        self.indices: Dict = {}
        self.matrices: Dict = {}
        
        # Feature mapping structures
        self.entity_feature_columns: Dict = {}
        self.relation_feature_columns: Dict = {}
        self.feature_info_mapping: Dict = {}
        
        # Configuration
        self.device = "cuda"
        self.num_nodes_graph: int = 0
        self.args = None
        
    @property
    def num_motifs(self) -> int:
        """Total number of motif rules."""
        return len(self.rules)
    
    def to_device(self, device: str):
        """Move all tensor matrices to specified device."""
        self.device = device
        for key in self.matrices:
            if isinstance(self.matrices[key], torch.Tensor):
                self.matrices[key] = self.matrices[key].to(device)
    
    def save(self, file_path: str):
        """
        Save the motif store to a pickle file.
        
        Args:
            file_path: Path where to save the .pkl file
        """
        # Convert torch tensors to CPU before saving
        matrices_cpu = {}
        for key, matrix in self.matrices.items():
            if isinstance(matrix, torch.Tensor):
                matrices_cpu[key] = matrix.cpu()
            else:
                matrices_cpu[key] = matrix
        
        # Prepare data dictionary matching the load_setup_variables format
        important_variables = {
            "entities": self.entities,
            "relations": self.relations,
            "keys": self.keys,
            "matrices": matrices_cpu,
            "rules": self.rules,
            "indices": self.indices,
            "attributes": self.attributes,
            "base_indices": self.base_indices,
            "mask_indices": self.mask_indices,
            "sort_indices": self.sort_indices,
            "stack_indices": self.stack_indices,
            "values": self.values,
            "prunes": self.prunes,
            "functors": self.functors,
            "variables": self.variables,
            "nodes": self.nodes,
            "states": self.states,
            "masks": self.masks,
            "multiples": self.multiples,
            "entity_feature_columns": self.entity_feature_columns,
            "relation_feature_columns": self.relation_feature_columns,
            "feature_info_mapping": self.feature_info_mapping,
            "num_nodes_graph": self.num_nodes_graph,
        }
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(file_path, "wb") as f:
            pickle.dump(important_variables, f)
        
        print(f"✓ Motif store saved to: {file_path}")
    
    @classmethod
    def load(cls, file_path: str, device: str = "cuda"):
        """
        Load a motif store from a pickle file.
        
        Args:
            file_path: Path to the .pkl file
            device: Device to load tensors to ('cuda' or 'cpu')
            
        Returns:
            Loaded RuleBasedMotifStore instance
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")
        
        print(f"📦 Loading motif store from pickle file: {file_path}")
        
        with open(file_path, "rb") as f:
            important_variables = pickle.load(f)
        
        # Create new store instance
        store = cls()
        
        # Assign loaded variables to class attributes (matching load_setup_variables)
        store.entities = important_variables["entities"]
        store.relations = important_variables["relations"]
        store.keys = important_variables["keys"]
        store.rules = important_variables["rules"]
        store.indices = important_variables["indices"]
        store.attributes = important_variables["attributes"]
        store.base_indices = important_variables["base_indices"]
        store.mask_indices = important_variables["mask_indices"]
        store.sort_indices = important_variables["sort_indices"]
        store.stack_indices = important_variables["stack_indices"]
        store.values = important_variables["values"]
        store.prunes = important_variables["prunes"]
        store.functors = important_variables["functors"]
        store.variables = important_variables["variables"]
        store.nodes = important_variables["nodes"]
        store.states = important_variables["states"]
        store.masks = important_variables["masks"]
        store.multiples = important_variables["multiples"]
        store.entity_feature_columns = important_variables.get("entity_feature_columns", {})
        store.relation_feature_columns = important_variables.get("relation_feature_columns", {})
        store.feature_info_mapping = important_variables.get("feature_info_mapping", {})
        store.num_nodes_graph = important_variables.get("num_nodes_graph", 0)
        
        # Load matrices and move to device
        store.matrices = {}
        for key, matrix in important_variables["matrices"].items():
            if isinstance(matrix, torch.Tensor):
                store.matrices[key] = matrix.to(device)
            else:
                store.matrices[key] = matrix
        
        store.device = device
        
        print(f"✓ Loaded {store.num_motifs} motif rules from pickle file")
        
        return store
    
    def __repr__(self):
        return f"RuleBasedMotifStore(num_motifs={self.num_motifs}, num_entities={len(self.entities)}, num_relations={len(self.relations)})"