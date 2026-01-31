# motif_dataset.py

from typing import Dict, List
from motif_counter import RelationalMotifCounter


class MotifAugmentedDataset:
    """
    Wraps a graph dataset and augments each graph with motif counts.
    Follows the standard dataset wrapper pattern.
    """
    
    def __init__(self, base_data: Dict, motif_counter: RelationalMotifCounter):
        """
        Initialize the augmented dataset.
        
        Args:
            base_data: Dictionary with graph data (adjacency, features, labels)
            motif_counter: RelationalMotifCounter instance for counting motifs
        """
        self.base_data = base_data
        self.motif_counter = motif_counter
        
        # Cache for motif counts (computed lazily)
        self._motif_counts = None
    
    @property
    def motif_counts(self) -> List:
        """
        Lazy computation of motif counts.
        Counts are computed once and cached.
        """
        if self._motif_counts is None:
            print("Computing motif counts...")
            self._motif_counts = self.motif_counter.count(self.base_data)
        return self._motif_counts
    
    def get_augmented_data(self) -> Dict:
        """
        Returns the original data augmented with motif counts.
        
        Returns:
            Dictionary with original data plus 'motif_counts' field
        """
        augmented = self.base_data.copy()
        augmented['motif_counts'] = self.motif_counts
        return augmented
    
    def get_motif_vector(self) -> List:
        """
        Returns just the motif count vector.
        
        Returns:
            List of motif counts
        """
        return self.motif_counts
    
    def __repr__(self):
        if self._motif_counts is not None:
            return f"MotifAugmentedDataset(num_motifs={len(self._motif_counts)}, computed=True)"
        else:
            return f"MotifAugmentedDataset(computed=False)"