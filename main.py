# main.py

import numpy as np
import torch
import random
import argparse

from data import DataLoader
from ds_to_db import reduce_node_features
from motif_store import RuleBasedMotifStore
from motif_counter import RelationalMotifCounter


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Motif Count')
    
    # Database configuration
    parser.add_argument('--database_name', type=str, default='cora',
                    help='Name of the database to use (e.g., cora, citeseer, imdb)')
    
    parser.add_argument('-dataset', dest="dataset", default="Cora_dgl",
                       help="Dataset name for display purposes")

    parser.add_argument('--graph_type', type=str, default='homogeneous',
                       choices=['homogeneous', 'heterogeneous'],
                       help='Graph type for motif counting')
    
    # Motif counting arguments
    parser.add_argument('--motif_loss', type=bool, default=True,
                       help='Enable motif loss term in objective function')
    parser.add_argument('--rule_prune', type=bool, default=False,
                       help='Enable rule pruning in motif counting')
    parser.add_argument('--rule_weight', type=bool, default=False,
                       help='Enable rule weighting (requires rule_prune=True)')
    parser.add_argument('--test_local_mults', type=bool, default=True,
                       help='Enable validation of motif count number against FactorBase outputs')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for computation')
    
    return parser.parse_args()


def set_random_seeds(seed=0):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """
    Main function with automatic pickle management.
    
    Usage Example:
    --------------
    python main.py --database_name cora
    """
    
    # ========== Configuration ==========
    print("="*60)
    print("Motif Counting Pipeline - Refactored Architecture")
    print("="*60)
    
    args = parse_arguments()
    set_random_seeds(seed=0)
    
    print(f"\nConfiguration:")
    print(f"  Database: {args.database_name}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Graph type: {args.graph_type}")
    print(f"  Device: {args.device}")
    
    # ========== Step 1: Load Graph Data ==========
    print("\n[Step 1/4] Loading graph data...")
    data_loader = DataLoader()
    data_loader.load_data()
    data = data_loader.get_data()
    print(f"  ✓ Loaded {data['num_nodes']} nodes, {len(data['edges'])} edges")
    
    # ========== Step 2: Feature Reduction ==========
    print("\n[Step 2/4] Reducing features...")
    x_reduced, important_feats = reduce_node_features(
        data['features'], 
        data['labels'], 
        random_seed=0, 
        n_components=5
    )
    
    labels_column = data['labels'].numpy().reshape(-1, 1)
    x_with_labels = np.concatenate([x_reduced, labels_column], axis=1)
    print(f"  ✓ Reduced to {x_with_labels.shape[1]} features (including label)")
    
    # ========== Step 3: Initialize Motif Store ==========
    # This automatically handles pickle load/save
    print(f"\n[Step 3/4] Initializing motif store...")
    try:
        motif_store = RuleBasedMotifStore(
            database_name=args.database_name,
            args=args
        )
        print(f"  ✓ Loaded {motif_store.num_motifs} motif rules")
        print(f"  ✓ {len(motif_store.entities)} entity tables")
        print(f"  ✓ {len(motif_store.relations)} relation tables")
    except Exception as e:
        print(f"\n✗ Error initializing motif store: {e}")
        return
    
    # ========== Step 4: Count Motifs ==========
    print("\n[Step 4/4] Counting motifs...")
    try:
        motif_counter = RelationalMotifCounter(
            database_name=args.database_name,
            args=args
        )
        
        # Prepare graph data for counting
        graph_data = {
            'adjacency_matrix': data['adjacency_matrix'],
            'features': x_with_labels,
            'labels': None
        }
        
        print("\n" + "="*60)
        print("Performing motif counting...")
        print("="*60)
        
        motif_counts = motif_counter.count(graph_data)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
        print(f"Total motifs counted: {len(motif_counts)}")
        
    except Exception as e:
        print(f"\n✗ Error during motif counting: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
