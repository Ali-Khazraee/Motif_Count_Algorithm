# main.py

import numpy as np
import torch
import random
import argparse

from data import DataLoader
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
    print("\n[Step 1/3] Loading and preprocessing graph data...")
    data_loader = DataLoader(n_components=5, random_seed=0)
    data_loader.load_data()
    data = data_loader.get_data()
    print(f"  ✓ Loaded {data['num_nodes']} nodes, {len(data['edges'])} edges")
    print(f"  ✓ Reduced to {data['num_features']} features (including label)")
    
    # ========== Step 2: Initialize Motif Store ==========
    # This automatically handles pickle load/save
    print(f"\n[Step 2/3] Initializing motif store...")
    try:
        motif_store = RuleBasedMotifStore(
            database_name=args.database_name,
            args=args
        )

    except Exception as e:
        print(f"\n✗ Error initializing motif store: {e}")
        return
    
    # ========== Step 3: Count Motifs ==========
    print("\n[Step 3/3] Counting motifs...")
    try:
        motif_counter = RelationalMotifCounter(
            database_name=args.database_name,
            args=args
        )
        
        # Prepare graph data for counting
        graph_data = {
            'adjacency_matrix': data['adjacency_matrix'],
            'features': data['features'],  # Already reduced with labels
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
