# main.py

import numpy as np
import torch
import random
import argparse
from pathlib import Path

from data import DataLoader
from ds_to_db import reduce_node_features
from motif_reader import DatabaseMotifReader
from motif_counter import RelationalMotifCounter
from motif_dataset import MotifAugmentedDataset


def parse_arguments():
    """Parse command line arguments with clear PKL/DB options."""
    parser = argparse.ArgumentParser(description='Motif Count')
    
    parser.add_argument('-dataset', dest="dataset", default="Cora_dgl",
                       help="Dataset name")

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
    # ========== PKL vs DB ARGUMENTS ==========
    parser.add_argument('--use_pkl', type=bool, default=False,
                       help='Load motif store from pickle file (mutually exclusive with --connect_db)')
    parser.add_argument('--connect_db', type=bool, default=True,
                       help='Connect to database to read motif store (mutually exclusive with --use_pkl)')
    parser.add_argument('--pkl_path', type=str, default='./motif_stores',
                       help='Path to pickle file (for loading with --use_pkl or saving with --save_pkl_after_db)')
    parser.add_argument('--save_pkl_after_db', type=bool, default=False,
                       help='Save motif store to pickle after reading from database (requires --pkl_path)')
    parser.add_argument('--pkl_dir', type=str, default='./motif_stores',
                       help='Default directory for pickle files (used if --pkl_path not specified)')
    
    return parser.parse_args()


def set_random_seeds(seed=0):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def validate_args(args):
    """Validate argument combinations."""
    # Check mutually exclusive options
    if args.use_pkl and args.connect_db:
        raise ValueError(
            "Arguments --use_pkl and --connect_db are mutually exclusive. "
            "Please use only one."
        )
    
    # Must specify at least one mode
    if not args.use_pkl and not args.connect_db:
        raise ValueError(
            "Must specify either --use_pkl True or --connect_db True"
        )
    
    # Check pkl_path requirements
    if args.use_pkl and args.pkl_path is None:
        raise ValueError(
            "When using --use_pkl True, you must specify --pkl_path"
        )
    
    if args.save_pkl_after_db and args.pkl_path is None:
        raise ValueError(
            "When using --save_pkl_after_db True, you must specify --pkl_path"
        )


def determine_mode_and_path(args):
    """
    Determine the mode ('pkl' or 'db') and pickle path.
    
    Returns:
        mode: 'pkl' or 'db'
        pkl_path: Path to pickle file (or None)
    """
    if args.use_pkl:
        # Mode 1: Load from PKL
        mode = 'pkl'
        pkl_path = args.pkl_path
        
        # Ensure full path
        if not Path(pkl_path).is_absolute():
            pkl_path = str(Path(args.pkl_dir) / pkl_path)
        
        return mode, pkl_path
    
    elif args.connect_db:
        # Mode 2: Connect to DB
        mode = 'db'
        pkl_path = None
        
        # If saving after DB read
        if args.save_pkl_after_db:
            pkl_path = args.pkl_path
            
            # Ensure full path
            if not Path(pkl_path).is_absolute():
                pkl_dir = Path(args.pkl_dir)
                pkl_dir.mkdir(parents=True, exist_ok=True)
                pkl_path = str(pkl_dir / pkl_path)
        
        return mode, pkl_path
    
    else:
        raise ValueError("Invalid configuration")


def main():
    """
    Main function with clear PKL/DB logic:
    
    Usage Examples:
    ---------------
    # Load from PKL:
    python main.py --use_pkl True --pkl_path "acm_no_prune.pkl"
    
    # Connect to DB and save to PKL:
    python main.py --connect_db True --save_pkl_after_db True --pkl_path "cora_output.pkl"
    
    # Connect to DB without saving:
    python main.py --connect_db True
    """
    
    # ========== Configuration ==========
    print("="*60)
    print("Motif Counting Pipeline - Clean Architecture")
    print("="*60)
    
    args = parse_arguments()
    
    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        print(f"\n✗ Argument Error: {e}\n")
        print("Usage Examples:")
        print("  Load from PKL:")
        print("    python main.py --use_pkl True --pkl_path 'acm_no_prune.pkl'")
        print("\n  Connect to DB and save:")
        print("    python main.py --connect_db True --save_pkl_after_db True --pkl_path 'output.pkl'")
        print("\n  Connect to DB without saving:")
        print("    python main.py --connect_db True")
        return
    
    set_random_seeds(seed=0)
    
    # Determine mode and path
    mode, pkl_path = determine_mode_and_path(args)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Mode: {'Load from PKL' if mode == 'pkl' else 'Connect to Database'}")
    if pkl_path:
        print(f"  Pickle path: {pkl_path}")
    if mode == 'db' and args.save_pkl_after_db:
        print(f"  Will save to PKL after reading from DB")
    
    # ========== Step 1: Load Graph Data ==========
    print("\n[Step 1/5] Loading graph data...")
    data_loader = DataLoader()
    data_loader.load_data()
    data = data_loader.get_data()
    print(f"  ✓ Loaded {data['num_nodes']} nodes, {len(data['edges'])} edges")
    
    # ========== Step 2: Feature Reduction ==========
    print("\n[Step 2/5] Reducing features...")
    x_reduced, important_feats = reduce_node_features(
        data['features'], 
        data['labels'], 
        random_seed=0, 
        n_components=5
    )
    
    labels_column = data['labels'].numpy().reshape(-1, 1)
    x_with_labels = np.concatenate([x_reduced, labels_column], axis=1)
    print(f"  ✓ Reduced to {x_with_labels.shape[1]} features (including label)")
    
    # ========== Step 3: Read Motif Rules ==========
    print(f"\n[Step 3/5] Reading motif rules...")
    
    motif_reader = DatabaseMotifReader(
        dataset_name=args.dataset, 
        args=args
    )
    
    try:
        motif_store = motif_reader.read(mode=mode, pkl_path=pkl_path)
        
        print(f"  ✓ Loaded {motif_store.num_motifs} motif rules")
        print(f"  ✓ {len(motif_store.entities)} entity tables")
        print(f"  ✓ {len(motif_store.relations)} relation tables")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return
    except RuntimeError as e:
        print(f"\n✗ Error: {e}")
        return
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return
    
    # ========== Step 4: Create Motif Counter ==========
    print("\n[Step 4/5] Creating motif counter...")
    motif_counter = RelationalMotifCounter(motif_store)
    print(f"  ✓ Motif counter initialized")
    
    # ========== Step 5: Create Augmented Dataset ==========
    print("\n[Step 5/5] Creating augmented dataset...")
    graph_data = {
        'adjacency_matrix': data['adjacency_matrix'],
        'features': x_with_labels,
        'labels': None
    }
    
    dataset = MotifAugmentedDataset(
        base_data=graph_data,
        motif_counter=motif_counter
    )
    print(f"  ✓ Dataset augmented and ready")
    
    # ========== Count Motifs ==========
    print("\n" + "="*60)
    print("Counting motifs...")
    print("="*60)
    motif_counts = dataset.motif_counts
    
    # # ========== Display Results ==========
    # print("\n" + "="*60)
    # print("Motif Counting Results")
    # print("="*60)
    # print(f"Total motifs counted: {len(motif_counts)}")
    # print(f"\nMotif counts:")
    # for i, count in enumerate(motif_counts):
    #     if isinstance(count, torch.Tensor):
    #         print(f"  Motif {i+1}: {count.item():.2f}")
    #     else:
    #         print(f"  Motif {i+1}: {count:.2f}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()