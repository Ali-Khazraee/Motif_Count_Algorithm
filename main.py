# main.py

import numpy as np
import torch
import random
import argparse
import time
from typing import List, Optional, Tuple, Dict


# ======================================================================== #
# Timing utility
# ======================================================================== #

class StepTimer:
    """
    Lightweight multi-step wall-clock timer.

    Usage
    -----
    timer = StepTimer()
    timer.start("Loading data")
    ...
    timer.stop("Loading data")

    timer.start("Counting motifs")
    ...
    timer.stop("Counting motifs")

    timer.report()
    """

    def __init__(self):
        self._starts:   Dict[str, float] = {}
        self._durations: Dict[str, float] = {}
        self._order:    List[str]         = []
        self._pipeline_start: float       = time.perf_counter()

    def start(self, label: str):
        if label not in self._order:
            self._order.append(label)
        self._starts[label] = time.perf_counter()

    def stop(self, label: str) -> float:
        elapsed = time.perf_counter() - self._starts[label]
        self._durations[label] = elapsed
        return elapsed

    @staticmethod
    def _fmt(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f}s"
        m, s = divmod(seconds, 60)
        if m < 60:
            return f"{int(m)}m {s:.1f}s"
        h, m = divmod(m, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s"

    def report(self):
        total = time.perf_counter() - self._pipeline_start
        bar   = "=" * 60
        print(f"\n{bar}")
        print("TIMING REPORT")
        print(bar)
        max_label = max((len(l) for l in self._order), default=20)
        for label in self._order:
            d = self._durations.get(label, 0.0)
            pct = (d / total * 100) if total > 0 else 0.0
            print(f"  {label:<{max_label}}  {self._fmt(d):>10}  ({pct:5.1f}%)")
        print(f"  {'-' * (max_label + 22)}")
        print(f"  {'TOTAL':<{max_label}}  {self._fmt(total):>10}  (100.0%)")
        print(bar)

from data import DataLoader
from data_preprocessor import DataPreprocessor          # ← NEW
from motif_store import RuleBasedMotifStore
from motif_counter import RelationalMotifCounter


def parse_arguments():
    parser = argparse.ArgumentParser(description='Motif Count')

    parser.add_argument('--dataset_type', type=str, default='qm9',
                        choices=['cora', 'qm9'])
    parser.add_argument('--max_graphs', type=int, default=None)
    parser.add_argument('--database_name', type=str, default='qm9')
    parser.add_argument('--graph_type', type=str, default='homogeneous',
                        choices=['homogeneous', 'heterogeneous'])
    parser.add_argument('--motif_loss', type=bool, default=True)
    parser.add_argument('--rule_prune', type=bool, default=False)
    parser.add_argument('--interactive', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    # Graph index selection: run counting only on a slice of graph_data_list
    parser.add_argument('--graph_index_start', type=int, default=None,
                        help='First graph index to count (inclusive). '
                             'Only valid when dataset has more than one graph.')
    parser.add_argument('--graph_index_end', type=int, default=None,
                        help='Last graph index to count (inclusive). '
                             'Only valid when dataset has more than one graph.')

    # Batched GPU counting for multi-graph datasets (e.g. QM9)
    parser.add_argument('--batch_size', type=int, default=50000,
                        help='Number of graphs to process simultaneously on GPU. '
                             'Only used for multi-graph datasets (QM9). '
                             'Tune to your VRAM: '
                             '8 GB → 2000 | 16 GB → 5000 | 24 GB+ → 30000.')

    return parser.parse_args()


def set_random_seeds(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ======================================================================== #
# Core counting function
# ======================================================================== #

def run_motif_counting(
    motif_counter:      'RelationalMotifCounter',
    preprocessor:       'DataPreprocessor',
    interactive:        bool,
    graph_index_start:  Optional[int] = None,
    graph_index_end:    Optional[int] = None,
    batch_size:         int = 1000,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Run motif counting over a (sub)set of graphs via the unified batched path.

    Both single-graph datasets (Cora) and multi-graph datasets (QM9) go
    through exactly the same code: count_batch(preprocessor).

    Parameters
    ----------
    motif_counter     : RelationalMotifCounter
    preprocessor      : DataPreprocessor  (already .preprocess()'d)
    interactive       : bool
    graph_index_start : int, optional  — inclusive start index
    graph_index_end   : int, optional  — inclusive end index
    batch_size        : int  — graphs per GPU mini-batch

    Returns
    -------
    aggregated_counts     : (num_motifs,) tensor — gradient intact
    selected_rules_values : dict or None
    """
    total = preprocessor.num_graphs

    # ------------------------------------------------------------------ #
    # Resolve the active index range
    # ------------------------------------------------------------------ #
    start = graph_index_start if graph_index_start is not None else 0
    end   = graph_index_end   if graph_index_end   is not None else total - 1

    start = max(0, min(start, total - 1))
    end   = max(0, min(end,   total - 1))

    if start > end:
        raise ValueError(
            f"graph_index_start ({start}) must be <= graph_index_end ({end})."
        )

    num_active = end - start + 1
    if num_active < total:
        print(f"\n  Graph subset: indices {start} to {end} "
              f"({num_active} of {total} graphs)")
        active_preprocessor = _slice_preprocessor(preprocessor, list(range(start, end + 1)))
    else:
        active_preprocessor = preprocessor

    # ------------------------------------------------------------------ #
    # Unified batched path — handles B=1 (Cora) and B=130k (QM9) equally
    # ------------------------------------------------------------------ #
    print(f"\n  Graphs to process : {active_preprocessor.num_graphs}")
    print(f"  Batch size        : {batch_size}")
    print(f"  Rules             : {motif_counter.num_motifs}")

    selected_rules_values = None
    if interactive:
        selected_rules_values = motif_counter.do_interactive_selection()

    all_motif_counts = motif_counter.count_batch(
        active_preprocessor,
        batch_size=batch_size,
        selected_rules_values=selected_rules_values,
    )

    aggregated = motif_counter.aggregate_motif_counts(all_motif_counts)
    return aggregated, selected_rules_values


def _slice_preprocessor(
    preprocessor: 'DataPreprocessor',
    indices: List[int],
) -> 'DataPreprocessor':
    """
    Return a DataPreprocessor shell whose tensors are rows `indices` of the
    full preprocessor. Pure index operation on CPU pinned tensors — no copy.
    """
    from data_preprocessor import DataPreprocessor
    idx = torch.tensor(indices, dtype=torch.long)

    sub = DataPreprocessor.__new__(DataPreprocessor)
    sub.device                  = preprocessor.device
    sub.N_max                   = preprocessor.N_max
    sub.num_graphs              = len(indices)
    sub.feature_onehot_mapping  = preprocessor.feature_onehot_mapping
    sub.total_onehot_dim        = preprocessor.total_onehot_dim
    sub.relation_keys           = preprocessor.relation_keys
    sub.has_edge_features       = preprocessor.has_edge_features
    sub.all_features            = preprocessor.all_features[idx]
    sub.all_feat_onehot         = preprocessor.all_feat_onehot[idx]
    sub.all_adj                 = {rel: preprocessor.all_adj[rel][idx]
                                   for rel in preprocessor.relation_keys}
    sub.all_edge                = ([e[idx] for e in preprocessor.all_edge]
                                   if preprocessor.all_edge is not None else None)
    return sub


# ======================================================================== #
# main
# ======================================================================== #

def main():
    """
    Pipeline overview
    -----------------
    Step 1  Load raw graph data (DataLoader)
    Step 2  Init motif store (reads DB once, caches to ./db/<n>.pkl)
    Step 3  Init motif counter (reads same .pkl)
    Step 4  Build graph_data_list with correct relation keys
    Step 4b DataPreprocessor: global N_max padding + one-hot feature encoding
    Step 5  run_motif_counting() — single-graph or batched GPU path

    Key changes vs previous version
    --------------------------------
    • DataPreprocessor (Step 4b) pads ALL graphs to the global N_max
      before any batching, so motif_counter never needs to pad per-batch.
    • DataPreprocessor also builds feat_onehot (N_max, D) for each graph
      and a shared feature_onehot_mapping = {col_idx: {val → oh_col_idx}}.
    • motif_counter uses feat_onehot[:, oh_col] directly instead of
      (feat[:, col] == val).float() — no boolean comparison in the
      gradient-tracked forward pass.
    """

    print("=" * 60)
    print("Motif Counting Pipeline")
    print("=" * 60)

    args = parse_arguments()
    set_random_seeds(seed=0)

    if args.database_name is None:
        args.database_name = args.dataset_type

    print(f"\nConfiguration:")
    print(f"  Dataset:     {args.dataset_type}")
    print(f"  Database:    {args.database_name}")
    print(f"  Graph type:  {args.graph_type}")
    print(f"  Device:      {args.device}")
    print(f"  Interactive: {args.interactive}")
    if args.dataset_type == 'qm9' and args.max_graphs:
        print(f"  Max graphs:  {args.max_graphs}")
    if args.graph_index_start is not None or args.graph_index_end is not None:
        print(f"  Graph range: {args.graph_index_start} -> {args.graph_index_end}")
    print(f"  Batch size:  {args.batch_size}  (multi-graph GPU batching)")

    timer = StepTimer()

    # ------------------------------------------------------------------ #
    # Step 1 — Load raw graph data
    # ------------------------------------------------------------------ #
    print("\n[Step 1/5] Loading and preprocessing graph data...")
    timer.start("Step 1 — Load graph data")
    data_loader = DataLoader(
        dataset_type=args.dataset_type,
        n_components=5,
        random_seed=0,
        max_graphs=args.max_graphs,
        device=args.device,
    )
    data_loader.load_data()
    data = data_loader.get_data()
    timer.stop("Step 1 — Load graph data")

    if args.dataset_type == 'cora':
        if data['num_nodes']:
            edges_count = len(data['edges']) if data['edges'] is not None else 'N/A'
            print(f"  Loaded {data['num_nodes']} nodes, {edges_count} edges")
        if data['num_features']:
            print(f"  Features: {data['num_features']} (including label)")
    elif args.dataset_type == 'qm9':
        print(f"  Loaded {data['num_graphs']} molecules")

    # ------------------------------------------------------------------ #
    # Step 2 — Init motif store
    # ------------------------------------------------------------------ #
    print(f"\n[Step 2/5] Initialising motif store...")
    timer.start("Step 2 — Init motif store")
    try:
        motif_store = RuleBasedMotifStore(
            database_name=args.database_name,
            args=args,
        )
        timer.stop("Step 2 — Init motif store")
    except Exception as e:
        timer.stop("Step 2 — Init motif store")
        print(f"\nError initialising motif store: {e}")
        timer.report()
        return

    # ------------------------------------------------------------------ #
    # Step 3 — Init motif counter
    # ------------------------------------------------------------------ #
    print("\n[Step 3/5] Initialising motif counter...")
    try:
        timer.start("Step 3 — Init motif counter")
        motif_counter = RelationalMotifCounter(
            database_name=args.database_name,
            args=args,
        )
        timer.stop("Step 3 — Init motif counter")

        # ---------------------------------------------------------------- #
        # Step 4 — Build graph_data_list with correct relation keys
        # ---------------------------------------------------------------- #
        print("\n[Step 4/5] Building graph data list...")
        timer.start("Step 4 — Build graph_data_list")
        graph_data_list = data_loader.get_graph_data_list(
            relation_keys=motif_counter.relation_keys
        )
        timer.stop("Step 4 — Build graph_data_list")

        # ---------------------------------------------------------------- #
        # Step 4b — Global padding + one-hot feature encoding
        #
        # DataPreprocessor runs a single pass over ALL graphs to:
        #   1. Find global N_max (max nodes across the ENTIRE dataset).
        #   2. Pad every graph's features, adjacency matrices, and edge
        #      feature tensors to N_max (in-place).
        #   3. Build a one-hot feature matrix  feat_onehot (N_max, D)
        #      for each graph, where D = total unique values across all
        #      feature columns.  The one-hot comparison is done HERE,
        #      outside any gradient-tracked computation.
        #   4. Store  feature_onehot_mapping = {col: {val → oh_col}}
        #      in each graph_data dict (same object for every graph).
        #
        # After this step, motif_counter._build_batch_tensors() only
        # needs to torch.stack() — no per-batch zero-padding.
        # ---------------------------------------------------------------- #
        print("\n[Step 4b/5] Running DataPreprocessor "
              "(global padding + one-hot features)...")
        timer.start("Step 4b — DataPreprocessor")
        preprocessor = DataPreprocessor(device=args.device)
        preprocessor.preprocess(graph_data_list)
        timer.stop("Step 4b — DataPreprocessor")

        print(f"\n{'='*60}")
        print("Performing motif counting...")
        print(f"{'='*60}")
        print(f"Total graphs:        {len(graph_data_list)}")
        print(f"Relation keys:       {motif_counter.relation_keys}")
        print(f"Global N_max:        {preprocessor.N_max}")
        print(f"One-hot feature dim: {preprocessor.total_onehot_dim}")

        # ---------------------------------------------------------------- #
        # Step 5 — Count (unified path for both Cora and QM9)
        # ---------------------------------------------------------------- #
        timer.start("Step 5 — Motif counting")
        aggregated_counts, selected_rules_values = run_motif_counting(
            motif_counter      = motif_counter,
            preprocessor       = preprocessor,
            interactive        = args.interactive,
            graph_index_start  = args.graph_index_start,
            graph_index_end    = args.graph_index_end,
            batch_size         = args.batch_size,
        )
        # GPU kernels are async — sync here so Step 5 timing reflects actual
        # GPU completion, not just kernel launch time.
        if args.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.stop("Step 5 — Motif counting")

        motif_counter.display_rules_and_motifs(aggregated_counts, selected_rules_values)

        print(f"\n{'='*60}")
        print("Pipeline completed successfully!")
        print(f"{'='*60}")
        print(f"Total unique rules:     {len(motif_counter.rules)}")
        print(f"Total motif values:     {aggregated_counts.numel()}")

    except Exception as e:
        # Stop any open timer so the report still shows what ran
        for label in ["Step 3 — Init motif counter",
                      "Step 4 — Build graph_data_list",
                      "Step 4b — DataPreprocessor",
                      "Step 5 — Motif counting"]:
            if label in timer._starts and label not in timer._durations:
                timer.stop(label)
        print(f"\nError during motif counting: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always print the timing report, even on error
        timer.report()


if __name__ == "__main__":
    main()
