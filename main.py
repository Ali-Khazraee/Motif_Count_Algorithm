# main.py

import numpy as np
import torch
import random
import argparse
from typing import List, Optional, Tuple, Dict

from data import DataLoader
from motif_store import RuleBasedMotifStore
from motif_counter import RelationalMotifCounter


def parse_arguments():
    parser = argparse.ArgumentParser(description='Motif Count')

    parser.add_argument('--dataset_type', type=str, default='cora',
                        choices=['cora', 'qm9'])
    parser.add_argument('--max_graphs', type=int, default=None)
    parser.add_argument('--database_name', type=str, default='cora')
    parser.add_argument('--graph_type', type=str, default='homogeneous',
                        choices=['homogeneous', 'heterogeneous'])
    parser.add_argument('--motif_loss', type=bool, default=True)
    parser.add_argument('--rule_prune', type=bool, default=False)
    parser.add_argument('--rule_weight', type=bool, default=False)
    parser.add_argument('--test_local_mults', type=bool, default=True)
    parser.add_argument('--interactive', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    # Graph index selection: run counting only on a slice of graph_data_list
    parser.add_argument('--graph_index_start', type=int, default=0,
                        help='First graph index to count (inclusive). '
                             'Only valid when dataset has more than one graph.')
    parser.add_argument('--graph_index_end', type=int, default=10,
                        help='Last graph index to count (inclusive). '
                             'Only valid when dataset has more than one graph.')

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
    motif_counter: 'RelationalMotifCounter',
    graph_data_list: List[Dict],
    interactive: bool,
    graph_index_start: Optional[int] = None,
    graph_index_end:   Optional[int] = None,
) -> Tuple[List[float], Optional[Dict]]:
    """
    Run motif counting over a (sub)set of graphs and return aggregated counts.

    Parameters
    ----------
    motif_counter : RelationalMotifCounter
        Initialised counter loaded from pickle.

    graph_data_list : list of dict
        Full list of graph_data dicts built by DataLoader.get_graph_data_list().

    interactive : bool
        If True, ask the user which rules/values to count.
        For multi-graph runs the selection is asked only once.

    graph_index_start : int, optional
        Inclusive start index into graph_data_list.
        Ignored (with a warning) when graph_data_list has only one graph.
        Defaults to 0 when None.

    graph_index_end : int, optional
        Inclusive end index into graph_data_list.
        Ignored (with a warning) when graph_data_list has only one graph.
        Defaults to len(graph_data_list) - 1 when None.

    Returns
    -------
    aggregated_counts : List[float]
        Summed motif counts across all processed graphs.

    selected_rules_values : dict or None
        The rule/value selection used (None in non-interactive mode).
    """
    total = len(graph_data_list)

    # ------------------------------------------------------------------ #
    # Resolve the graph subset
    # ------------------------------------------------------------------ #
    if total == 1:
        # Single-graph dataset — index selection makes no sense
        if graph_index_start is not None or graph_index_end is not None:
            print(
                "\n  Note: graph_index_start / graph_index_end are ignored "
                "because this dataset contains only one graph."
            )
        active_indices = [0]

    else:
        # Multi-graph dataset — apply the requested slice
        start = graph_index_start if graph_index_start is not None else 0
        end   = graph_index_end   if graph_index_end   is not None else total - 1

        # Clamp to valid range
        start = max(0, min(start, total - 1))
        end   = max(0, min(end,   total - 1))

        if start > end:
            raise ValueError(
                f"graph_index_start ({start}) must be <= graph_index_end ({end})."
            )

        active_indices = list(range(start, end + 1))
        print(f"\n  Graph subset: indices {start} to {end} "
              f"({len(active_indices)} of {total} graphs)")

    # ------------------------------------------------------------------ #
    # Rule / value selection (asked only once for multi-graph runs)
    # ------------------------------------------------------------------ #
    selected_rules_values = None

    if interactive and len(active_indices) > 1:
        print(f"\n{len(active_indices)} graphs to process — "
              f"selecting rules ONCE then applying to all graphs.\n")
        selected_rules_values = motif_counter.do_interactive_selection()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    all_motif_counts = []

    for loop_pos, graph_idx in enumerate(active_indices):
        graph_data = graph_data_list[graph_idx]
        print(f"\n--- Graph {graph_idx} "
              f"[{loop_pos + 1}/{len(active_indices)}] ---")

        if interactive:
            if selected_rules_values is not None:
                # Multi-graph path: reuse pre-selected rules/values
                motif_counts = motif_counter.count(
                    graph_data,
                    selected_rules_values=selected_rules_values,
                )
            else:
                # Single-graph path: interactive selection inside count()
                motif_counts, selected_rules_values = motif_counter.count(
                    graph_data, interactive=True,
                )
        else:
            motif_counts = motif_counter.count(graph_data, interactive=False)

        all_motif_counts.append(motif_counts)
        print(f"  -> {len(motif_counts)} motif values counted")

    # ------------------------------------------------------------------ #
    # Aggregate
    # ------------------------------------------------------------------ #
    aggregated_counts = motif_counter.aggregate_motif_counts(all_motif_counts)
    return aggregated_counts, selected_rules_values


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
    Step 5  run_motif_counting() — optionally restricted to a graph index range
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

    # ------------------------------------------------------------------ #
    # Step 1 — Load raw graph data
    # ------------------------------------------------------------------ #
    print("\n[Step 1/3] Loading and preprocessing graph data...")
    data_loader = DataLoader(
        dataset_type=args.dataset_type,
        n_components=5,
        random_seed=0,
        max_graphs=args.max_graphs,
        device=args.device,
    )
    data_loader.load_data()
    data = data_loader.get_data()

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
    print(f"\n[Step 2/3] Initialising motif store...")
    try:
        motif_store = RuleBasedMotifStore(
            database_name=args.database_name,
            args=args,
        )
    except Exception as e:
        print(f"\nError initialising motif store: {e}")
        return

    # ------------------------------------------------------------------ #
    # Step 3 — Init motif counter
    # ------------------------------------------------------------------ #
    print("\n[Step 3/3] Counting motifs...")
    try:
        motif_counter = RelationalMotifCounter(
            database_name=args.database_name,
            args=args,
        )

        # ---------------------------------------------------------------- #
        # Step 4 — Build graph_data_list with correct relation keys
        # ---------------------------------------------------------------- #
        graph_data_list = data_loader.get_graph_data_list(
            relation_keys=motif_counter.relation_keys
        )

        print(f"\n{'='*60}")
        print("Performing motif counting...")
        print(f"{'='*60}")
        print(f"Total graphs available:  {len(graph_data_list)}")
        print(f"Relation keys:           {motif_counter.relation_keys}")

        # ---------------------------------------------------------------- #
        # Step 5 — Count (optionally restricted to a graph index range)
        # ---------------------------------------------------------------- #
        aggregated_counts, selected_rules_values = run_motif_counting(
            motif_counter      = motif_counter,
            graph_data_list    = graph_data_list,
            interactive        = args.interactive,
            graph_index_start  = args.graph_index_start,
            graph_index_end    = args.graph_index_end,
        )

        motif_counter.display_rules_and_motifs(aggregated_counts, selected_rules_values)

        print(f"\n{'='*60}")
        print("Pipeline completed successfully!")
        print(f"{'='*60}")
        print(f"Total graphs processed: "
              f"{len(graph_data_list) if args.graph_index_start is None and args.graph_index_end is None else 'see range above'}")
        print(f"Total unique rules:     {len(motif_counter.rules)}")
        print(f"Total motif values:     {len(aggregated_counts)}")

    except Exception as e:
        print(f"\nError during motif counting: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
