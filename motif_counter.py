# motif_counter.py

import torch
import pickle
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional


class RelationalMotifCounter:
    """
    Counts motifs in a graph using relational algebra and Bayesian Network rules.
    Loads all required data from pickle file in ./db directory.

    STATELESS design
    ----------------
    self.matrices  → template dict loaded from the pickle (DB schema only).
                     NEVER written after __init__.

    Each call to count(graph_data) receives graph_data built by DataLoader
    and pre-processed by DataPreprocessor:
        graph_data['matrices']             → {relation_name: (N_max, N_max) tensor}
        graph_data['features']             → (N_max, F) node features  (pre-padded)
        graph_data['feat_onehot']          → (N_max, D) one-hot features (pre-padded)
        graph_data['feature_onehot_mapping'] → {col_idx: {val_int: oh_col_idx}}
        graph_data['labels']               → edge-feature tensors | None (pre-padded)
        graph_data['N_max']                → int — global N_max for the dataset

    GRADIENT-SAFE feature predicates
    ---------------------------------
    The old code computed feature masks with:
        (feat_b[:, :, indx] == val).float()          ← boolean, no gradient
    This is replaced throughout by direct indexing into the pre-built one-hot
    matrix:
        feat_onehot_b[:, :, mapping[indx][val]]      ← pure slice, gradient ✓
    The boolean comparison is performed ONCE during DataPreprocessor.preprocess()
    — outside any gradient-tracked computation.

    BATCHED design (count_batch)
    ----------------------------
    Since DataPreprocessor already pads ALL graphs to the global N_max, every
    graph_data tensor has identical shape.  _build_batch_tensors() now only
    needs to torch.stack() — no per-batch shape checks or zero-padding.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, database_name: str, args):
        self.database_name = database_name
        self.args = args

        db_dir = Path('./db')
        pickle_path = db_dir / f"{database_name}.pkl"

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Pickle file not found: {pickle_path}\n"
                f"Please ensure motif store has been initialised first."
            )

        print(f"  Loading motif data from: {pickle_path}")
        self._load_from_pickle(pickle_path)
        print(f"  Loaded {self.num_motifs} motif rules")

    def _load_from_pickle(self, pickle_path: Path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        self.entities              = data["entities"]
        self.relations             = data["relations"]
        self.keys                  = data["keys"]
        self.rules                 = data["rules"]
        self.indices               = data["indices"]
        self.attributes            = data["attributes"]
        self.base_indices          = data["base_indices"]
        self.mask_indices          = data["mask_indices"]
        self.sort_indices          = data["sort_indices"]
        self.stack_indices         = data["stack_indices"]
        self.functors              = data["functors"]
        self.variables             = data["variables"]
        self.nodes                 = data["nodes"]
        self.states                = data["states"]
        self.masks                 = data["masks"]
        self.multiples             = data["multiples"]
        self.entity_feature_columns   = data.get("entity_feature_columns", {})
        self.relation_feature_columns = data.get("relation_feature_columns", {})
        self.feature_info_mapping  = data.get("feature_info_mapping", {})
        self.num_nodes_graph       = data.get("num_nodes_graph", 0)

        # ── Select value set based on --rule_prune ────────────────────
        rule_prune = getattr(self.args, 'rule_prune', False)

        if "values_full" in data:
            # New-format pickle
            if rule_prune:
                self.values = data["values_pruned"]
                n_full   = sum(len(v) for v in data["values_full"])
                n_pruned = sum(len(v) for v in data["values_pruned"])
                print(f"  rule_prune=True: {n_pruned} / {n_full} value combinations kept")
            else:
                self.values = data["values_full"]
                print(f"  rule_prune=False: using all {sum(len(v) for v in data['values_full'])} value combinations")
        else:
            # Old-format pickle — use whatever was stored
            self.values = data["values"]
            print(f"  Warning: old-format pickle — delete db/{self.database_name}.pkl "
                  f"to regenerate with both value sets cached.")

        self.device = getattr(self.args, 'device', 'cuda')

        # Template matrices — kept ONLY to expose relation key names to DataLoader.
        # Never mutated after this point.
        self.matrices: Dict[str, torch.Tensor] = {}
        for key, matrix in data["matrices"].items():
            self.matrices[key] = (
                matrix.to(self.device) if isinstance(matrix, torch.Tensor) else matrix
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def num_motifs(self) -> int:
        return len(self.rules)

    @property
    def relation_keys(self) -> List[str]:
        """
        Relation names the DataLoader must use as keys inside graph_data['matrices'].
        Pass directly to data_loader.get_graph_data_list(relation_keys=...).

        Example:
            graph_data_list = data_loader.get_graph_data_list(
                relation_keys=motif_counter.relation_keys
            )
        """
        return list(self.matrices.keys())

    def do_interactive_selection(self) -> Dict:
        """Interactive rule/value selection for multi-graph runs (ask only once)."""
        print("\n" + "="*80)
        print("INTERACTIVE RULE SELECTION")
        print("="*80)
        print("(This selection will be applied to all graphs)")
        print("="*80 + "\n")
        selected = self._interactive_rule_selection()
        print("\n" + "="*80)
        print("Selection complete — will be applied to all graphs.")
        print("="*80)
        return selected

    # ------------------------------------------------------------------
    # Main entry point — batched (PARALLELISED OVER GRAPHS)
    # ------------------------------------------------------------------

    def count_batch(
        self,
        preprocessor: 'DataPreprocessor',
        batch_size: int = 1000,
        selected_rules_values: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Count motifs for all graphs via batched GPU tensor ops.

        Returns a (num_graphs, num_motifs) tensor. No .item() is called
        anywhere in this path — gradient flows intact through all bmm
        operations back to the adjacency tensors.

        For inference/display, call .detach() on the result.
        For training loss, use the result directly in F.mse_loss() etc.

        Parameters
        ----------
        preprocessor : DataPreprocessor
        batch_size   : graphs per GPU mini-batch
        selected_rules_values : dict, optional — subset of rules to count

        Returns
        -------
        torch.Tensor  shape (num_graphs, num_motifs)
        """
        batch_tensors = []
        total  = preprocessor.num_graphs
        N_max  = preprocessor.N_max
        fom    = preprocessor.feature_onehot_mapping

        for start in range(0, total, batch_size):
            end_excl = min(start + batch_size, total)
            B        = end_excl - start
            t0       = time.perf_counter()

            feat_b, feat_onehot_b, adj_b, edge_b = preprocessor.get_batch(start, end_excl)

            batch_result = self._iteration_function_batched(
                feat_b, feat_onehot_b, edge_b, adj_b, fom, B, N_max,
                selected_rules_values,
            )                                                            # (B, num_motifs)

            batch_tensors.append(batch_result)

            # Sync so elapsed time reflects actual GPU completion, not just launch.
            if self.device == 'cuda':
                torch.cuda.synchronize()

            elapsed        = time.perf_counter() - t0
            graphs_per_sec = B / elapsed if elapsed > 0 else float('inf')
            eta_sec        = (total - end_excl) / graphs_per_sec if graphs_per_sec > 0 else 0
            print(
                f"  Batch {start:>7}–{end_excl-1:<7}  [{B:>5} graphs]"
                f"  {elapsed:>6.2f}s"
                f"  ({graphs_per_sec:>8.1f} graphs/s)"
                f"  {end_excl}/{total} done"
                f"  ETA {self._fmt_time(eta_sec)}"
            )

        return torch.cat(batch_tensors, dim=0)                          # (num_graphs, num_motifs)

    # ------------------------------------------------------------------
    # Batched iteration loop  (unified — fully differentiable)
    # ------------------------------------------------------------------

    def _iteration_function_batched(
        self,
        feat_b:                torch.Tensor,                   # (B, N_max, F)
        feat_onehot_b:         torch.Tensor,                   # (B, N_max, D)
        edge_b:                Optional[List[torch.Tensor]],   # list[(B,C,N_max,N_max)] or None
        adj_b:                 Dict[str, torch.Tensor],        # {rel: (B, N_max, N_max)}
        feature_onehot_mapping: Dict[int, Dict[int, int]],
        B:                     int,
        N_max:                 int,
        selected_rules_values: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Unified differentiable batched motif counting.

        Returns (B, num_motifs) — no .item(), no detach(), no boolean
        comparisons. Gradient flows intact through all bmm operations
        back to adj_b tensors (and feat_onehot_b if it requires grad).

        count_batch()               — call .detach() on result for display/inference
        training loss               — use result directly in F.mse_loss() etc.
        """
        if selected_rules_values is not None:
            iteration_plan = [
                (rule_idx, value_idx, self.values[rule_idx][value_idx])
                for rule_idx, value_indices in selected_rules_values.items()
                for value_idx in value_indices
            ]
        else:
            iteration_plan = [
                (table, indexx, table_row)
                for table in range(len(self.rules))
                for indexx, table_row in enumerate(self.values[table])
            ]

        motif_tensors: List[torch.Tensor] = []   # each element: (B,)

        for table, indexx, table_row in iteration_plan:

            unmasked = self._compute_unmasked_matrices_batched(
                table, table_row,
                feat_b, feat_onehot_b, feature_onehot_mapping,
                edge_b, adj_b, B, N_max
            )
            masked  = self._compute_masked_matrices_batched(
                unmasked, self.base_indices[table], self.mask_indices[table]
            )
            sorted_ = self._compute_sorted_matrices_batched(
                masked, self.sort_indices[table]
            )
            stacked = self._compute_stacked_matrices_batched(
                sorted_, self.stack_indices[table], B
            )
            result  = self._compute_result_batched(stacked)             # (B,)

            motif_tensors.append(result)

            del unmasked, masked, sorted_, stacked

        if not motif_tensors:
            # No rules matched — return zero tensor of shape (B, 0)
            device = next(iter(adj_b.values())).device
            return torch.zeros(B, 0, dtype=torch.float32, device=device)

        return torch.stack(motif_tensors, dim=1)                        # (B, num_motifs)

    # ------------------------------------------------------------------
    # Batched state handlers
    # ------------------------------------------------------------------

    def _compute_unmasked_matrices_batched(
        self,
        table:                 int,
        table_row,
        feat_b:                torch.Tensor,                  # (B, N_max, F)
        feat_onehot_b:         torch.Tensor,                  # (B, N_max, D)
        feature_onehot_mapping: Dict[int, Dict[int, int]],
        edge_b:                Optional[List[torch.Tensor]],
        adj_b:                 Dict[str, torch.Tensor],
        B:                     int,
        N_max:                 int,
    ) -> List[torch.Tensor]:
        """Batched counterpart of _compute_unmasked_matrices (mode='test' path only)."""
        unmasked: List[torch.Tensor] = []

        for column in range(len(self.rules[table])):
            functor             = self.functors[table][column]
            table_functor_value = table_row[column + self.multiples[table]]
            state               = self.states[table][column]

            if state == 0:
                unmasked.append(
                    self._compute_state_zero_batched(
                        functor, table_functor_value,
                        feat_b, feat_onehot_b, feature_onehot_mapping,
                    )
                )
            elif state == 1:
                mats = self._compute_state_one_batched(
                    functor, table_functor_value,
                    self.variables[table][column],
                    self.masks[table][column],
                    feat_b, feat_onehot_b, feature_onehot_mapping,
                )
                unmasked.extend(mats)
            elif state == 2:
                unmasked.append(
                    self._compute_state_two_batched(functor, table_functor_value, adj_b)
                )
            elif state == 3:
                unmasked.append(
                    self._compute_state_three_batched(edge_b, functor, table_functor_value)
                )

        return unmasked

    def _compute_state_zero_batched(
        self,
        functor:               str,
        table_functor_value,
        feat_b:                torch.Tensor,                  # (B, N_max, F)
        feat_onehot_b:         torch.Tensor,                  # (B, N_max, D)
        feature_onehot_mapping: Dict[int, Dict[int, int]],
    ) -> torch.Tensor:
        """
        Unary feature predicate.
        Returns (B, N_max, 1)  — 1 where node matches the predicate, 0 elsewhere.

        GRADIENT-SAFE: uses a direct column slice of the pre-built one-hot
        matrix instead of the old boolean comparison `(fv == val).float()`.

        Padding rows in feat_onehot_b are all-zero by construction, so padded
        nodes naturally contribute 0 to all downstream products.
        """
        found, indx, _ = self._find_feature(functor)
        if found:
            val      = int(table_functor_value)
            col_map  = feature_onehot_mapping.get(indx, {})
            if val in col_map:
                oh_col = col_map[val]
                # Direct index into pre-built one-hot — no == comparison here
                return feat_onehot_b[:, :, oh_col].unsqueeze(2)      # (B, N_max, 1)
            else:
                # Value not seen during preprocessing — return zeros
                return torch.zeros(
                    feat_onehot_b.shape[0], feat_onehot_b.shape[1], 1,
                    dtype=torch.float32, device=self.device,
                )

        # Fallback: treat value as a raw feature column index (e.g. label column)
        col = int(table_functor_value)
        return feat_b[:, :, col].float().unsqueeze(2)                 # (B, N_max, 1)

    def _compute_state_one_batched(
        self,
        functor:               str,
        table_functor_value,
        variable:              str,
        masks_list:            List,
        feat_b:                torch.Tensor,                  # (B, N_max, F)
        feat_onehot_b:         torch.Tensor,                  # (B, N_max, D)
        feature_onehot_mapping: Dict[int, Dict[int, int]],
    ) -> List[torch.Tensor]:
        """
        Masked-variable predicate.
        Returns one (B, N_max, 1) or (B, 1, N_max) tensor per mask entry.

        GRADIENT-SAFE: same one-hot index strategy as _compute_state_zero_batched.
        """
        mats: List[torch.Tensor] = []
        found, indx, _ = self._find_feature(functor)

        for mask_info in masks_list:
            if found:
                val     = int(table_functor_value)
                col_map = feature_onehot_mapping.get(indx, {})
                if val in col_map:
                    oh_col   = col_map[val]
                    col_vals = feat_onehot_b[:, :, oh_col]            # (B, N_max)
                else:
                    col_vals = torch.zeros(
                        feat_onehot_b.shape[0], feat_onehot_b.shape[1],
                        dtype=torch.float32, device=self.device,
                    )

                if variable == mask_info[1]:
                    mats.append(col_vals.unsqueeze(2))                # (B, N_max, 1)
                else:
                    mats.append(col_vals.unsqueeze(1))                # (B, 1, N_max)
            else:
                # Fallback: raw feature column
                col = int(table_functor_value)
                fv  = feat_b[:, :, col].float()                       # (B, N_max)
                if variable == mask_info[1]:
                    mats.append(fv.unsqueeze(2))                      # (B, N_max, 1)
                else:
                    mats.append(fv.unsqueeze(1))                      # (B, 1, N_max)

        return mats

    def _compute_state_two_batched(
        self,
        functor: str,
        table_functor_value,
        adj_b: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Adjacency / relation matrix.
        Returns (B, N_max, N_max).
        """
        adj = adj_b[functor]                                          # (B, N_max, N_max)
        return (1 - adj) if table_functor_value == 'F' else adj

    def _compute_state_three_batched(
        self,
        edge_b: List[torch.Tensor],   # list of (B, C, N_max, N_max)
        functor: str,
        table_functor_value,
    ) -> torch.Tensor:
        """
        Edge feature predicate (QM9 bond types).
        Returns (B, N_max, N_max).
        """
        feature_idx = next(
            idx for idx, info in self.feature_info_mapping.items()
            if info['feature_name'] == functor
        )
        target = edge_b[feature_idx]                                  # (B, C, N_max, N_max)

        if table_functor_value == 'N/A':
            return torch.sum(target, dim=1)                           # (B, N_max, N_max)

        value_mapping   = self.feature_info_mapping[feature_idx]['value_index_mapping']
        reverse_mapping = {v: k for k, v in value_mapping.items()}
        val_idx         = reverse_mapping[int(table_functor_value)]
        return target[:, val_idx, :, :]                               # (B, N_max, N_max)

    # ------------------------------------------------------------------
    # Batched matrix algebra
    # ------------------------------------------------------------------

    def _compute_masked_matrices_batched(
        self,
        unmasked:     List[torch.Tensor],
        base_indices: List[int],
        mask_indices: List[List[int]],
    ) -> List[torch.Tensor]:
        """
        Element-wise masking — identical logic to the single-graph version.
        Tensors carry a leading batch dimension but broadcasting handles it.
        """
        masked = [unmasked[k] for k in base_indices]
        for k in mask_indices:
            masked[k[0]] = masked[k[0]] * unmasked[k[1]]
        return masked

    def _compute_sorted_matrices_batched(
        self,
        masked:       List[torch.Tensor],
        sort_indices: List,
    ) -> List[torch.Tensor]:
        """
        Transpose swaps dims 1 and 2 (batch dim 0 is untouched).
          (B, N, 1)  ↔  (B, 1, N)
          (B, N, N)  ↔  (B, N, N)^T
        """
        result = []
        for si in sort_indices:
            m = masked[si[1]]
            result.append(m.transpose(1, 2) if si[0] else m)
        return result

    def _compute_stacked_matrices_batched(
        self,
        sorted_:       List[torch.Tensor],
        stack_indices: List,
        B:             int,
    ) -> List[torch.Tensor]:
        """
        Batched matrix chain multiplication using torch.bmm.
        The diagonal masking step uses a (B, N, N) identity expanded over B.

        Key shapes (example for a 3-atom rule):
          (B, 1, N) @ (B, N, N) = (B, 1, N)
          (B, 1, N) @ (B, N, 1) = (B, 1, 1)   → squeezed to (B,) by _compute_result_batched
        """
        stacked     = sorted_.copy()
        pop_counter = 0

        for k in stack_indices:
            for _ in range(k[1] - k[0] - pop_counter):
                stacked[k[0]] = torch.bmm(stacked[k[0]], stacked[k[0] + 1])
                stacked.pop(k[0] + 1)
                pop_counter += 1

            # Diagonal masking — only for square matrices
            mat = stacked[k[0]]
            if mat.shape[1] == mat.shape[2]:
                N   = mat.shape[1]
                eye = (
                    torch.eye(N, dtype=torch.float32, device=self.device)
                    .unsqueeze(0)
                    .expand(B, -1, -1)
                )
                stacked[k[0]] = mat * eye

        return stacked

    def _compute_result_batched(
        self,
        stacked: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Final batched chain multiply → sum all spatial dims → (B,).

        The spatial shape after multiplication is NOT always (B, 1, 1):
          - Relational rules  (B, 1, N) @ (B, N, N) @ (B, N, 1) = (B, 1, 1)
          - Unary rules       result stays (B, N, 1) or (B, N, N)

        Mirroring the single-graph path which does torch.sum(result) over
        the whole matrix, we flatten every dim except the batch dim and sum.
        This is correct for all rule types.
        """
        result = stacked[0]
        for k in range(1, len(stacked)):
            result = torch.bmm(result, stacked[k])
        # Sum over all spatial dimensions, keep only the batch dimension
        return result.reshape(result.shape[0], -1).sum(dim=1)         # (B,)
    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Human-readable duration string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(seconds, 60)
        if m < 60:
            return f"{int(m)}m {s:.0f}s"
        h, m = divmod(m, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s"

    def _find_feature(self, functor: str) -> Tuple[bool, Optional[int], Optional[str]]:
        for key, feature_list in self.entity_feature_columns.items():
            if functor in feature_list:
                return True, feature_list.index(functor), key
        for key, feature_list in self.relation_feature_columns.items():
            if functor in feature_list:
                return True, feature_list.index(functor), key
        return False, None, None

    # ------------------------------------------------------------------
    # Aggregation & display
    # ------------------------------------------------------------------

    def get_rule_motif_mapping(self) -> List[Tuple[int, int]]:
        return [(i, len(self.values[i])) for i in range(len(self.rules))]

    def aggregate_motif_counts(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Sum motif counts across all graphs.

        Parameters
        ----------
        counts : (num_graphs, num_motifs) tensor — output of count_batch()

        Returns
        -------
        (num_motifs,) tensor — summed counts, gradient intact.
        Call .detach().tolist() for display.
        """
        return counts.sum(dim=0)                                        # (num_motifs,)

    def display_rules_and_motifs(
        self, aggregated_counts: torch.Tensor, selected_rules_values: Dict = None
    ):
        # Convert to plain list only at the display boundary
        counts_list = aggregated_counts.detach().cpu().tolist()
        print("\n" + "="*80)
        print("RULES AND MOTIF COUNTS")
        print("="*80)
        if selected_rules_values is not None:
            self._display_selective_results(counts_list, selected_rules_values)
        else:
            self._display_full_results(counts_list)

    def _display_full_results(self, aggregated_counts: List[float]):
        count_idx = 0
        for rule_idx in range(len(self.rules)):
            num_values = len(self.values[rule_idx])
            print(f"\nRule {rule_idx + 1}: {self.rules[rule_idx]}")
            print("-" * 80)
            for value_idx in range(num_values):
                print(f"  Value {value_idx + 1}/{num_values}: {aggregated_counts[count_idx]:.4f}")
                count_idx += 1

    def _display_selective_results(
        self, aggregated_counts: List[float], selected_rules_values: Dict
    ):
        count_idx = 0
        for rule_idx, value_indices in selected_rules_values.items():
            rule = self.rules[rule_idx]
            print(f"\nRule {rule_idx + 1}: {rule}")
            print("-" * 80)
            start_idx = self.multiples[rule_idx]
            for value_idx in value_indices:
                count     = aggregated_counts[count_idx]
                table_row = self.values[rule_idx][value_idx]
                functor_vals = [
                    f"{f}={table_row[start_idx + fi]}"
                    for fi, f in enumerate(rule)
                    if start_idx + fi < len(table_row)
                ]
                print(f"  [{value_idx}] {', '.join(functor_vals)} -> {count:.4f}")
                count_idx += 1

    # ------------------------------------------------------------------
    # Interactive selection helpers
    # ------------------------------------------------------------------

    def _interactive_rule_selection(self) -> Dict:
        print("\n" + "="*80)
        print("AVAILABLE RULES")
        print("="*80)

        for rule_idx in range(len(self.rules)):
            print(f"\n[{rule_idx}] Rule {rule_idx + 1}: {self.rules[rule_idx]}")
            print(f"    Number of value combinations: {len(self.values[rule_idx])}")

        print("\n" + "="*80)

        while True:
            rule_selection = input(
                "\nEnter rule indices to count (comma-separated, or 'all'): "
            ).strip()
            if rule_selection.lower() == 'all':
                selected_rule_indices = list(range(len(self.rules)))
                break
            try:
                selected_rule_indices = [int(x.strip()) for x in rule_selection.split(',')]
                if all(0 <= idx < len(self.rules) for idx in selected_rule_indices):
                    break
                print(f"Error: indices must be 0-{len(self.rules)-1}")
            except ValueError:
                print("Error: enter numbers separated by commas, or 'all'")

        selected_rules_values = {}
        for rule_idx in selected_rule_indices:
            print(f"\n{'='*80}")
            print(f"Selecting values for Rule {rule_idx + 1}: {self.rules[rule_idx]}")
            print("="*80)

            functor_value_options = self._get_functor_value_options(rule_idx)
            if not functor_value_options:
                print("No value combinations available. Skipping.")
                continue

            selected_functor_values = {}
            for functor_name, unique_values in functor_value_options.items():
                print(f"\n{functor_name}\n  Possible values: {unique_values}")
                while True:
                    val_sel = input("  Select values (comma-separated, or 'all'): ").strip()
                    if val_sel.lower() == 'all':
                        selected_functor_values[functor_name] = unique_values
                        break
                    selected_vals, invalid_vals = [], []
                    for v in val_sel.split(','):
                        matched = self._match_value_to_options(v.strip(), unique_values)
                        if matched is not None:
                            selected_vals.append(matched)
                        else:
                            invalid_vals.append(v.strip())
                    for iv in invalid_vals:
                        print(f"  Warning: '{iv}' is not a valid option")
                    if selected_vals:
                        selected_functor_values[functor_name] = selected_vals
                        break
                    print("  Error: no valid values selected. Try again.")

            while True:
                filtered = self._filter_combinations_by_functor_values(
                    rule_idx, selected_functor_values
                )
                if filtered:
                    print(f"\n  -> {len(filtered)} combinations match your selection")
                    break
                print(f"\n  -> 0 combinations match your selection — no rows in the database "
                      f"have this exact combination. Please try different values.")
                # Re-prompt all functors for this rule
                selected_functor_values = {}
                for functor_name, unique_values in functor_value_options.items():
                    print(f"\n{functor_name}\n  Possible values: {unique_values}")
                    while True:
                        val_sel = input("  Select values (comma-separated, or 'all'): ").strip()
                        if val_sel.lower() == 'all':
                            selected_functor_values[functor_name] = unique_values
                            break
                        selected_vals, invalid_vals = [], []
                        for v in val_sel.split(','):
                            matched = self._match_value_to_options(v.strip(), unique_values)
                            if matched is not None:
                                selected_vals.append(matched)
                            else:
                                invalid_vals.append(v.strip())
                        for iv in invalid_vals:
                            print(f"  Warning: '{iv}' is not a valid option")
                        if selected_vals:
                            selected_functor_values[functor_name] = selected_vals
                            break
                        print("  Error: no valid values selected. Try again.")

            selected_rules_values[rule_idx] = filtered

        return selected_rules_values

    def _match_value_to_options(self, user_input: str, options: List) -> Any:
        if user_input in options:
            return user_input
        try:
            user_float = float(user_input)
            user_int   = int(user_float) if user_float == int(user_float) else None
            if user_float in options:           return user_float
            if user_int is not None:
                if user_int in options:         return user_int
                if str(user_int) in options:    return str(user_int)
            if str(user_float) in options:      return str(user_float)
        except ValueError:
            pass
        return None

    def _get_functor_value_options(self, rule_idx: int) -> Dict[str, List]:
        rule = self.rules[rule_idx]
        functor_values: Dict[str, set] = {f: set() for f in rule}
        start_idx = self.multiples[rule_idx]
        for table_row in self.values[rule_idx]:
            for fi, functor in enumerate(rule):
                vi = start_idx + fi
                if vi < len(table_row):
                    functor_values[functor].add(table_row[vi])
        return {
            f: sorted(list(vs), key=lambda x: (isinstance(x, str), x))
            for f, vs in functor_values.items()
        }

    def _filter_combinations_by_functor_values(
        self, rule_idx: int, selected_functor_values: Dict[str, List]
    ) -> List[int]:
        rule = self.rules[rule_idx]
        matching = []
        start_idx = self.multiples[rule_idx]
        for row_idx, table_row in enumerate(self.values[rule_idx]):
            matches = True
            for fi, functor in enumerate(rule):
                vi = start_idx + fi
                if vi < len(table_row) and functor in selected_functor_values:
                    if table_row[vi] not in selected_functor_values[functor]:
                        matches = False
                        break
            if matches:
                matching.append(row_idx)
        return matching
