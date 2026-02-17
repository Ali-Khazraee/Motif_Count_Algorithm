# motif_counter.py

import torch
import pickle
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

    Each call to count(graph_data) receives graph_data built by DataLoader:
        graph_data['matrices']  → {relation_name: (N,N) tensor}   <- built by DataLoader
        graph_data['features']  → (N, F) node features
        graph_data['labels']    → edge-feature tensors | None

    _build_local_inputs() reads graph_data['matrices'] DIRECTLY as local_matrices.
    There is no copying or substitution from self.matrices.
    The pattern is identical to reconstructed_x_slice / reconstructed_labels.
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
        self.values                = data["values"]
        self.prunes                = data["prunes"]
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
    # Main entry point
    # ------------------------------------------------------------------

    def count(
        self,
        graph_data: Dict,
        interactive: bool = False,
        selected_rules_values: Optional[Dict] = None,
    ):
        """
        Count motifs for a single graph.

        Parameters
        ----------
        graph_data : dict  (built by DataLoader.get_graph_data_list)
            'matrices'  – {relation_name: (N,N) tensor}  used directly as local_matrices
            'features'  – (N, F) node features
            'labels'    – edge-feature tensors | None

        Returns
        -------
        List[float]                  when not interactive
        Tuple[List[float], Dict]     when interactive=True and no pre-selection
        """
        reconstructed_x_slice, reconstructed_labels, local_matrices = (
            self._build_local_inputs(graph_data)
        )

        if selected_rules_values is not None:
            return self._iteration_function_selective(
                reconstructed_x_slice, reconstructed_labels, local_matrices,
                mode="test", selected_rules_values=selected_rules_values,
            )

        if interactive:
            selected_rules_values = self._interactive_rule_selection()
            motif_counts = self._iteration_function_selective(
                reconstructed_x_slice, reconstructed_labels, local_matrices,
                mode="test", selected_rules_values=selected_rules_values,
            )
            return motif_counts, selected_rules_values

        return self._iteration_function(
            reconstructed_x_slice, reconstructed_labels, local_matrices, mode="test"
        )

    # ------------------------------------------------------------------
    # Build per-graph inputs  ← THE KEY METHOD
    # ------------------------------------------------------------------

    def _build_local_inputs(
        self, graph_data: Dict
    ) -> Tuple[torch.Tensor, Any, Dict[str, torch.Tensor]]:
        """
        Extract the three independent per-graph inputs from graph_data.

        local_matrices
            Read directly from graph_data['matrices'].
            The DataLoader built this dict with the correct relation keys
            and the correct adjacency tensor for THIS graph.
            self.matrices is NEVER read or copied here.

        reconstructed_x_slice
            graph_data['features']  — node feature tensor / array.

        reconstructed_labels
            graph_data['labels']    — edge features (QM9) or None (Cora).
        """
        # ---- adjacency dict: read directly from graph_data, no copy needed ----
        local_matrices: Dict[str, torch.Tensor] = {
            key: (mat.to(self.device) if isinstance(mat, torch.Tensor) else mat)
            for key, mat in graph_data['matrices'].items()
        }

        # ---- node features ----
        features = graph_data['features']
        reconstructed_x_slice = (
            features.to(self.device) if torch.is_tensor(features)
            else torch.tensor(features).to(self.device)
        )

        # ---- edge/label features ----
        reconstructed_labels = graph_data.get('labels', None)

        return reconstructed_x_slice, reconstructed_labels, local_matrices

    # ------------------------------------------------------------------
    # Iteration loops
    # ------------------------------------------------------------------

    def _iteration_function(
        self,
        reconstructed_x_slice,
        reconstructed_labels,
        local_matrices: Dict,
        mode: str,
    ) -> List:
        motif_list: List = []
        functor_value_dict: Dict = {}
        counter = counter_c1 = 0

        for table in range(len(self.rules)):
            for indexx, table_row in enumerate(self.values[table]):
                count, functor_value_dict, counter, counter_c1 = (
                    self._count_single_rule_value(
                        table, indexx, table_row,
                        reconstructed_x_slice, reconstructed_labels,
                        local_matrices, mode,
                        functor_value_dict, counter, counter_c1,
                    )
                )
                motif_list.append(count)

        return motif_list

    def _iteration_function_selective(
        self,
        reconstructed_x_slice,
        reconstructed_labels,
        local_matrices: Dict,
        mode: str,
        selected_rules_values: Dict,
    ) -> List:
        motif_list: List = []
        functor_value_dict: Dict = {}
        counter = counter_c1 = 0

        for rule_idx, value_indices in selected_rules_values.items():
            print(f"\nCounting Rule {rule_idx + 1}: {self.rules[rule_idx]}")
            for value_idx in value_indices:
                table_row = self.values[rule_idx][value_idx]
                count, functor_value_dict, counter, counter_c1 = (
                    self._count_single_rule_value(
                        rule_idx, value_idx, table_row,
                        reconstructed_x_slice, reconstructed_labels,
                        local_matrices, mode,
                        functor_value_dict, counter, counter_c1,
                    )
                )
                motif_list.append(count)
                print(f"  Value combination {value_idx}: {count}")

        return motif_list

    # ------------------------------------------------------------------
    # Core counting pipeline
    # ------------------------------------------------------------------

    def _count_single_rule_value(
        self, table, indexx, table_row,
        reconstructed_x_slice, reconstructed_labels,
        local_matrices: Dict,
        mode: str,
        functor_value_dict: dict,
        counter: int, counter_c1: int,
    ):
        unmasked, functor_value_dict, counter, counter_c1 = (
            self._compute_unmasked_matrices(
                table, table_row,
                reconstructed_x_slice, reconstructed_labels,
                local_matrices, mode,
                functor_value_dict, counter, counter_c1,
            )
        )
        masked   = self._compute_masked_matrices(unmasked, self.base_indices[table], self.mask_indices[table])
        sorted_  = self._compute_sorted_matrices(masked, self.sort_indices[table])
        stacked  = self._compute_stacked_matrices(sorted_, self.stack_indices[table])
        result   = self._compute_result(stacked)

        count = (torch.sum(result) * self.prunes[table][indexx]
                 if self.args.rule_weight else torch.sum(result))

        del unmasked, masked, sorted_, stacked, result
        return count, functor_value_dict, counter, counter_c1

    # ------------------------------------------------------------------
    # Unmasked matrix computation
    # ------------------------------------------------------------------

    def _compute_unmasked_matrices(
        self, table, table_row,
        reconstructed_x_slice, reconstructed_labels,
        local_matrices: Dict,
        mode: str,
        functor_value_dict: dict,
        counter: int, counter_c1: int,
    ):
        unmasked_matrices = []

        for column in range(len(self.rules[table])):
            functor             = self.functors[table][column]
            table_functor_value = table_row[column + self.multiples[table]]
            fvd_key             = (table_functor_value, functor, '0', ('0', '0', '0'))

            if mode == 'metric_observed' and self.states[table][column] != 1:
                cached = functor_value_dict.get(fvd_key)
                if cached is not None:
                    unmasked_matrices.append(cached)
                    counter += 1
                    continue

            state = self.states[table][column]

            if state == 0:
                matrix = self._compute_state_zero(
                    functor, table_functor_value, self.nodes[table][column],
                    reconstructed_x_slice, reconstructed_labels, mode,
                )
                unmasked_matrices.append(matrix)
                if mode == 'metric_observed':
                    functor_value_dict[fvd_key] = matrix

            elif state == 1:
                matrices_list, functor_value_dict, counter, counter_c1 = (
                    self._compute_state_one(
                        functor, table_functor_value,
                        self.variables[table][column],
                        self.nodes[table][column],
                        self.masks[table][column],
                        reconstructed_x_slice, reconstructed_labels,
                        local_matrices, mode,
                        functor_value_dict, counter, counter_c1,
                    )
                )
                unmasked_matrices.extend(matrices_list)

            elif state == 2:
                matrix = self._compute_state_two(functor, table_functor_value, local_matrices)
                unmasked_matrices.append(matrix)
                if mode == 'metric_observed':
                    functor_value_dict[fvd_key] = matrix

            elif state == 3:
                matrix = self._compute_state_three(
                    reconstructed_labels, functor, table_functor_value
                )
                unmasked_matrices.append(matrix)
                if mode == 'metric_observed':
                    functor_value_dict[fvd_key] = matrix

        return unmasked_matrices, functor_value_dict, counter, counter_c1

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _compute_state_zero(
        self, functor, table_functor_value, functor_address,
        reconstructed_x_slice, reconstructed_labels, mode,
    ):
        if mode == 'metric_observed':
            primary_key = self.keys[functor_address]
            matrix = torch.zeros(
                (len(self.entities[functor_address].index), 1), device=self.device
            )
            for entity_index in range(len(self.entities[functor_address][functor])):
                functor_value = self.entities[functor_address][functor][entity_index]
                if isinstance(table_functor_value, str):
                    if isinstance(functor_value, (np.int64, np.int32)):
                        functor_value = str(functor_value)
                    elif isinstance(functor_value, (np.float64, np.float32)):
                        functor_value = str(int(functor_value))
                if functor_value == table_functor_value:
                    key_index = self.entities[functor_address][primary_key][entity_index]
                    row_index = self.indices[primary_key][key_index]
                    matrix[row_index][0] = 1
        else:
            found, indx, entity_or_relation_key = self._find_feature(functor)
            if found:
                table_functor_value = int(table_functor_value)
                if self.args.test_local_mults:
                    fv = (reconstructed_x_slice[entity_or_relation_key][:, indx]
                          if self.args.graph_type == 'heterogeneous'
                          else reconstructed_x_slice[:, indx])
                    matrix = (fv == table_functor_value).float().view(-1, 1)
            else:
                matrix = (
                    reconstructed_labels[:, int(table_functor_value)]
                    .float().view(-1, 1).to(self.device)
                )
        return matrix

    def _compute_state_one(
        self,
        functor, table_functor_value, variable, functor_address, masks_list,
        reconstructed_x_slice, reconstructed_labels,
        local_matrices: Dict,
        mode: str,
        functor_value_dict: dict,
        counter: int, counter_c1: int,
    ):
        matrices_list = []
        primary_key = self.keys[functor_address]

        for mask_info in masks_list:
            fvd_key = (table_functor_value, functor, variable, tuple(mask_info))

            if mode == 'metric_observed':
                cached = functor_value_dict.get(fvd_key)
                if cached is not None:
                    matrices_list.append(cached)
                    counter += 1
                    counter_c1 += 1
                    continue

                # Shape from the local matrix for THIS graph
                if variable == mask_info[1]:
                    matrix = torch.zeros(
                        (local_matrices[mask_info[0]].shape[0], 1), device=self.device
                    )
                else:
                    matrix = torch.zeros(
                        (1, local_matrices[mask_info[0]].shape[1]), device=self.device
                    )

                for entity_index in range(len(self.entities[functor_address][functor])):
                    functor_value = self.entities[functor_address][functor][entity_index]
                    if functor_value == table_functor_value:
                        key_index = self.entities[functor_address][primary_key][entity_index]
                        index = self.indices[primary_key][key_index]
                        if variable == mask_info[1]:
                            matrix[index, 0] = 1
                        else:
                            matrix[0, index] = 1

                matrices_list.append(matrix)
                functor_value_dict[fvd_key] = matrix

            else:
                if variable == mask_info[1]:
                    matrix = self._compute_state_one_variable(
                        functor, table_functor_value, functor_address,
                        reconstructed_x_slice, reconstructed_labels,
                    )
                else:
                    matrix = self._compute_state_one_variable_transpose(
                        functor, table_functor_value, functor_address,
                        reconstructed_x_slice, reconstructed_labels,
                    )
                matrices_list.append(matrix)

        return matrices_list, functor_value_dict, counter, counter_c1

    def _compute_state_one_variable(
        self, functor, table_functor_value, functor_address,
        reconstructed_x_slice, reconstructed_labels,
    ):
        found, indx, entity_or_relation_key = self._find_feature(functor)
        if found:
            table_functor_value = int(table_functor_value)
            if self.args.test_local_mults:
                fv = (reconstructed_x_slice[entity_or_relation_key][:, indx]
                      if self.args.graph_type == 'heterogeneous'
                      else reconstructed_x_slice[:, indx])
                matrix = (fv == table_functor_value).float().view(-1, 1)
        else:
            matrix = reconstructed_labels[:, int(table_functor_value)].float().view(-1, 1)
        return matrix

    def _compute_state_one_variable_transpose(
        self, functor, table_functor_value, functor_address,
        reconstructed_x_slice, reconstructed_labels,
    ):
        found, indx, entity_or_relation_key = self._find_feature(functor)
        if found:
            table_functor_value = int(table_functor_value)
            if self.args.test_local_mults:
                fv = (reconstructed_x_slice[entity_or_relation_key][:, indx]
                      if self.args.graph_type == 'heterogeneous'
                      else reconstructed_x_slice[:, indx])
                matrix = (fv == table_functor_value).float().view(1, -1)
        else:
            matrix = (
                reconstructed_labels[:, int(table_functor_value)]
                .view(1, -1).to(self.device)
            )
        return matrix

    def _compute_state_two(self, functor, table_functor_value, local_matrices: Dict):
        """Read adjacency from local_matrices (which came from graph_data['matrices'])."""
        if table_functor_value == 'F':
            return 1 - local_matrices[functor]
        return local_matrices[functor]

    def _compute_state_three(self, reconstructed_labels, functor, table_functor_value):
        feature_idx = None
        for idx, info in self.feature_info_mapping.items():
            if info['feature_name'] == functor:
                feature_idx = idx
                break
        target_tensor = reconstructed_labels[feature_idx]
        if table_functor_value == 'N/A':
            return torch.sum(target_tensor, dim=0)
        value_mapping   = self.feature_info_mapping[feature_idx]['value_index_mapping']
        reverse_mapping = {v: k for k, v in value_mapping.items()}
        return target_tensor[reverse_mapping[int(table_functor_value)]]

    # ------------------------------------------------------------------
    # Matrix algebra helpers
    # ------------------------------------------------------------------

    def _compute_masked_matrices(self, unmasked, base_indices, mask_indices):
        masked = [unmasked[k] for k in base_indices]
        for k in mask_indices:
            masked[k[0]] = torch.mul(masked[k[0]], unmasked[k[1]])
        return masked

    def _compute_sorted_matrices(self, masked, sort_indices):
        return [masked[k[1]].T if k[0] else masked[k[1]] for k in sort_indices]

    def _compute_stacked_matrices(self, sorted_, stack_indices):
        stacked = sorted_.copy()
        pop_counter = 0
        for k in stack_indices:
            for _ in range(k[1] - k[0] - pop_counter):
                stacked[k[0]] = torch.mm(stacked[k[0]], stacked[k[0] + 1])
                stacked.pop(k[0] + 1)
                pop_counter += 1
            stacked[k[0]] = torch.mul(
                stacked[k[0]],
                torch.eye(len(stacked[k[0]]), device=self.device),
            )
        return stacked

    def _compute_result(self, stacked):
        result = stacked[0]
        for k in range(1, len(stacked)):
            result = torch.mm(result, stacked[k])
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

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

    def aggregate_motif_counts(self, all_motif_counts: List[List]) -> List[float]:
        if not all_motif_counts:
            return []
        aggregated = [0.0] * len(all_motif_counts[0])
        for graph_counts in all_motif_counts:
            for idx, count in enumerate(graph_counts):
                aggregated[idx] += count.item() if isinstance(count, torch.Tensor) else count
        return aggregated

    def display_rules_and_motifs(
        self, aggregated_counts: List[float], selected_rules_values: Dict = None
    ):
        print("\n" + "="*80)
        print("RULES AND MOTIF COUNTS")
        print("="*80)
        if selected_rules_values is not None:
            self._display_selective_results(aggregated_counts, selected_rules_values)
        else:
            self._display_full_results(aggregated_counts)

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

            filtered = self._filter_combinations_by_functor_values(
                rule_idx, selected_functor_values
            )
            print(f"\n  -> {len(filtered)} combinations match your selection")
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
