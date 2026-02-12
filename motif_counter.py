# motif_counter.py

import torch
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any


class RelationalMotifCounter:
    """
    Counts motifs in a graph using relational algebra and Bayesian Network rules.
    Loads all required data from pickle file in ./db directory.
    """
    
    def __init__(self, database_name: str, args):
        """
        Initialize the motif counter by loading from pickle.
        
        Args:
            database_name: Name of the database (e.g., 'cora', 'citeseer')
            args: Arguments object containing configuration
        """
        self.database_name = database_name
        self.args = args
        
        # Determine pickle path
        db_dir = Path('./db')
        pickle_path = db_dir / f"{database_name}.pkl"
        
        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Pickle file not found: {pickle_path}\n"
                f"Please ensure motif store has been initialized first."
            )
        
        # Load all data from pickle
        print(f"  📦 Loading motif data from: {pickle_path}")
        self._load_from_pickle(pickle_path)
        print(f"  ✓ Loaded {self.num_motifs} motif rules")
    
    def _load_from_pickle(self, pickle_path: Path):
        """Load all required data from pickle file."""
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        
        # Load all required attributes
        self.entities = data["entities"]
        self.relations = data["relations"]
        self.keys = data["keys"]
        self.rules = data["rules"]
        self.indices = data["indices"]
        self.attributes = data["attributes"]
        self.base_indices = data["base_indices"]
        self.mask_indices = data["mask_indices"]
        self.sort_indices = data["sort_indices"]
        self.stack_indices = data["stack_indices"]
        self.values = data["values"]
        self.prunes = data["prunes"]
        self.functors = data["functors"]
        self.variables = data["variables"]
        self.nodes = data["nodes"]
        self.states = data["states"]
        self.masks = data["masks"]
        self.multiples = data["multiples"]
        self.entity_feature_columns = data.get("entity_feature_columns", {})
        self.relation_feature_columns = data.get("relation_feature_columns", {})
        self.feature_info_mapping = data.get("feature_info_mapping", {})
        self.num_nodes_graph = data.get("num_nodes_graph", 0)
        
        # Load matrices and move to device
        self.device = getattr(self.args, 'device', 'cuda')
        self.matrices = {}
        for key, matrix in data["matrices"].items():
            if isinstance(matrix, torch.Tensor):
                self.matrices[key] = matrix.to(self.device)
            else:
                self.matrices[key] = matrix
    
    @property
    def num_motifs(self) -> int:
        """Total number of motif rules."""
        return len(self.rules)
    
    def count(self, graph_data: Dict, interactive: bool = False):
        """
        Main entry point: counts all motifs in a graph.
        
        Args:
            graph_data: Dictionary containing:
                - 'adjacency_matrix': Adjacency tensor
                - 'features': Node features
                - 'labels': Node labels (optional)
            interactive: If True, ask user which rules/values to count
                
        Returns:
            If interactive=False: List[float] - motif counts
            If interactive=True: Tuple[List[float], Dict] - (motif counts, selected_rules_values)
        """
        # Process and update graph data
        reconstructed_x_slice, reconstructed_labels = self._process_graph_data(graph_data)
        
        if interactive:
            # Interactive mode: user selects rules and values
            selected_rules_values = self._interactive_rule_selection()
            motif_counts = self._iteration_function_selective(
                reconstructed_x_slice, 
                reconstructed_labels, 
                mode="test",
                selected_rules_values=selected_rules_values
            )
            return motif_counts, selected_rules_values
        else:
            # Default mode: count all rules and values
            motif_counts = self._iteration_function(
                reconstructed_x_slice, 
                reconstructed_labels, 
                mode="test"
            )
            return motif_counts
    
    def _process_graph_data(self, graph_data: Dict) -> Tuple:
        """Process graph data and update internal matrices."""
        adjacency = graph_data['adjacency_matrix']
        features = graph_data['features']
        labels = graph_data.get('labels', None)
        
        # Update the adjacency matrix
        key = list(self.matrices.keys())[0]
        self.matrices[key] = adjacency.to(self.device)
        
        # Process features
        reconstructed_x_slice = torch.tensor(features).to(self.device)
        reconstructed_labels = None
        
        return reconstructed_x_slice, reconstructed_labels
    
    def _interactive_rule_selection(self) -> Dict:
        """
        Interactive mode: display rules and let user select which to count.
        
        Returns:
            Dictionary mapping rule_index -> list of value indices to count
        """
        print("\n" + "="*80)
        print("AVAILABLE RULES")
        print("="*80)
        
        # Display all rules
        for rule_idx in range(len(self.rules)):
            rule = self.rules[rule_idx]
            num_values = len(self.values[rule_idx])
            print(f"\n[{rule_idx}] Rule {rule_idx + 1}: {rule}")
            print(f"    Number of value combinations: {num_values}")
        
        print("\n" + "="*80)
        
        # Ask user which rules to count
        while True:
            rule_selection = input("\nEnter rule indices to count (comma-separated, or 'all' for all rules): ").strip()
            
            if rule_selection.lower() == 'all':
                selected_rule_indices = list(range(len(self.rules)))
                break
            else:
                try:
                    selected_rule_indices = [int(x.strip()) for x in rule_selection.split(',')]
                    # Validate indices
                    if all(0 <= idx < len(self.rules) for idx in selected_rule_indices):
                        break
                    else:
                        print(f"Error: Rule indices must be between 0 and {len(self.rules)-1}")
                except ValueError:
                    print("Error: Please enter valid numbers separated by commas, or 'all'")
        
        # For each selected rule, ask about functor values
        selected_rules_values = {}
        
        for rule_idx in selected_rule_indices:
            print(f"\n" + "="*80)
            print(f"Selecting values for Rule {rule_idx + 1}: {self.rules[rule_idx]}")
            print("="*80)
            
            # Get unique values for each functor in this rule
            functor_value_options = self._get_functor_value_options(rule_idx)
            
            if not functor_value_options:
                print(f"No value combinations available for this rule. Skipping.")
                continue
            
            # Display unique values for each functor
            selected_functor_values = {}
            for functor_name, unique_values in functor_value_options.items():
                print(f"\n{functor_name}")
                print(f"  Possible values: {unique_values}")
                
                while True:
                    value_selection = input(f"  Select values (comma-separated, or 'all'): ").strip()
                    
                    if value_selection.lower() == 'all':
                        selected_functor_values[functor_name] = unique_values
                        break
                    else:
                        selected_vals = []
                        invalid_vals = []
                        
                        for val in value_selection.split(','):
                            val = val.strip()
                            matched_value = self._match_value_to_options(val, unique_values)
                            
                            if matched_value is not None:
                                selected_vals.append(matched_value)
                            else:
                                invalid_vals.append(val)
                        
                        # Show warnings for invalid values
                        for invalid_val in invalid_vals:
                            print(f"  Warning: '{invalid_val}' is not a valid option")
                        
                        if selected_vals:
                            selected_functor_values[functor_name] = selected_vals
                            break
                        else:
                            print("  Error: No valid values selected. Try again.")
            
            # Filter combinations based on selected functor values
            filtered_indices = self._filter_combinations_by_functor_values(
                rule_idx, selected_functor_values
            )
            
            print(f"\n  → {len(filtered_indices)} combinations match your selection")
            selected_rules_values[rule_idx] = filtered_indices
        
        return selected_rules_values
    
    def _match_value_to_options(self, user_input: str, options: List) -> any:
        """
        Match user input string to actual value in options list.
        Handles different type combinations (string vs numeric).
        
        Args:
            user_input: User's input as string
            options: List of valid options (can be strings, floats, ints, etc.)
            
        Returns:
            Matched value from options, or None if no match
        """
        # Strategy 1: Exact string match (e.g., user types 'T', option is 'T')
        if user_input in options:
            return user_input
        
        # Strategy 2: Try numeric conversions
        try:
            # Parse as number
            user_float = float(user_input)
            user_int = int(user_float) if user_float == int(user_float) else None
            
            # Check if float version exists in options
            if user_float in options:
                return user_float
            
            # Check if int version exists in options
            if user_int is not None and user_int in options:
                return user_int
            
            # Check if string version of int exists (e.g., user types 1, option is '1')
            if user_int is not None and str(user_int) in options:
                return str(user_int)
            
            # Check if string version of float exists (e.g., user types 1.0, option is '1.0')
            if str(user_float) in options:
                return str(user_float)
                
        except ValueError:
            # Not a number, that's fine
            pass
        
        # No match found
        return None
    
    def _get_functor_value_options(self, rule_idx: int) -> Dict[str, List]:
        """
        Get unique possible values for each functor in a rule.
        
        Args:
            rule_idx: Index of the rule
            
        Returns:
            Dictionary mapping functor_name -> list of unique values
        """
        rule = self.rules[rule_idx]
        functor_values = {}
        
        # Initialize dict for each functor
        for functor in rule:
            functor_values[functor] = set()
        
        # Collect unique values from all combinations
        start_idx = self.multiples[rule_idx]
        for table_row in self.values[rule_idx]:
            for functor_idx, functor in enumerate(rule):
                value_idx = start_idx + functor_idx
                if value_idx < len(table_row):
                    functor_value = table_row[value_idx]
                    functor_values[functor].add(functor_value)
        
        # Convert sets to sorted lists
        result = {}
        for functor, values in functor_values.items():
            result[functor] = sorted(list(values), key=lambda x: (isinstance(x, str), x))
        
        return result
    
    def _filter_combinations_by_functor_values(self, rule_idx: int, 
                                              selected_functor_values: Dict[str, List]) -> List[int]:
        """
        Filter combination indices based on selected functor values.
        
        Args:
            rule_idx: Index of the rule
            selected_functor_values: Dict mapping functor_name -> list of selected values
            
        Returns:
            List of row indices that match the selection
        """
        rule = self.rules[rule_idx]
        matching_indices = []
        start_idx = self.multiples[rule_idx]
        
        for row_idx, table_row in enumerate(self.values[rule_idx]):
            # Check if this combination matches all selected functor values
            matches = True
            for functor_idx, functor in enumerate(rule):
                value_idx = start_idx + functor_idx
                if value_idx < len(table_row):
                    functor_value = table_row[value_idx]
                    
                    # Check if this value is in the selected values for this functor
                    if functor in selected_functor_values:
                        if functor_value not in selected_functor_values[functor]:
                            matches = False
                            break
            
            if matches:
                matching_indices.append(row_idx)
        
        return matching_indices
    
    def _get_value_combinations_for_rule(self, rule_idx: int) -> List[Dict]:
        """
        Get all possible value combinations for a given rule.
        
        Args:
            rule_idx: Index of the rule
            
        Returns:
            List of dictionaries, each representing a value combination with functor names and values
        """
        combinations = []
        rule = self.rules[rule_idx]
        
        # Iterate through all rows in self.values[rule_idx]
        for row_idx, table_row in enumerate(self.values[rule_idx]):
            combination = {}
            
            # Extract functor values from the row
            # Skip first element if multiples[rule_idx] == 1
            start_idx = self.multiples[rule_idx]
            
            for functor_idx, functor in enumerate(rule):
                value_idx = start_idx + functor_idx
                if value_idx < len(table_row):
                    functor_value = table_row[value_idx]
                    combination[f"{functor}"] = functor_value
            
            combinations.append({
                'row_index': row_idx,
                'values': combination
            })
        
        return combinations
    
    def _iteration_function_selective(self, reconstructed_x_slice, reconstructed_labels, 
                                     mode: str, selected_rules_values: Dict) -> List:
        """
        Perform iteration over SELECTED rules and values only.
        
        Args:
            reconstructed_x_slice: Node features
            reconstructed_labels: Node labels
            mode: Counting mode
            selected_rules_values: Dict mapping rule_index -> list of value row indices
            
        Returns:
            List of motif counts for selected combinations only
        """
        motif_list = []
        functor_value_dict = dict()
        counter = 0
        counter_c1 = 0
        
        for rule_idx, value_indices in selected_rules_values.items():
            print(f"\nCounting Rule {rule_idx + 1}: {self.rules[rule_idx]}")
            
            for value_idx in value_indices:
                table_row = self.values[rule_idx][value_idx]
                
                # Count motif for this specific rule-value combination
                count, functor_value_dict, counter, counter_c1 = self._count_single_rule_value(
                    rule_idx, value_idx, table_row, 
                    reconstructed_x_slice, reconstructed_labels, mode,
                    functor_value_dict, counter, counter_c1
                )
                
                # Append the count to motif list
                motif_list.append(count)
                print(f"  Value combination {value_idx}: {count}")
        
        return motif_list
    
    def _iteration_function(self, reconstructed_x_slice, reconstructed_labels, mode: str) -> List:
        """
        Perform iteration over all rules to count motifs.
        
        This is the main counting loop that processes each rule and computes
        the final motif count through matrix operations.
        """
        motif_list = []
        functor_value_dict = dict()
        counter = 0
        counter_c1 = 0
        
        for table in range(len(self.rules)):
            # print(self.rules[table])
            
            for indexx, table_row in enumerate(self.values[table]):
                # Count motif for this specific rule-value combination
                count, functor_value_dict, counter, counter_c1 = self._count_single_rule_value(
                    table, indexx, table_row, 
                    reconstructed_x_slice, reconstructed_labels, mode,
                    functor_value_dict, counter, counter_c1
                )
                
                # Append the count to motif list
                motif_list.append(count)
                # print(count)
        
        return motif_list
    
    def get_rule_motif_mapping(self) -> List[Tuple[int, int]]:
        """
        Get mapping of how many motif counts belong to each rule.
        
        Returns:
            List of tuples (rule_index, num_values) indicating how many
            motif counts in the final list correspond to each rule.
        """
        mapping = []
        for rule_idx in range(len(self.rules)):
            num_values = len(self.values[rule_idx])
            mapping.append((rule_idx, num_values))
        return mapping
    
    def aggregate_motif_counts(self, all_motif_counts: List[List]) -> List[float]:
        """
        Aggregate motif counts across multiple graphs by summing same indices.
        
        Args:
            all_motif_counts: List of motif count lists, one per graph
                             e.g., [[graph1_counts], [graph2_counts], ...]
        
        Returns:
            Single list with aggregated counts (sum across all graphs)
        """
        if len(all_motif_counts) == 0:
            return []
        
        # Convert tensors to floats if needed
        num_motifs = len(all_motif_counts[0])
        aggregated = [0.0] * num_motifs
        
        for graph_counts in all_motif_counts:
            for idx, count in enumerate(graph_counts):
                # Handle torch tensors
                if isinstance(count, torch.Tensor):
                    count = count.item()
                aggregated[idx] += count
        
        return aggregated
    
    def display_rules_and_motifs(self, aggregated_counts: List[float], selected_rules_values: Dict = None):
        """
        Display rules with their corresponding aggregated motif counts.
        
        Args:
            aggregated_counts: List of aggregated motif counts
            selected_rules_values: Optional dict from selective counting mode
        """
        print("\n" + "="*80)
        print("RULES AND MOTIF COUNTS")
        print("="*80)
        
        if selected_rules_values is not None:
            # Selective mode: only display selected rules/values
            self._display_selective_results(aggregated_counts, selected_rules_values)
        else:
            # Full mode: display all rules/values
            self._display_full_results(aggregated_counts)
    
    def _display_full_results(self, aggregated_counts: List[float]):
        """Display results when all rules/values were counted."""
        count_idx = 0
        for rule_idx in range(len(self.rules)):
            rule = self.rules[rule_idx]
            num_values = len(self.values[rule_idx])
            
            print(f"\nRule {rule_idx + 1}: {rule}")
            print("-" * 80)
            
            # Display all motif counts for this rule
            for value_idx in range(num_values):
                count = aggregated_counts[count_idx]
                print(f"  Value {value_idx + 1}/{num_values}: {count:.4f}")
                count_idx += 1
            
            # Calculate total for this rule
            # rule_total = sum(aggregated_counts[count_idx - num_values:count_idx])
            # print(f"  → Rule Total: {rule_total:.4f}")
        
        # print("\n" + "="*80)
        # print(f"Grand Total: {sum(aggregated_counts):.4f}")
        # print("="*80)
    
    def _display_selective_results(self, aggregated_counts: List[float], selected_rules_values: Dict):
        """Display results when only specific rules/values were counted."""
        count_idx = 0
        
        for rule_idx, value_indices in selected_rules_values.items():
            rule = self.rules[rule_idx]
            
            print(f"\nRule {rule_idx + 1}: {rule}")
            print("-" * 80)
            
            # Display only the selected value combinations
            rule_counts = []
            start_idx = self.multiples[rule_idx]
            
            for value_idx in value_indices:
                count = aggregated_counts[count_idx]
                table_row = self.values[rule_idx][value_idx]
                
                # Extract functor values for display
                functor_vals = []
                for functor_idx, functor in enumerate(rule):
                    val_idx = start_idx + functor_idx
                    if val_idx < len(table_row):
                        functor_value = table_row[val_idx]
                        functor_vals.append(f"{functor}={functor_value}")
                
                print(f"  [{value_idx}] {', '.join(functor_vals)} → {count:.4f}")
                
                rule_counts.append(count)
                count_idx += 1
            
            # Calculate total for this rule
            # rule_total = sum(rule_counts)
            # print(f"  → Rule Total: {rule_total:.4f}")
        
        # print("\n" + "="*80)
        # print(f"Grand Total: {sum(aggregated_counts):.4f}")
        # print("="*80)
    
    def _compute_unmasked_matrices(self, table, table_row, reconstructed_x_slice, 
                                   reconstructed_labels, mode, functor_value_dict, 
                                   counter, counter_c1):
        """
        Compute unmasked matrices for a given rule and table row.
        
        Processes each atom in the rule and creates corresponding matrices.
        """
        unmasked_matrices = []
        
        for column in range(len(self.rules[table])):
            functor = self.functors[table][column]
            table_functor_value = table_row[column + self.multiples[table]]
            tuple_mask_info = ('0', '0', '0')
            variable = '0'
            functor_value_dict_key = (table_functor_value, functor, variable, tuple_mask_info)
            
            if mode == 'metric_observed':
                if self.states[table][column] != 1:
                    if functor_value_dict.get(functor_value_dict_key) is not None:
                        matrix = functor_value_dict[functor_value_dict_key]
                        unmasked_matrices.append(matrix)
                        counter += 1
                        continue
            
            state = self.states[table][column]
            
            if state == 0:
                # State 0: Unary predicates without relations
                matrix = self._compute_state_zero(
                    functor, table_functor_value, self.nodes[table][column],
                    reconstructed_x_slice, reconstructed_labels, mode
                )
                unmasked_matrices.append(matrix)
                if mode == 'metric_observed':
                    functor_value_dict[functor_value_dict_key] = matrix
                    
            elif state == 1:
                # State 1: Masked variables
                matrices_list, functor_value_dict, counter, counter_c1 = self._compute_state_one(
                    functor, table_functor_value, self.variables[table][column],
                    self.nodes[table][column], self.masks[table][column],
                    reconstructed_x_slice, reconstructed_labels, mode,
                    functor_value_dict, counter, counter_c1
                )
                unmasked_matrices.extend(matrices_list)
                
            elif state == 2:
                # State 2: Known relations
                matrix = self._compute_state_two(functor, table_functor_value)
                unmasked_matrices.append(matrix)
                if mode == 'metric_observed':
                    functor_value_dict[functor_value_dict_key] = matrix
                    
            elif state == 3:
                # State 3: Attribute relations
                matrix = self._compute_state_three(
                    reconstructed_labels, functor, table_functor_value
                )
                unmasked_matrices.append(matrix)
                if mode == 'metric_observed':
                    functor_value_dict[functor_value_dict_key] = matrix
        
        return unmasked_matrices, functor_value_dict, counter, counter_c1
    
    def _compute_state_zero(self, functor, table_functor_value, functor_address, 
                           reconstructed_x_slice, reconstructed_labels, mode):
        """Compute matrix for state 0 (unary predicates without relations)."""
        if mode == 'metric_observed':
            primary_key = self.keys[functor_address]
            matrix = torch.zeros((len(self.entities[functor_address].index), 1), 
                               device=self.device)
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
            found = False
            indx = None
            entity_or_relation_key = None
            
            # Search in entity feature columns
            for key, feature_list in self.entity_feature_columns.items():
                if functor in feature_list:
                    indx = feature_list.index(functor)
                    entity_or_relation_key = key
                    found = True
                    break
            
            # Search in relation feature columns if not found
            if not found:
                for key, feature_list in self.relation_feature_columns.items():
                    if functor in feature_list:
                        indx = feature_list.index(functor)
                        entity_or_relation_key = key
                        found = True
                        break
            
            if found:
                table_functor_value = int(table_functor_value)
                if self.args.test_local_mults:
                    if self.args.graph_type == 'heterogeneous':
                        feature_values = reconstructed_x_slice[entity_or_relation_key][:, indx]
                        matrix = (feature_values == table_functor_value).float().view(-1, 1)
                    else:
                        feature_values = reconstructed_x_slice[:, indx]
                        matrix = (feature_values == table_functor_value).float().view(-1, 1)
                else:
                    print("here")
            else:
                matrix = reconstructed_labels[:, int(table_functor_value)].float().view(-1, 1).to(self.device)
        
        return matrix
    
    def _compute_state_one(self, functor, table_functor_value, variable, functor_address, masks_list,
                          reconstructed_x_slice, reconstructed_labels, mode, functor_value_dict, 
                          counter, counter_c1):
        """Compute matrices for state 1 (masked variables)."""
        matrices_list = []
        primary_key = self.keys[functor_address]
        
        for mask_info in masks_list:
            tuple_mask_info = tuple(mask_info)
            functor_value_dict_key = (table_functor_value, functor, variable, tuple_mask_info)
            
            if mode == 'metric_observed':
                if functor_value_dict.get(functor_value_dict_key) is not None:
                    matrix = functor_value_dict[functor_value_dict_key]
                    matrices_list.append(matrix)
                    counter += 1
                    counter_c1 += 1
                    continue
                
                # Create vector or matrix based on variable position
                if variable == mask_info[1]:
                    matrix = torch.zeros((self.matrices[mask_info[0]].shape[0], 1), 
                                       device=self.device)
                else:
                    matrix = torch.zeros((1, self.matrices[mask_info[0]].shape[1]), 
                                       device=self.device)
                
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
                if mode == 'metric_observed':
                    functor_value_dict[functor_value_dict_key] = matrix
            else:
                # Use reconstructed data
                if variable == mask_info[1]:
                    matrix = self._compute_state_one_variable(
                        functor, table_functor_value, functor_address, 
                        reconstructed_x_slice, reconstructed_labels
                    )
                else:
                    matrix = self._compute_state_one_variable_transpose(
                        functor, table_functor_value, functor_address, 
                        reconstructed_x_slice, reconstructed_labels
                    )
                matrices_list.append(matrix)
        
        return matrices_list, functor_value_dict, counter, counter_c1
    
    def _compute_state_one_variable(self, functor, table_functor_value, functor_address, 
                                   reconstructed_x_slice, reconstructed_labels):
        """Compute matrix for state 1 variable when variable matches mask variable."""
        found = False
        indx = None
        entity_or_relation_key = None
        
        # Search in entity feature columns
        for key, feature_list in self.entity_feature_columns.items():
            if functor in feature_list:
                indx = feature_list.index(functor)
                entity_or_relation_key = key
                found = True
                break
        
        # Search in relation feature columns if not found
        if not found:
            for key, feature_list in self.relation_feature_columns.items():
                if functor in feature_list:
                    indx = feature_list.index(functor)
                    entity_or_relation_key = key
                    found = True
                    break
        
        if found:
            table_functor_value = int(table_functor_value)
            if self.args.test_local_mults:
                if self.args.graph_type == 'heterogeneous':
                    feature_values = reconstructed_x_slice[entity_or_relation_key][:, indx]
                    matrix = (feature_values == table_functor_value).float().view(-1, 1)
                else:
                    feature_values = reconstructed_x_slice[:, indx]
                    matrix = (feature_values == table_functor_value).float().view(-1, 1)
            else:
                print("fix here")
        else:
            matrix = reconstructed_labels[:, int(table_functor_value)].float().view(-1, 1)
        
        return matrix
    
    def _compute_state_one_variable_transpose(self, functor, table_functor_value, functor_address, 
                                             reconstructed_x_slice, reconstructed_labels):
        """Compute matrix for state 1 variable when variable doesn't match mask variable."""
        found = False
        indx = None
        entity_or_relation_key = None
        
        # Search in entity feature columns
        for key, feature_list in self.entity_feature_columns.items():
            if functor in feature_list:
                indx = feature_list.index(functor)
                entity_or_relation_key = key
                found = True
                break
        
        # Search in relation feature columns if not found
        if not found:
            for key, feature_list in self.relation_feature_columns.items():
                if functor in feature_list:
                    indx = feature_list.index(functor)
                    entity_or_relation_key = key
                    found = True
                    break
        
        if found:
            table_functor_value = int(table_functor_value)
            if self.args.test_local_mults:
                if self.args.graph_type == 'heterogeneous':
                    feature_values = reconstructed_x_slice[entity_or_relation_key][:, indx]
                    matrix = (feature_values == table_functor_value).float().view(1, -1)
                else:
                    feature_values = reconstructed_x_slice[:, indx]
                    matrix = (feature_values == table_functor_value).float().view(1, -1)
            else:
                print("fix here")
        else:
            matrix = reconstructed_labels[:, int(table_functor_value)].view(1, -1).to(self.device)
        
        return matrix
    
    def _compute_state_two(self, functor, table_functor_value):
        """Retrieve or invert the relation matrix for state 2."""
        if table_functor_value == 'F':
            # Invert matrix if value is 'F' (false)
            matrix = 1 - self.matrices[functor]
        else:
            matrix = self.matrices[functor]
        return matrix
    
    def _compute_state_three(self, reconstructed_labels, functor, table_functor_value):
        """Compute matrix for state 3 (attribute relations)."""
        mode = False
        
        if mode == True:
            table_name = self.attributes[functor]
            primary_key = self.keys[table_name]
            
            if table_functor_value == 'N/A':
                matrix = 1 - self.matrices[table_name]
            else:
                matrix = torch.zeros_like(self.matrices[table_name], device=self.device)
                for index_relation in range(len(self.relations[table_name][functor])):
                    functor_value = self.relations[table_name][functor][index_relation]
                    if functor_value == table_functor_value:
                        pk0_value = self.relations[table_name][primary_key[0]][index_relation]
                        pk1_value = self.relations[table_name][primary_key[1]][index_relation]
                        index1 = self.indices[primary_key[0]][pk0_value]
                        index2 = self.indices[primary_key[1]][pk1_value]
                        matrix[index1, index2] = 1
        else:
            feature_idx = None
            for idx, info in self.feature_info_mapping.items():
                if info['feature_name'] == functor:
                    feature_idx = idx
                    break
            
            target_tensor = reconstructed_labels[feature_idx]
            
            if table_functor_value == 'N/A':
                matrix = torch.sum(target_tensor, dim=0)
            else:
                value_mapping = self.feature_info_mapping[feature_idx]['value_index_mapping']
                reverse_mapping = {v: k for k, v in value_mapping.items()}
                value_idx = reverse_mapping[int(table_functor_value)]
                matrix = target_tensor[value_idx]
        
        return matrix
    
    def _compute_masked_matrices(self, unmasked_matrices, base_indices, mask_indices):
        """Apply masking to unmasked matrices based on base and mask indices."""
        # Initialize with base matrices
        masked_matrices = [unmasked_matrices[k] for k in base_indices]
        
        # Apply element-wise multiplication for masking
        for k in mask_indices:
            masked_matrices[k[0]] = torch.mul(masked_matrices[k[0]], unmasked_matrices[k[1]])
        
        return masked_matrices
    
    def _compute_sorted_matrices(self, masked_matrices, sort_indices):
        """Sort masked matrices based on sort indices."""
        sorted_matrices = []
        
        for k in sort_indices:
            if k[0]:
                # Transpose if needed
                sorted_matrices.append(masked_matrices[k[1]].T)
            else:
                sorted_matrices.append(masked_matrices[k[1]])
        
        return sorted_matrices
    
    def _compute_stacked_matrices(self, sorted_matrices, stack_indices):
        """Stack matrices according to stack indices for multiplication."""
        stacked_matrices = sorted_matrices.copy()
        pop_counter = 0
        
        for k in stack_indices:
            for _ in range(k[1] - k[0] - pop_counter):
                # Multiply adjacent matrices
                stacked_matrices[k[0]] = torch.mm(stacked_matrices[k[0]], stacked_matrices[k[0] + 1])
                stacked_matrices.pop(k[0] + 1)
                pop_counter += 1
            
            # Element-wise multiplication with identity
            stacked_matrices[k[0]] = torch.mul(
                stacked_matrices[k[0]],
                torch.eye(len(stacked_matrices[k[0]]), device=self.device)
            )
        
        return stacked_matrices
    
    def _compute_result(self, stacked_matrices):
        """Compute final result by multiplying all stacked matrices."""
        result = stacked_matrices[0]
        
        for k in range(1, len(stacked_matrices)):
            result = torch.mm(result, stacked_matrices[k])
        
        return result



    def _count_single_rule_value(self, table: int, indexx: int, table_row,
                                 reconstructed_x_slice, reconstructed_labels, mode: str,
                                 functor_value_dict: dict, counter: int, counter_c1: int):

        # Compute unmasked matrices for this rule
        unmasked_matrices, functor_value_dict, counter, counter_c1 = self._compute_unmasked_matrices(
            table, table_row, reconstructed_x_slice, reconstructed_labels, mode,
            functor_value_dict, counter, counter_c1
        )
        
        # Apply masking
        masked_matrices = self._compute_masked_matrices(
            unmasked_matrices, 
            self.base_indices[table], 
            self.mask_indices[table]
        )
        
        # Sort matrices for multiplication
        sorted_matrices = self._compute_sorted_matrices(
            masked_matrices, 
            self.sort_indices[table]
        )
        
        # Stack matrices according to dependencies
        stacked_matrices = self._compute_stacked_matrices(
            sorted_matrices, 
            self.stack_indices[table]
        )
        
        # Compute final result through matrix multiplication
        result = self._compute_result(stacked_matrices)
        
        # Calculate count with optional weighting
        if self.args.rule_weight:
            count = torch.sum(result) * self.prunes[table][indexx]
        else:
            count = torch.sum(result)
        
        # Cleanup to free memory
        del unmasked_matrices, masked_matrices, sorted_matrices, stacked_matrices, result
        
        return count, functor_value_dict, counter, counter_c1
