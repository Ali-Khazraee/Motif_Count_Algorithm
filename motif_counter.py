# motif_counter.py

import torch
import numpy as np
from typing import List, Dict, Tuple, Any

from motif_store import RuleBasedMotifStore


class RelationalMotifCounter:
    """
    Counts motifs in a graph using relational algebra and Bayesian Network rules.
    Implements the complete motif counting pipeline.
    """
    
    def __init__(self, motif_store: RuleBasedMotifStore):
        """
        Initialize the motif counter with a populated motif store.
        
        Args:
            motif_store: RuleBasedMotifStore containing all rules and data
        """
        self.store = motif_store
        self.args = motif_store.args
        
    def count(self, graph_data: Dict) -> List[float]:
        """
        Main entry point: counts all motifs in a graph.
        
        Args:
            graph_data: Dictionary containing:
                - 'adjacency_matrix': Adjacency tensor
                - 'features': Node features
                - 'labels': Node labels (optional)
                
        Returns:
            List of motif counts (one count per rule)
        """
        # Process and update graph data
        reconstructed_x_slice, reconstructed_labels = self._process_graph_data(graph_data)
        
        # Perform iteration to count motifs
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
        key = list(self.store.matrices.keys())[0]
        self.store.matrices[key] = adjacency.to(self.store.device)
        
        # Process features
        reconstructed_x_slice = torch.tensor(features).to(self.store.device)
        reconstructed_labels = None
        
        return reconstructed_x_slice, reconstructed_labels
    
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
        
        for table in range(len(self.store.rules)):
            print(self.store.rules[table])
            indexx = -1
            
            for table_row in self.store.values[table]:
                indexx += 1
                
                # Compute unmasked matrices for this rule
                unmasked_matrices, functor_value_dict, counter, counter_c1 = self._compute_unmasked_matrices(
                    table, table_row, reconstructed_x_slice, reconstructed_labels, mode,
                    functor_value_dict, counter, counter_c1
                )
                
                # Apply masking
                masked_matrices = self._compute_masked_matrices(
                    unmasked_matrices, 
                    self.store.base_indices[table], 
                    self.store.mask_indices[table]
                )
                
                # Sort matrices for multiplication
                sorted_matrices = self._compute_sorted_matrices(
                    masked_matrices, 
                    self.store.sort_indices[table]
                )
                
                # Stack matrices according to dependencies
                stacked_matrices = self._compute_stacked_matrices(
                    sorted_matrices, 
                    self.store.stack_indices[table]
                )
                
                # Compute final result through matrix multiplication
                result = self._compute_result(stacked_matrices)
                
                # Append result with optional weighting
                if self.args.rule_weight:
                    motif_list.append(torch.sum(result) * self.store.prunes[table][indexx])
                else:
                    motif_list.append(torch.sum(result))
                
                print(torch.sum(result))
                
                # Cleanup to free memory
                del unmasked_matrices, masked_matrices, sorted_matrices, stacked_matrices, result
        
        return motif_list
    
    def _compute_unmasked_matrices(self, table, table_row, reconstructed_x_slice, 
                                   reconstructed_labels, mode, functor_value_dict, 
                                   counter, counter_c1):
        """
        Compute unmasked matrices for a given rule and table row.
        
        Processes each atom in the rule and creates corresponding matrices.
        """
        unmasked_matrices = []
        
        for column in range(len(self.store.rules[table])):
            functor = self.store.functors[table][column]
            table_functor_value = table_row[column + self.store.multiples[table]]
            tuple_mask_info = ('0', '0', '0')
            variable = '0'
            functor_value_dict_key = (table_functor_value, functor, variable, tuple_mask_info)
            
            if mode == 'metric_observed':
                if self.store.states[table][column] != 1:
                    if functor_value_dict.get(functor_value_dict_key) is not None:
                        matrix = functor_value_dict[functor_value_dict_key]
                        unmasked_matrices.append(matrix)
                        counter += 1
                        continue
            
            state = self.store.states[table][column]
            
            if state == 0:
                # State 0: Unary predicates without relations
                matrix = self._compute_state_zero(
                    functor, table_functor_value, self.store.nodes[table][column],
                    reconstructed_x_slice, reconstructed_labels, mode
                )
                unmasked_matrices.append(matrix)
                if mode == 'metric_observed':
                    functor_value_dict[functor_value_dict_key] = matrix
                    
            elif state == 1:
                # State 1: Masked variables
                matrices_list, functor_value_dict, counter, counter_c1 = self._compute_state_one(
                    functor, table_functor_value, self.store.variables[table][column],
                    self.store.nodes[table][column], self.store.masks[table][column],
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
            primary_key = self.store.keys[functor_address]
            matrix = torch.zeros((len(self.store.entities[functor_address].index), 1), 
                               device=self.store.device)
            for entity_index in range(len(self.store.entities[functor_address][functor])):
                functor_value = self.store.entities[functor_address][functor][entity_index]
                if isinstance(table_functor_value, str):
                    if isinstance(functor_value, (np.int64, np.int32)):
                        functor_value = str(functor_value)
                    elif isinstance(functor_value, (np.float64, np.float32)):
                        functor_value = str(int(functor_value))
                if functor_value == table_functor_value:
                    key_index = self.store.entities[functor_address][primary_key][entity_index]
                    row_index = self.store.indices[primary_key][key_index]
                    matrix[row_index][0] = 1
        else:
            found = False
            indx = None
            entity_or_relation_key = None
            
            # Search in entity feature columns
            for key, feature_list in self.store.entity_feature_columns.items():
                if functor in feature_list:
                    indx = feature_list.index(functor)
                    entity_or_relation_key = key
                    found = True
                    break
            
            # Search in relation feature columns if not found
            if not found:
                for key, feature_list in self.store.relation_feature_columns.items():
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
                matrix = reconstructed_labels[:, int(table_functor_value)].float().view(-1, 1).to(self.store.device)
        
        return matrix
    
    def _compute_state_one(self, functor, table_functor_value, variable, functor_address, masks_list,
                          reconstructed_x_slice, reconstructed_labels, mode, functor_value_dict, 
                          counter, counter_c1):
        """Compute matrices for state 1 (masked variables)."""
        matrices_list = []
        primary_key = self.store.keys[functor_address]
        
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
                    matrix = torch.zeros((self.store.matrices[mask_info[0]].shape[0], 1), 
                                       device=self.store.device)
                else:
                    matrix = torch.zeros((1, self.store.matrices[mask_info[0]].shape[1]), 
                                       device=self.store.device)
                
                for entity_index in range(len(self.store.entities[functor_address][functor])):
                    functor_value = self.store.entities[functor_address][functor][entity_index]
                    if functor_value == table_functor_value:
                        key_index = self.store.entities[functor_address][primary_key][entity_index]
                        index = self.store.indices[primary_key][key_index]
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
        for key, feature_list in self.store.entity_feature_columns.items():
            if functor in feature_list:
                indx = feature_list.index(functor)
                entity_or_relation_key = key
                found = True
                break
        
        # Search in relation feature columns if not found
        if not found:
            for key, feature_list in self.store.relation_feature_columns.items():
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
        for key, feature_list in self.store.entity_feature_columns.items():
            if functor in feature_list:
                indx = feature_list.index(functor)
                entity_or_relation_key = key
                found = True
                break
        
        # Search in relation feature columns if not found
        if not found:
            for key, feature_list in self.store.relation_feature_columns.items():
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
            matrix = reconstructed_labels[:, int(table_functor_value)].view(1, -1).to(self.store.device)
        
        return matrix
    
    def _compute_state_two(self, functor, table_functor_value):
        """Retrieve or invert the relation matrix for state 2."""
        if table_functor_value == 'F':
            # Invert matrix if value is 'F' (false)
            matrix = 1 - self.store.matrices[functor]
        else:
            matrix = self.store.matrices[functor]
        return matrix
    
    def _compute_state_three(self, reconstructed_labels, functor, table_functor_value):
        """Compute matrix for state 3 (attribute relations)."""
        mode = False
        
        if mode == True:
            table_name = self.store.attributes[functor]
            primary_key = self.store.keys[table_name]
            
            if table_functor_value == 'N/A':
                matrix = 1 - self.store.matrices[table_name]
            else:
                matrix = torch.zeros_like(self.store.matrices[table_name], device=self.store.device)
                for index_relation in range(len(self.store.relations[table_name][functor])):
                    functor_value = self.store.relations[table_name][functor][index_relation]
                    if functor_value == table_functor_value:
                        pk0_value = self.store.relations[table_name][primary_key[0]][index_relation]
                        pk1_value = self.store.relations[table_name][primary_key[1]][index_relation]
                        index1 = self.store.indices[primary_key[0]][pk0_value]
                        index2 = self.store.indices[primary_key[1]][pk1_value]
                        matrix[index1, index2] = 1
        else:
            feature_idx = None
            for idx, info in self.store.feature_info_mapping.items():
                if info['feature_name'] == functor:
                    feature_idx = idx
                    break
            
            target_tensor = reconstructed_labels[feature_idx]
            
            if table_functor_value == 'N/A':
                matrix = torch.sum(target_tensor, dim=0)
            else:
                value_mapping = self.store.feature_info_mapping[feature_idx]['value_index_mapping']
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
                torch.eye(len(stacked_matrices[k[0]]), device=self.store.device)
            )
        
        return stacked_matrices
    
    def _compute_result(self, stacked_matrices):
        """Compute final result by multiplying all stacked matrices."""
        result = stacked_matrices[0]
        
        for k in range(1, len(stacked_matrices)):
            result = torch.mm(result, stacked_matrices[k])
        
        return result