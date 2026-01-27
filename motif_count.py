import torch
import numpy as np
from pymysql import connect
from pandas import DataFrame
from itertools import permutations
from math import log
import pickle


class Motif_Count:
# Motif_Count class for counting motifs in graphs based on given rules
    def __init__(self, args):
        # Initialize the Motif_Count class
        """
        Initialize the Motif_Count class and set up necessary data structures.
        """
        self.args = args


        self.rules = []
        self.multiples = []
        self.states = []
        self.functors = {}
        self.variables = {}
        self.nodes = {}
        self.masks = {}
        self.base_indices = []
        self.mask_indices = []
        self.sort_indices = []
        self.stack_indices = []
        self.values = []
        self.prunes = []
        self.keys = {}
        self.indices = {}
        self.matrices = {}
        self.entities = {}
        self.attributes = {}
        self.relations = {}
        self.device = "cuda"
        self.entity_feature_columns = {}
        self.relation_feature_columns = {}
        # self.mapping_details = mapping_details
        # self.important_feat_ids = important_feat_ids

        
        # self.setup_function()




    def load_setup_variables(self, file_path):
        """
        Load all important variables from a .pkl file.
        
        Parameters:
            file_path (str): Path to the .pkl file to load the variables.
        """
        with open(file_path, "rb") as f:
            important_variables = pickle.load(f)

        # Assign loaded variables to class attributes
        self.entities = important_variables["entities"]
        self.relations = important_variables["relations"]
        self.keys = important_variables["keys"]
        self.matrices = important_variables["matrices"]
        self.rules = important_variables["rules"]
        self.indices = important_variables["indices"]
        self.attributes = important_variables["attributes"]
        self.base_indices = important_variables["base_indices"]
        self.mask_indices = important_variables["mask_indices"]
        self.sort_indices = important_variables["sort_indices"]
        self.stack_indices = important_variables["stack_indices"]
        self.values = important_variables["values"]
        self.prunes = important_variables["prunes"]
        self.functors = important_variables["functors"]
        self.variables = important_variables["variables"]
        self.nodes = important_variables["nodes"]
        self.states = important_variables["states"]
        self.masks = important_variables["masks"]
        self.multiples = important_variables["multiples"]







# Main setup function to initialize data structures and process rules
    def setup_function(self):
        """
        Main setup function to initialize data from SQL, create indices,
        mask matrices, and process rules and values.
        """
        # Fetch data from SQL
        self.fetch_data_from_sql()

        self.create_feature_info_mapping()

        # Create indices for quick lookup
        self.create_indices()

        # Create mask matrices based on relations
        self.create_mask_matrices()

        # Process rules and values for motif counting
        self.process_rules()

        # Close database connections
        self.cursor.close()
        self.connection.close()
        self.cursor_setup.close()
        self.connection_setup.close()
        self.cursor_bn.close()
        self.connection_bn.close()


        # self.load_setup_variables('acm_no_prune.pkl')


# Fetch data from the SQL database and establish connections
    def fetch_data_from_sql(self):
        """
        Fetch entities, relations, attributes, and keys from the SQL database.
        Establish database connections.
        """
        database_name = {
            "Cora_dgl": "cora",
            "citeseer": "citeseer",
            "imdb-multi": "imdb",
            "acm-multi": "acm-multi",
            "CiteSeer_dgl":"citeseer", 
            "IMDB":"imdb", 
            "grid":"grid"
        }
        self.db_name = database_name[self.args.dataset]
        self.db = self.db_name
        host_name = 'localhost'
        user_name = 'fbuser'
        password_name = ''

        # Connect to main database
        self.connection = connect(host=host_name, user=user_name, password=password_name, db=self.db_name)
        self.cursor = self.connection.cursor()

        # Connect to setup database
        db_setup = self.db_name + "_setup"
        self.connection_setup = connect(host=host_name, user=user_name, password=password_name, db=db_setup)
        self.cursor_setup = self.connection_setup.cursor()

        # Connect to Bayesian Network (BN) database
        db_bn = self.db_name + "_BN"
        self.connection_bn = connect(host=host_name, user=user_name, password=password_name, db=db_bn)
        self.cursor_bn = self.connection_bn.cursor()

        # Fetch entity tables and their primary keys
        self.cursor_setup.execute("SELECT TABLE_NAME FROM EntityTables")
        entity_tables = self.cursor_setup.fetchall()
        self.entities = {}
        self.keys = {}

        for (table_name,) in entity_tables:
            # Fetch all rows from the entity table
            self.cursor.execute(f"SELECT * FROM {table_name}")
            rows = self.cursor.fetchall()

            # Get column names
            self.cursor.execute(f"SHOW COLUMNS FROM {self.db}.{table_name}")
            columns = self.cursor.fetchall()
            columns_names = [column[0] for column in columns]
            self.entity_feature_columns[table_name] = columns_names[1:]  # Exclude primary key column


            # Store the entity data in a DataFrame
            self.entities[table_name] = DataFrame(rows, columns=columns_names)

            # Get the primary key for the entity table
            self.cursor_setup.execute("SELECT COLUMN_NAME FROM EntityTables WHERE TABLE_NAME = %s", (table_name,))
            key = self.cursor_setup.fetchall()
            self.keys[table_name] = key[0][0]

        # Fetch relation tables and their foreign keys
        self.cursor_setup.execute("SELECT TABLE_NAME FROM RelationTables")
        relation_tables = self.cursor_setup.fetchall()
        self.relations = {}
        for (table_name,) in relation_tables:
            # Fetch all rows from the relation table
            self.cursor.execute(f"SELECT * FROM {table_name}")
            rows = self.cursor.fetchall()

            # Get column names
            self.cursor.execute(f"SHOW COLUMNS FROM {self.db}.{table_name}")
            columns = self.cursor.fetchall()
            columns_names = [column[0] for column in columns]
            self.relation_feature_columns[table_name] = columns_names[2:]  # Exclude primary keys column


            # Store the relation data in a DataFrame
            self.relations[table_name] = DataFrame(rows, columns=columns_names)

            # Get foreign keys for the relation table
            self.cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
            key = self.cursor_setup.fetchall()
            self.keys[table_name] = (key[0][0], key[1][0])

        # Fetch attributes
        self.cursor_setup.execute("SELECT COLUMN_NAME, TABLE_NAME FROM AttributeColumns")
        attribute_columns = self.cursor_setup.fetchall()
        self.attributes = {}
        for column_name, table_name in attribute_columns:
            self.attributes[column_name] = table_name


# Create indices for entities based on their primary keys
    def create_indices(self):
        """
        Create indices for quick lookup of entity keys.
        """
        self.indices = {}
        for table_name, df in self.entities.items():
            key = self.keys[table_name]
            # Map primary key values to their row indices
            self.indices[key] = {row[key]: idx for idx, row in df.iterrows()}


# Create mask matrices representing relations between entities
    def create_mask_matrices(self):
        """
        Create mask matrices for relations to represent connections between entities.
        """
        self.matrices = {}
        for table_name, df in self.relations.items():
            # Get foreign keys and referenced entities
            self.cursor_setup.execute("SELECT REFERENCED_TABLE_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
            reference = self.cursor_setup.fetchall()
            entity1 = reference[0][0]
            entity2 = reference[1][0]

            # Initialize the matrix with zeros
            shape = (len(self.entities[entity1].index), len(self.entities[entity2].index))
            self.matrices[table_name] = torch.zeros(shape, dtype=torch.float32, device=self.device)

        for table_name, df in self.relations.items():
            # Populate the mask matrices with ones where relations exist
            self.cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
            key = self.cursor_setup.fetchall()
            self.cursor_setup.execute("SELECT COLUMN_NAME, REFERENCED_COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
            reference = self.cursor_setup.fetchall()

            # Collect row and column indices for non-zero entries
            rows_indices = []
            cols_indices = []
            for index, row in df.iterrows():
                row_index = self.indices[reference[0][1]][row[key[0][0]]]
                col_index = self.indices[reference[1][1]][row[key[1][0]]]
                rows_indices.append(row_index)
                cols_indices.append(col_index)

            # Convert indices to tensors
            rows_indices_tensor = torch.tensor(rows_indices, dtype=torch.long)
            cols_indices_tensor = torch.tensor(cols_indices, dtype=torch.long)

            # Set corresponding entries in the matrix to 1
            self.matrices[table_name][rows_indices_tensor, cols_indices_tensor] = 1



# Process the rules from the Bayesian Network and prepare for motif counting
    def process_rules(self):
        """
        Process rules from the Bayesian Network and prepare data structures
        for motif counting.
        """
        # Fetch distinct children (target variables) from the Bayesian Network view
        self.cursor_bn.execute("SELECT DISTINCT child FROM Final_Path_BayesNets_view")
        childs = self.cursor_bn.fetchall()

        # Get relation names for reference
        relation_names = tuple(self.relations.keys())

        for i in range(len(childs)):
            # Build the rule starting with the child
            rule = [childs[i][0]]
            # Fetch parents of the child to complete the rule
            self.cursor_bn.execute("SELECT parent FROM Final_Path_BayesNets_view WHERE child = %s", (childs[i][0],))
            parents = self.cursor_bn.fetchall()
            for (parent,) in parents:
                if parent != '':
                    rule.append(parent)
            self.rules.append(rule)

            # Determine if the rule is multiple (has parents)
            self.multiples.append(1 if len(rule) > 1 else 0)

            # Check if the rule contains relations
            relation_check = any(',' in atom for atom in rule)

            # Initialize structures to store functors, variables, and states
            functor = {}
            variable = {}
            node = {}
            state = []
            mask = {}
            unmasked_variables = []

            for j in range(len(rule)):
                fun = rule[j].split('(')[0]
                functor[j] = fun
                if ',' not in rule[j]:
                    # Handle unary predicates
                    var = rule[j].split('(')[1][:-1]
                    variable[j] = var
                    node[j] = var[:-1]
                    if not relation_check:
                        unmasked_variables.append(var)
                        state.append(0)  # State 0: No relation
                    else:
                        mas = []
                        for k in rule:
                            func = k.split('(')[0]
                            if func not in relation_names:
                                func = self.attributes.get(func, func)
                            if ',' in k and var in k:
                                var1, var2 = k.split('(')[1][:-1].split(',')
                                mas.append([func, var1, var2])
                                unmasked_variables.append(k.split('(')[1][:-1])
                        mask[j] = mas
                        state.append(1)  # State 1: Masked variable
                else:
                    # Handle binary predicates (relations)
                    unmasked_variables.append(rule[j].split('(')[1][:-1])
                    if fun in relation_names:
                        state.append(2)  # State 2: Known relation
                    else:
                        state.append(3)  # State 3: Attribute relation
            self.functors[i] = functor
            self.variables[i] = variable
            self.nodes[i] = node
            self.states.append(state)
            self.masks[i] = mask

            # Identify base and mask indices for variables
            masked_variables = [unmasked_variables[0]]
            base_indice = [0]
            mask_indice = []
            for j in range(1, len(unmasked_variables)):
                mask_check = False
                for k in range(len(masked_variables)):
                    if unmasked_variables[j] == masked_variables[k]:
                        mask_indice.append([k, j])
                        mask_check = True
                        break
                if not mask_check:
                    base_indice.append(j)
                    masked_variables.append(unmasked_variables[j])

            # Create sort indices for variable ordering
            sort_indice, sorted_variables = self.create_sort_indices(masked_variables, relation_check, relation_names)

            self.base_indices.append(base_indice)
            self.mask_indices.append(mask_indice)
            self.sort_indices.append(sort_indice)

            # Create stack indices for matrix operations
            stack_indice = self.create_stack_indices(sorted_variables)
            self.stack_indices.append(stack_indice)

            # Fetch conditional probabilities for the rule
            self.cursor_bn.execute(f"SELECT * FROM `{childs[i][0]}_CP`")
            value = self.cursor_bn.fetchall()
            if self.args.rule_prune and not self.args.rule_weight:
                # Prune rules based on statistical significance
                pruned_value = []
                for j in value:
                    size = len(j)
                    if self.multiples[i]:
                        if 2 * j[size - 4] * (log(j[size - 3]) - log(j[size - 1])) - log(j[size - 4]) > 0:
                            pruned_value.append(j)
                    else:
                        if 2 * int(j[size - 3]) * (log(j[size - 5]) - log(j[size - 1])) - log(int(j[size - 3])) > 0:
                            pruned_value.append(j)
                self.values.append(pruned_value)
            elif self.args.rule_prune and self.args.rule_weight:
                # Prune and weight rules
                pruned_value = []
                prune = []
                for j in value:
                    size = len(j)
                    if self.multiples[i]:
                        p = 2 * j[size - 4] * (log(j[size - 3]) - log(j[size - 1])) - log(j[size - 4])
                        if p > 0:
                            pruned_value.append(j)
                            prune.append(p)
                    else:
                        p = 2 * int(j[size - 3]) * (log(j[size - 5]) - log(j[size - 1])) - log(int(j[size - 3]))
                        if p > 0:
                            pruned_value.append(j)
                            prune.append(p)
                self.prunes.append(prune)
                self.values.append(pruned_value)
            elif not self.args.rule_prune and self.args.rule_weight:
                raise Exception('Rule weighting requires rule pruning to be enabled.')
            else:
                self.values.append(value)

        # Adjust matrices to correct shapes if necessary
        self.adjust_matrices()


    # Create indices to sort variables correctly for matrix operations
    def create_sort_indices(self, masked_variables, relation_check, relation_names):
        """
        Create indices to sort variables in the correct order for matrix operations.
        """
        sort_indice = []
        sorted_variables = []
        if not relation_check:
            # If no relations, sorting is straightforward
            sort_indice.append([False, 0])
            sorted_variables.append(masked_variables[0])
        else:
            # Try to find a valid permutation that forms a chain for matrix multiplication
            indices_permutations = list(permutations(range(len(masked_variables))))
            variables_permutations = list(permutations(masked_variables))
            found_chain = False
            for idx_perm, var_perm in zip(indices_permutations, variables_permutations):
                indices_chain = []
                variables_chain = []
                first = var_perm[0].split(',')[0]
                second = var_perm[0].split(',')[1]
                indices_chain.append([False, idx_perm[0]])
                variables_chain.append(var_perm[0])
                untransposed_check = True
                for k in range(1, len(var_perm)):
                    next_first = var_perm[k].split(',')[0]
                    next_second = var_perm[k].split(',')[1]
                    if second == next_first:
                        second = next_second
                        indices_chain.append([False, idx_perm[k]])
                        variables_chain.append(var_perm[k])
                    elif second == next_second:
                        second = next_first
                        indices_chain.append([True, idx_perm[k]])
                        variables_chain.append(next_second + ',' + next_first)
                    else:
                        untransposed_check = False
                        break
                if untransposed_check:
                    sort_indice = indices_chain
                    sorted_variables = variables_chain
                    found_chain = True
                    break
            if not found_chain:
                # Try transposing the first element if no chain found
                indices_chain = []
                variables_chain = []
                first = var_perm[0].split(',')[1]
                second = var_perm[0].split(',')[0]
                indices_chain.append([True, idx_perm[0]])
                variables_chain.append(first + ',' + second)
                untransposed_check = True
                for k in range(1, len(var_perm)):
                    next_first = var_perm[k].split(',')[0]
                    next_second = var_perm[k].split(',')[1]
                    if second == next_first:
                        second = next_second
                        indices_chain.append([False, idx_perm[k]])
                        variables_chain.append(var_perm[k])
                    elif second == next_second:
                        second = next_first
                        indices_chain.append([True, idx_perm[k]])
                        variables_chain.append(next_second + ',' + next_first)
                    else:
                        untransposed_check = False
                        break
                if untransposed_check:
                    sort_indice = indices_chain
                    sorted_variables = variables_chain
                    found_chain = True
        return sort_indice, sorted_variables


# Create indices for stacking matrices in the correct order
    def create_stack_indices(self, sorted_variables):
        """
        Create indices for stacking matrices in the correct order.
        """
        stack_indices = []
        for j in range(1, len(sorted_variables)):
            second = sorted_variables[j].split(',')[1]
            for k in range(j -1, -1, -1):
                previous_first = sorted_variables[k].split(',')[0]
                if previous_first == second:
                    stack_indices.append([k, j])
        return stack_indices


# Adjust matrices to have the correct shape by transposing if necessary
    def adjust_matrices(self):
        """
        Adjust matrices to have the correct shape by transposing if necessary.
        """
        # Identify unique relation functors in the rules
        relation_functors = [item for sublist in self.rules for item in sublist if ',' in item and item in self.relations.keys()]
        unique_relation_functors = list(set(relation_functors))
        for relation_functor in unique_relation_functors:
            # Extract involved entities
            entities_involved = relation_functor.replace(')', '').split('(')[1].split(',')
            entities_clean = [entity[:-1] for entity in entities_involved]
            correct_shape = (len(self.entities[entities_clean[0]]), len(self.entities[entities_clean[1]]))
            matrix_name = relation_functor.split('(')[0]
            # Transpose the matrix if shapes do not match
            if self.matrices[matrix_name].shape != correct_shape:
                self.matrices[matrix_name] = self.matrices[matrix_name].t()



# Perform an iteration over the rules to count motifs

    def iteration_function(self, reconstructed_x_slice, reconstructed_labels, mode):
        """
    
        Perform an iteration over the rules to count motifs based on the reconstructed data.
        """
        motif_list = []
        functor_value_dict = dict()
        counter = 0
        counter_c1 = 0

        for table in range(len(self.rules)):
            # if table != 0:
            #     continue
            print(self.rules[table])
            indexx = -1
            for table_row in self.values[table]:
                indexx += 1
                # Compute unmasked matrices for the current rule and table row
                unmasked_matrices, functor_value_dict, counter, counter_c1 = self.compute_unmasked_matrices(
                     table, table_row, reconstructed_x_slice, reconstructed_labels, mode,
                    functor_value_dict, counter, counter_c1, self.feature_info_mapping
                )

                # Apply masking to the matrices
                masked_matrices = self.compute_masked_matrices(unmasked_matrices, self.base_indices[table], self.mask_indices[table])

                # Sort matrices based on the sort indices
                sorted_matrices = self.compute_sorted_matrices(masked_matrices, self.sort_indices[table])

                # Stack matrices according to the stack indices
                stacked_matrices = self.compute_stacked_matrices(sorted_matrices, self.stack_indices[table])

                # Compute the final result by multiplying stacked matrices
                result = self.compute_result(stacked_matrices)

                # Append the result to the motif list, applying pruning weight if necessary
                if self.args.rule_weight:
                    motif_list.append(torch.sum(result) * self.prunes[table][indexx]) 
                else:
                    motif_list.append(torch.sum(result))

                print(torch.sum(result))


                # Clean up to free memory
                del unmasked_matrices, masked_matrices, sorted_matrices, stacked_matrices, result

        return motif_list


# Compute the unmasked matrices for a given rule and table row
    def compute_unmasked_matrices(self, table, table_row, reconstructed_x_slice, reconstructed_labels, mode,
                                  functor_value_dict, counter, counter_c1, feature_info_mapping):
        """
        Compute the unmasked matrices for a given rule and table row.
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
                    # Use cached matrix if available
                    if functor_value_dict.get(functor_value_dict_key) is not None:
                        matrix = functor_value_dict[functor_value_dict_key]
                        unmasked_matrices.append(matrix)
                        counter += 1
                        continue
            state = self.states[table][column]

            if state == 0:
                # Compute matrix for state 0
                matrix = self.compute_state_zero(
                     functor, table_functor_value, self.nodes[table][column], reconstructed_x_slice, reconstructed_labels, mode
                )
                unmasked_matrices.append(matrix)
                if mode == 'metric_observed':
                    functor_value_dict[functor_value_dict_key] = matrix
            elif state == 1:
                # Compute matrices for state 1
                matrices_list, functor_value_dict, counter, counter_c1 = self.compute_state_one(
                     functor, table_functor_value, self.variables[table][column], self.nodes[table][column], self.masks[table][column],
                    reconstructed_x_slice, reconstructed_labels, mode,
                    functor_value_dict, counter, counter_c1
                )
                unmasked_matrices.extend(matrices_list)
            elif state == 2:
                # Use the relation matrix for state 2
                matrix = self.compute_state_two(functor, table_functor_value)
                unmasked_matrices.append(matrix)
                if mode == 'metric_observed':
                    functor_value_dict[functor_value_dict_key] = matrix
            elif state == 3:
                # Compute matrix for attribute relations in state 3
                matrix = self.compute_state_three(reconstructed_labels,
                    functor, table_functor_value, feature_info_mapping
                )
                unmasked_matrices.append(matrix)
                if mode == 'metric_observed':
                    functor_value_dict[functor_value_dict_key] = matrix
        return unmasked_matrices, functor_value_dict, counter, counter_c1


# Compute matrix for state 0 (unary predicates without relations)
    def compute_state_zero(self, functor, table_functor_value, functor_address, reconstructed_x_slice, reconstructed_labels, mode):
        """
        Compute matrix for state 0, which involves unary predicates without relations.
        """
        if mode == 'metric_observed':
            # Create a column vector indicating entities matching the functor value
            primary_key = self.keys[functor_address]
            matrix = torch.zeros((len(self.entities[functor_address].index), 1), device=self.device)
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
            # Search for functor in entity feature columns
            found = False
            indx = None
            entity_or_relation_key = None
            
            # First search in entity feature columns
            for key, feature_list in self.entity_feature_columns.items():
                if functor in feature_list:
                    indx = feature_list.index(functor)
                    entity_or_relation_key = key
                    found = True
                    break
            
            # If not found in entity features, search in relation feature columns
            if not found:
                for key, feature_list in self.relation_feature_columns.items():
                    if functor in feature_list:
                        indx = feature_list.index(functor)
                        entity_or_relation_key = key
                        found = True
                        break
            
            # If found in feature columns, use the index
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

    def compute_state_one(self, functor, table_functor_value, variable, functor_address, masks_list,
                          reconstructed_x_slice, reconstructed_labels, mode, functor_value_dict, counter, counter_c1):
        # Compute matrices for state 1 (masked variables)
        """
        Compute matrices for state 1, which involves masked variables.
        """
        matrices_list = []
        primary_key = self.keys[functor_address]
        for mask_info in masks_list:
            tuple_mask_info = tuple(mask_info)
            functor_value_dict_key = (table_functor_value, functor, variable, tuple_mask_info)
            if mode == 'metric_observed':
                # Use cached matrix if available
                if functor_value_dict.get(functor_value_dict_key) is not None:
                    matrix = functor_value_dict[functor_value_dict_key]
                    matrices_list.append(matrix)
                    counter += 1
                    counter_c1 += 1
                    continue
                # Create a vector or matrix based on variable position
                if variable == mask_info[1]:
                    matrix = torch.zeros((self.matrices[mask_info[0]].shape[0], 1), device=self.device)
                else:
                    matrix = torch.zeros((1, self.matrices[mask_info[0]].shape[1]), device=self.device)
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
                # Use reconstructed data to create the matrix
                if variable == mask_info[1]:
                    matrix = self.compute_state_one_variable(
                         functor, table_functor_value, functor_address, reconstructed_x_slice, reconstructed_labels
                    )
                else:
                    matrix = self.compute_state_one_variable_transpose(
                         functor, table_functor_value, functor_address, reconstructed_x_slice, reconstructed_labels
                    )
                matrices_list.append(matrix)
        return matrices_list, functor_value_dict, counter, counter_c1


# Compute matrix when the variable matches the mask variable

    def compute_state_one_variable(self, functor, table_functor_value, functor_address, reconstructed_x_slice, reconstructed_labels):
        """
        Compute matrix for state 1 variable when the variable matches the mask variable.
        """
        # Search for functor in entity feature columns
        found = False
        indx = None
        entity_or_relation_key = None
        
        # First search in entity feature columns
        for key, feature_list in self.entity_feature_columns.items():
            if functor in feature_list:
                indx = feature_list.index(functor)
                entity_or_relation_key = key
                found = True
                break
        
        # If not found in entity features, search in relation feature columns
        if not found:
            for key, feature_list in self.relation_feature_columns.items():
                if functor in feature_list:
                    indx = feature_list.index(functor)
                    entity_or_relation_key = key
                    found = True
                    break
        
        # If found in feature columns, use the index
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

    def compute_state_one_variable_transpose(self, functor, table_functor_value, functor_address, reconstructed_x_slice, reconstructed_labels):
        """
        Compute matrix for state 1 variable when the variable does not match the mask variable.
        """
        # Search for functor in entity feature columns
        found = False
        indx = None
        entity_or_relation_key = None
        
        # First search in entity feature columns
        for key, feature_list in self.entity_feature_columns.items():
            if functor in feature_list:
                indx = feature_list.index(functor)
                entity_or_relation_key = key
                found = True
                break
        
        # If not found in entity features, search in relation feature columns
        if not found:
            for key, feature_list in self.relation_feature_columns.items():
                if functor in feature_list:
                    indx = feature_list.index(functor)
                    entity_or_relation_key = key
                    found = True
                    break
        
        # If found in feature columns, use the index
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

# Retrieve or invert the relation matrix for state 2
    def compute_state_two(self, functor, table_functor_value):
        """
        Retrieve or invert the relation matrix for state 2.
        """
        if table_functor_value == 'F':
            # Invert the matrix if the value is 'F' (false)
            matrix = 1 - self.matrices[functor]
        else:
            matrix = self.matrices[functor]
        return matrix



# Compute matrix for state 3 involving attribute relations
    def compute_state_three(self, reconstructed_labels, functor, table_functor_value, feature_info_mapping):
        """
        Compute matrix for state 3, involving attribute relations.
        """
        mode = False
        if mode == True:
            table_name = self.attributes[functor]
            primary_key = self.keys[table_name]
            if table_functor_value == 'N/A':
                # Invert the matrix if the value is 'N/A'
                matrix = 1 - self.matrices[table_name]
            else:
                # Create a matrix indicating where the attribute matches the value
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
            for idx, info in feature_info_mapping.items():
                if info['feature_name'] == functor:
                    feature_idx = idx
                    break

            target_tensor = reconstructed_labels[feature_idx]

            if table_functor_value == 'N/A':

                matrix = torch.sum(target_tensor, dim=0)
            else:
                value_mapping = feature_info_mapping[feature_idx]['value_index_mapping']
                

                reverse_mapping = {v: k for k, v in value_mapping.items()}
                
                value_idx = reverse_mapping[int(table_functor_value)]
                

                matrix = target_tensor[value_idx]

                
        return matrix

    def create_feature_info_mapping(self):
        """
        Creates feature info mapping for all edge features in all relations.
        Follows the same logic as adjacency_tensors_for_edge_features.
        
        Sets:
            self.feature_info_mapping: Dictionary mapping index to feature metadata
            self.num_nodes_graph: Total number of nodes in the graph
        """
        
        # Initialize dictionary to map index to feature information
        self.feature_info_mapping = {}
        
        # ===== Step 1: Determine total number of nodes =====
        # Iterate through all relations to find the maximum node ID
        num_nodes = 0
        for relation_name, relation_df in self.relations.items():
            # Get all columns from the dataframe
            all_columns = list(relation_df.columns)
            
            # First two columns are assumed to be node IDs (source, target)
            node_id_cols = all_columns[:2]
            
            # Find maximum node ID from both columns
            max_node = max(relation_df[node_id_cols[0]].max(), 
                        relation_df[node_id_cols[1]].max())
            
            # Update total number of nodes (max_id + 1 for 0-indexing)
            num_nodes = max(num_nodes, max_node + 1)
        
        # Store total number of nodes
        self.num_nodes_graph = num_nodes
        
        # ===== Step 2: Process each relation and its features =====
        feature_index = 0  # Global index counter across all relations
        
        for relation_name, relation_df in self.relations.items():
            # Get all column names
            all_columns = list(relation_df.columns)
            
            # First two columns are node IDs
            node_id_cols = all_columns[:2]
            
            # Remaining columns are edge features
            feature_columns = all_columns[2:]
            
            # ===== Step 3: Process each feature column =====
            for feature_col in feature_columns:
                # Get all unique values for this feature
                unique_values = sorted(relation_df[feature_col].unique())
                
                # Create mapping from tensor index to actual feature value
                # e.g., {0: 0, 1: 1} means tensor index 0 -> feature value 0
                value_index_mapping = {i: int(val) for i, val in enumerate(unique_values)}
                
                # Calculate what the tensor shape would be
                num_unique_values = len(unique_values)
                tensor_shape = [num_unique_values, num_nodes, num_nodes]
                
                # ===== Step 4: Store metadata for this feature =====
                self.feature_info_mapping[feature_index] = {
                    'relation_name': relation_name,          # Which relation this feature belongs to
                    'feature_name': feature_col,             # Name of the feature column
                    'value_index_mapping': value_index_mapping,  # Index -> value mapping
                    'node_id_columns': node_id_cols,         # Names of node ID columns
                    'tensor_shape': tensor_shape             # Shape of the tensor
                }
                
                # Increment global feature index
                feature_index += 1
        

# Apply masking to the unmasked matrices based on base and mask indices
    def compute_masked_matrices(self, unmasked_matrices, base_indices, mask_indices):
        """
        Apply masking to the unmasked matrices based on base and mask indices.
        """
        # Initialize masked matrices with base matrices
        masked_matrices = [unmasked_matrices[k] for k in base_indices]
        for k in mask_indices:
            # Apply element-wise multiplication for masking
            masked_matrices[k[0]] = torch.mul(masked_matrices[k[0]], unmasked_matrices[k[1]])
        return masked_matrices


# Sort the masked matrices based on the sort indices
    def compute_sorted_matrices(self, masked_matrices, sort_indices):
        """
        Sort the masked matrices based on the sort indices.
        """
        sorted_matrices = []
        for k in sort_indices:
            if k[0]:
                # Transpose the matrix if needed
                sorted_matrices.append(masked_matrices[k[1]].T)
            else:
                sorted_matrices.append(masked_matrices[k[1]])
        return sorted_matrices


# Stack matrices according to the stack indices for multiplication
    def compute_stacked_matrices(self, sorted_matrices, stack_indices):
        """
        Stack matrices according to the stack indices for multiplication.
        """
        stacked_matrices = sorted_matrices.copy()
        pop_counter = 0
        for k in stack_indices:
            for _ in range(k[1] - k[0] - pop_counter):
                # Multiply adjacent matrices
                stacked_matrices[k[0]] = torch.mm(stacked_matrices[k[0]], stacked_matrices[k[0] + 1])
                stacked_matrices.pop(k[0] + 1)
                pop_counter += 1
            # Element-wise multiplication with identity to preserve dimensions
            stacked_matrices[k[0]] = torch.mul(
                stacked_matrices[k[0]],
                torch.eye(len(stacked_matrices[k[0]]), device=self.device)
            )
        return stacked_matrices



# Compute the final result by multiplying all stacked matrices
    def compute_result(self, stacked_matrices):
        """
        Compute the final result by multiplying all stacked matrices.
        """
        result = stacked_matrices[0]
        for k in range(1, len(stacked_matrices)):
            result = torch.mm(result, stacked_matrices[k])
        return result


# Process reconstructed adjacency and feature matrices to update internal matrices
    def process_reconstructed_data(self, mapping_details, reconstructed_adjacency, reconstructed_x,
                                   important_feat_ids, reconstructed_edge_features):
        """
        Process reconstructed adjacency and feature matrices to update internal matrices.
        """
        if self.args.graph_type == 'heterogeneous':
            # Map edge types to node types
            edge_encoding_to_node_types = {v: k for k, v in mapping_details['edge_type_encoding'].items()}
            filtered_reconstruct_adj = []
            
            for idx, adj_matrix in enumerate(reconstructed_adjacency):
                # Get source and destination node types
                node_types = edge_encoding_to_node_types[idx + 1]  
                src_type, dst_type = node_types

                # Get indices for slicing
                src_start, src_end = mapping_details['node_type_to_index_map'][src_type]
                dst_start, dst_end = mapping_details['node_type_to_index_map'][dst_type]

                # Slice the adjacency matrix
                filtered_matrix = adj_matrix[src_start:src_end, dst_start:dst_end]
                filtered_reconstruct_adj.append(filtered_matrix)
            
            # Convert matrices to tensors and move to device
            filtered_reconstruct_adj_tensors = [matrix.to(self.device) for matrix in filtered_reconstruct_adj]
            
            for filtered_matrix in filtered_reconstruct_adj_tensors:
                filtered_shape = filtered_matrix.shape 
                
                for key, matrix in self.matrices.items():
                    # Update internal matrices with reconstructed ones
                    if matrix.shape == filtered_shape or matrix.shape == filtered_matrix.T.shape:
                        if filtered_matrix.size() == self.matrices[key].size():
                            self.matrices[key] = filtered_matrix.to(self.device)
                        elif filtered_matrix.T.size() == self.matrices[key].size():
                            self.matrices[key] = filtered_matrix.T.to(self.device)
                        break 
                    
            # Split reconstructed features based on node types
            reconstructed_x_splits = {}
            for node_type, (start_idx, end_idx) in mapping_details['node_type_to_index_map'].items():
                reconstructed_x_splits[f"{node_type}"] = reconstructed_x[start_idx:end_idx,:].to(self.device)
                
            # Identify node types that appear multiple times
            node_type_counts = {}
            for node_types in mapping_details['edge_type_encoding'].keys():
                for node_type in node_types:
                    if node_type in node_type_counts:
                        node_type_counts[node_type] += 1
                    else:
                        node_type_counts[node_type] = 1

            repeated_node_types = [node_type for node_type, count in node_type_counts.items() if count > 1]
            st_idx, en_idx = mapping_details['node_type_to_index_map'][repeated_node_types[0]]
            reconstructed_labels_m = reconstructed_edge_features[st_idx:en_idx].to(self.device)
                
        else:
            # For homogeneous graphs
            reconstructed_x_splits = torch.tensor(reconstructed_x).to(self.device)
            key = list(self.matrices.keys())[0]
            self.matrices[key] = reconstructed_adjacency[0].to(self.device)
            #reconstructed_labels_m = [t.to(self.device) for t in reconstructed_edge_features]
            reconstructed_labels_m = None
        return reconstructed_x_splits, reconstructed_labels_m



        # Update the internal matrices with the training adjacency matrices


#Update the internal matrices with the training adjacency matrices.
    def update_matrices(self, mapping_details, adj_with_self_loops):
        """
        Update the internal matrices with the training adjacency matrices.
        """
        for i in range(len(adj_with_self_loops)):
            adj_with_self_loops[i] = torch.tensor(adj_with_self_loops[i])

        for key, matrix in self.matrices.items():
            # Parse entity names from the key
            entities = key.split('_')
            entity1, entity2 = entities[0][:-1], entities[1][:-1] 

            transpose = False
            # Determine the relation number and whether to transpose
            if (entity1, entity2) in mapping_details['edge_type_encoding']:
                relation_number = mapping_details['edge_type_encoding'][(entity1, entity2)]
            elif (entity2, entity1) in mapping_details['edge_type_encoding']:
                relation_number = mapping_details['edge_type_encoding'][(entity2, entity1)]
                transpose = True  
            else:
                raise Exception(f"No relation found for key: {key}")

            relation_index = relation_number - 1

            # Get indices for slicing
            if transpose:
                start_idx1, end_idx1 = mapping_details['node_type_to_index_map'][entity2]
                start_idx2, end_idx2 = mapping_details['node_type_to_index_map'][entity1]
            else:
                start_idx1, end_idx1 = mapping_details['node_type_to_index_map'][entity1]
                start_idx2, end_idx2 = mapping_details['node_type_to_index_map'][entity2]

            # Slice the adjacency matrix
            sliced_matrix = adj_with_self_loops[relation_index][start_idx1:end_idx1, start_idx2:end_idx2]
            if transpose:
                sliced_matrix = sliced_matrix.T 
            
            # Update the internal matrix
            self.matrices[key] = sliced_matrix.to(self.device)