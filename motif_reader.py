# motif_reader.py

from pymysql import connect
from pymysql.err import OperationalError, MySQLError
from pandas import DataFrame
from itertools import permutations
from math import log
from typing import Dict, Tuple, Optional
import torch
from pathlib import Path

from motif_store import RuleBasedMotifStore


class DatabaseMotifReader:
    """
    Reads motif rules from a relational database OR pickle file.
    
    Two modes:
    1. Load from PKL: Fast loading from pre-saved pickle file
    2. Connect to DB: Read from MySQL database and optionally save to PKL
    """
    
    def __init__(self, dataset_name: str, args, host='localhost', user='fbuser', password=''):
        """
        Initialize the database motif reader.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'Cora_dgl', 'citeseer')
            args: Arguments object containing configuration
            host: Database host
            user: Database user
            password: Database password
        """
        self.dataset_name = dataset_name
        self.args = args
        self.host = host
        self.user = user
        self.password = password
        
        # Database name mapping
        self.database_mapping = {
            "Cora_dgl": "cora",
            "citeseer": "citeseer",
            "imdb-multi": "imdb",
            "acm-multi": "acm-multi",
            "CiteSeer_dgl": "citeseer",
            "IMDB": "imdb",
            "grid": "grid"
        }
        
        self.db_name = self.database_mapping.get(dataset_name, dataset_name)
    
    def read(self, mode: str, pkl_path: Optional[str] = None) -> RuleBasedMotifStore:
        """
        Main entry point: reads motif data based on mode.
        
        Args:
            mode: 'pkl' to load from pickle, 'db' to read from database
            pkl_path: Path to pickle file (required for 'pkl' mode, optional for 'db' mode to save)
            
        Returns:
            Populated MotifStore with all rules, entities, relations, and matrices.
        """
        if mode == 'pkl':
            # Mode 1: Load from pickle file
            if pkl_path is None:
                raise ValueError("pkl_path must be provided when mode='pkl'")
            return self._load_from_pickle(pkl_path)
        
        elif mode == 'db':
            # Mode 2: Read from database
            store = self._read_from_database()
            
            # Optionally save to pickle after reading from DB
            if pkl_path is not None:
                store.save(pkl_path)
            
            return store
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'pkl' or 'db'")
    
    def _load_from_pickle(self, pkl_path: str) -> RuleBasedMotifStore:
        """
        Load motif store from a pickle file.
        
        Args:
            pkl_path: Path to the .pkl file
            
        Returns:
            Loaded RuleBasedMotifStore
        """
        device = self.args.device if hasattr(self.args, 'device') else 'cuda'
        
        try:
            store = RuleBasedMotifStore.load(pkl_path, device=device)
            store.args = self.args  # Attach current args
            return store
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Pickle file not found: {pkl_path}\n"
                f"Please check the path or use --connect_db True to read from database."
            )
        except Exception as e:
            raise RuntimeError(f"Error loading pickle file: {e}")
    
    def _read_from_database(self) -> RuleBasedMotifStore:
        """
        Read motif store from database.
        
        Returns:
            Populated RuleBasedMotifStore
        """
        print(f"🗄️  Connecting to database: {self.db_name}")
        
        store = RuleBasedMotifStore()
        store.args = self.args
        
        try:
            # Connect to databases
            connections = self._connect_to_databases()
            
            try:
                # Fetch all data from database
                print("  Reading entities...")
                self._fetch_entities(connections['main'], connections['setup'], store)
                
                print("  Reading relations...")
                self._fetch_relations(connections['main'], connections['setup'], store)
                
                print("  Reading attributes...")
                self._fetch_attributes(connections['setup'], store)
                
                print("  Creating indices...")
                self._create_indices(store)
                
                print("  Creating mask matrices...")
                self._create_mask_matrices(connections['setup'], store)
                
                print("  Processing Bayesian Network rules...")
                self._process_rules(connections['bn'], connections['setup'], store)
                
                print("  Creating feature mappings...")
                self._create_feature_info_mapping(store)
                
                print(f"✓ Successfully read all data from database")
                
            finally:
                # Always close connections
                self._close_connections(connections)
            
            return store
            
        except (OperationalError, MySQLError) as e:
            error_msg = (
                f"✗ Database connection failed: {e}\n"
                f"  Please ensure:\n"
                f"    1. MySQL is running\n"
                f"    2. Database '{self.db_name}' exists\n"
                f"    3. Database credentials are correct\n"
                f"  Alternatively, use --use_pkl True to load from pickle file."
            )
            raise RuntimeError(error_msg)
        except Exception as e:
            raise RuntimeError(f"Error reading from database: {e}")
    
    def _connect_to_databases(self) -> Dict:
        """Establish connections to main, setup, and Bayesian Network databases."""
        connections = {}
        
        try:
            # Main database connection
            conn_main = connect(host=self.host, user=self.user, password=self.password, db=self.db_name)
            connections['main'] = {'connection': conn_main, 'cursor': conn_main.cursor()}
            
            # Setup database connection
            db_setup = f"{self.db_name}_setup"
            conn_setup = connect(host=self.host, user=self.user, password=self.password, db=db_setup)
            connections['setup'] = {'connection': conn_setup, 'cursor': conn_setup.cursor()}
            
            # Bayesian Network database connection
            db_bn = f"{self.db_name}_BN"
            conn_bn = connect(host=self.host, user=self.user, password=self.password, db=db_bn)
            connections['bn'] = {'connection': conn_bn, 'cursor': conn_bn.cursor()}
            
        except OperationalError as e:
            # Close any connections that were opened
            for conn_dict in connections.values():
                try:
                    conn_dict['cursor'].close()
                    conn_dict['connection'].close()
                except:
                    pass
            raise
        
        return connections
    
    def _fetch_entities(self, main_conn, setup_conn, store: RuleBasedMotifStore):
        """Fetch entity tables and their primary keys."""
        cursor_main = main_conn['cursor']
        cursor_setup = setup_conn['cursor']
        
        cursor_setup.execute("SELECT TABLE_NAME FROM EntityTables")
        entity_tables = cursor_setup.fetchall()
        
        for (table_name,) in entity_tables:
            cursor_main.execute(f"SELECT * FROM {table_name}")
            rows = cursor_main.fetchall()
            
            cursor_main.execute(f"SHOW COLUMNS FROM {self.db_name}.{table_name}")
            columns = cursor_main.fetchall()
            columns_names = [column[0] for column in columns]
            
            store.entities[table_name] = DataFrame(rows, columns=columns_names)
            store.entity_feature_columns[table_name] = columns_names[1:]
            
            cursor_setup.execute("SELECT COLUMN_NAME FROM EntityTables WHERE TABLE_NAME = %s", (table_name,))
            key = cursor_setup.fetchall()
            store.keys[table_name] = key[0][0]
    
    def _fetch_relations(self, main_conn, setup_conn, store: RuleBasedMotifStore):
        """Fetch relation tables and their foreign keys."""
        cursor_main = main_conn['cursor']
        cursor_setup = setup_conn['cursor']
        
        cursor_setup.execute("SELECT TABLE_NAME FROM RelationTables")
        relation_tables = cursor_setup.fetchall()
        
        for (table_name,) in relation_tables:
            cursor_main.execute(f"SELECT * FROM {table_name}")
            rows = cursor_main.fetchall()
            
            cursor_main.execute(f"SHOW COLUMNS FROM {self.db_name}.{table_name}")
            columns = cursor_main.fetchall()
            columns_names = [column[0] for column in columns]
            
            store.relations[table_name] = DataFrame(rows, columns=columns_names)
            store.relation_feature_columns[table_name] = columns_names[2:]
            
            cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
            key = cursor_setup.fetchall()
            store.keys[table_name] = (key[0][0], key[1][0])
    
    def _fetch_attributes(self, setup_conn, store: RuleBasedMotifStore):
        """Fetch attribute columns."""
        cursor_setup = setup_conn['cursor']
        cursor_setup.execute("SELECT COLUMN_NAME, TABLE_NAME FROM AttributeColumns")
        attribute_columns = cursor_setup.fetchall()
        
        for column_name, table_name in attribute_columns:
            store.attributes[column_name] = table_name
    
    def _create_indices(self, store: RuleBasedMotifStore):
        """Create indices for quick lookup of entity keys."""
        for table_name, df in store.entities.items():
            key = store.keys[table_name]
            store.indices[key] = {row[key]: idx for idx, row in df.iterrows()}
    
    def _create_mask_matrices(self, setup_conn, store: RuleBasedMotifStore):
        """Create mask matrices representing relations between entities."""
        cursor_setup = setup_conn['cursor']
        
        for table_name, df in store.relations.items():
            cursor_setup.execute("SELECT REFERENCED_TABLE_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
            reference = cursor_setup.fetchall()
            entity1 = reference[0][0]
            entity2 = reference[1][0]
            
            shape = (len(store.entities[entity1].index), len(store.entities[entity2].index))
            store.matrices[table_name] = torch.zeros(shape, dtype=torch.float32, device=store.device)
        
        for table_name, df in store.relations.items():
            cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
            key = cursor_setup.fetchall()
            cursor_setup.execute("SELECT COLUMN_NAME, REFERENCED_COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
            reference = cursor_setup.fetchall()
            
            rows_indices = []
            cols_indices = []
            for index, row in df.iterrows():
                row_index = store.indices[reference[0][1]][row[key[0][0]]]
                col_index = store.indices[reference[1][1]][row[key[1][0]]]
                rows_indices.append(row_index)
                cols_indices.append(col_index)
            
            rows_indices_tensor = torch.tensor(rows_indices, dtype=torch.long)
            cols_indices_tensor = torch.tensor(cols_indices, dtype=torch.long)
            store.matrices[table_name][rows_indices_tensor, cols_indices_tensor] = 1
    
    def _process_rules(self, bn_conn, setup_conn, store: RuleBasedMotifStore):
        """Process rules from the Bayesian Network and prepare for counting."""
        cursor_bn = bn_conn['cursor']
        cursor_setup = setup_conn['cursor']
        
        cursor_bn.execute("SELECT DISTINCT child FROM Final_Path_BayesNets_view")
        childs = cursor_bn.fetchall()
        
        relation_names = tuple(store.relations.keys())
        
        for i in range(len(childs)):
            rule = [childs[i][0]]
            cursor_bn.execute("SELECT parent FROM Final_Path_BayesNets_view WHERE child = %s", (childs[i][0],))
            parents = cursor_bn.fetchall()
            for (parent,) in parents:
                if parent != '':
                    rule.append(parent)
            
            store.rules.append(rule)
            store.multiples.append(1 if len(rule) > 1 else 0)
            
            relation_check = any(',' in atom for atom in rule)
            functor, variable, node, state, mask = {}, {}, {}, [], {}
            unmasked_variables = []
            
            for j in range(len(rule)):
                fun = rule[j].split('(')[0]
                functor[j] = fun
                
                if ',' not in rule[j]:
                    var = rule[j].split('(')[1][:-1]
                    variable[j] = var
                    node[j] = var[:-1]
                    
                    if not relation_check:
                        unmasked_variables.append(var)
                        state.append(0)
                    else:
                        mas = []
                        for k in rule:
                            func = k.split('(')[0]
                            if func not in relation_names:
                                func = store.attributes.get(func, func)
                            if ',' in k and var in k:
                                var1, var2 = k.split('(')[1][:-1].split(',')
                                mas.append([func, var1, var2])
                                unmasked_variables.append(k.split('(')[1][:-1])
                        mask[j] = mas
                        state.append(1)
                else:
                    unmasked_variables.append(rule[j].split('(')[1][:-1])
                    if fun in relation_names:
                        state.append(2)
                    else:
                        state.append(3)
            
            store.functors[i] = functor
            store.variables[i] = variable
            store.nodes[i] = node
            store.states.append(state)
            store.masks[i] = mask
            
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
            
            sort_indice, sorted_variables = self._create_sort_indices(masked_variables, relation_check, relation_names)
            stack_indice = self._create_stack_indices(sorted_variables)
            
            store.base_indices.append(base_indice)
            store.mask_indices.append(mask_indice)
            store.sort_indices.append(sort_indice)
            store.stack_indices.append(stack_indice)
            
            cursor_bn.execute(f"SELECT * FROM `{childs[i][0]}_CP`")
            value = cursor_bn.fetchall()
            
            if self.args.rule_prune and not self.args.rule_weight:
                pruned_value = []
                for j in value:
                    size = len(j)
                    if store.multiples[i]:
                        if 2 * j[size - 4] * (log(j[size - 3]) - log(j[size - 1])) - log(j[size - 4]) > 0:
                            pruned_value.append(j)
                    else:
                        if 2 * int(j[size - 3]) * (log(j[size - 5]) - log(j[size - 1])) - log(int(j[size - 3])) > 0:
                            pruned_value.append(j)
                store.values.append(pruned_value)
            elif self.args.rule_prune and self.args.rule_weight:
                pruned_value = []
                prune = []
                for j in value:
                    size = len(j)
                    if store.multiples[i]:
                        p = 2 * j[size - 4] * (log(j[size - 3]) - log(j[size - 1])) - log(j[size - 4])
                        if p > 0:
                            pruned_value.append(j)
                            prune.append(p)
                    else:
                        p = 2 * int(j[size - 3]) * (log(j[size - 5]) - log(j[size - 1])) - log(int(j[size - 3]))
                        if p > 0:
                            pruned_value.append(j)
                            prune.append(p)
                store.prunes.append(prune)
                store.values.append(pruned_value)
            elif not self.args.rule_prune and self.args.rule_weight:
                raise Exception('Rule weighting requires rule pruning to be enabled.')
            else:
                store.values.append(value)
        
        self._adjust_matrices(store)
    
    def _create_sort_indices(self, masked_variables, relation_check, relation_names):
        """Create indices to sort variables for matrix multiplication chain."""
        sort_indice = []
        sorted_variables = []
        
        if not relation_check:
            sort_indice.append([False, 0])
            sorted_variables.append(masked_variables[0])
        else:
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
        
        return sort_indice, sorted_variables
    
    def _create_stack_indices(self, sorted_variables):
        """Create indices for stacking matrices in correct order."""
        stack_indices = []
        for j in range(1, len(sorted_variables)):
            second = sorted_variables[j].split(',')[1]
            for k in range(j - 1, -1, -1):
                previous_first = sorted_variables[k].split(',')[0]
                if previous_first == second:
                    stack_indices.append([k, j])
        return stack_indices
    
    def _adjust_matrices(self, store: RuleBasedMotifStore):
        """Adjust matrices to correct shape by transposing if necessary."""
        relation_functors = [item for sublist in store.rules for item in sublist 
                           if ',' in item and item in store.relations.keys()]
        unique_relation_functors = list(set(relation_functors))
        
        for relation_functor in unique_relation_functors:
            entities_involved = relation_functor.replace(')', '').split('(')[1].split(',')
            entities_clean = [entity[:-1] for entity in entities_involved]
            correct_shape = (len(store.entities[entities_clean[0]]), len(store.entities[entities_clean[1]]))
            matrix_name = relation_functor.split('(')[0]
            
            if store.matrices[matrix_name].shape != correct_shape:
                store.matrices[matrix_name] = store.matrices[matrix_name].t()
    
    def _create_feature_info_mapping(self, store: RuleBasedMotifStore):
        """Create feature info mapping for all edge features in all relations."""
        num_nodes = 0
        for relation_name, relation_df in store.relations.items():
            all_columns = list(relation_df.columns)
            node_id_cols = all_columns[:2]
            max_node = max(relation_df[node_id_cols[0]].max(), relation_df[node_id_cols[1]].max())
            num_nodes = max(num_nodes, max_node + 1)
        
        store.num_nodes_graph = num_nodes
        
        feature_index = 0
        for relation_name, relation_df in store.relations.items():
            all_columns = list(relation_df.columns)
            node_id_cols = all_columns[:2]
            feature_columns = all_columns[2:]
            
            for feature_col in feature_columns:
                unique_values = sorted(relation_df[feature_col].unique())
                value_index_mapping = {i: int(val) for i, val in enumerate(unique_values)}
                num_unique_values = len(unique_values)
                tensor_shape = [num_unique_values, num_nodes, num_nodes]
                
                store.feature_info_mapping[feature_index] = {
                    'relation_name': relation_name,
                    'feature_name': feature_col,
                    'value_index_mapping': value_index_mapping,
                    'node_id_columns': node_id_cols,
                    'tensor_shape': tensor_shape
                }
                feature_index += 1
    
    def _close_connections(self, connections: Dict):
        """Close all database connections."""
        for conn_dict in connections.values():
            try:
                conn_dict['cursor'].close()
                conn_dict['connection'].close()
            except:
                pass