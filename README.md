 # What Each File Does - Deep Dive


 ## 1. main.py - The Conductor.



Role: Coordinates the entire pipeline from start to finish

What it does:

Parses arguments - Reads command-line inputs to determine:

Should I load from pickle or database?
Which dataset to use?
Where to save/load files?


Validates configuration - Checks that arguments make sense:

Can't use both --use_pkl and --connect_db
If using pickle, must provide path
If saving after DB, must provide path


the pipeline in order:
### Load Graph → Reduce Features → Read Motifs → Create Counter → Count Motifs



## 2. data.py - The Graph Warehouse
Role: Loads raw graph data and prepares it for processing
What it does:

Connects to DGL - Uses Deep Graph Library to load Cora dataset
Extracts components:

Nodes: 2708 papers
Edges: 10556 citations (who cites whom)
Features: 1433-dimensional bag-of-words vectors (word counts in each paper)
Labels: 7 categories (Neural_Networks, Rule_Learning, etc.)
Masks: Train/validation/test splits


Binarizes features - Converts word counts to 0/1 (present/absent)
Creates adjacency matrix - Square matrix showing which papers cite which
Returns everything in a clean dictionary


Key Class:

DataLoader - Main class that loads and stores all graph data

Key Methods:

load_data() - Does the actual loading from DGL
get_data() - Returns a dictionary with all data


## 3. ds_to_db.py - The Feature Compressor
Role: Reduces high-dimensional features to essential ones
What it does:

Takes 1433 features (original bag-of-words)
Uses ExtraTreesClassifier to rank feature importance

Trains on node labels to see which features predict categories best


Selects top 5 features - Keeps only most discriminative features
Returns reduced features - Now only 5 dimensions instead of 1433



## 4. motif_store.py - The Data Container
Role: A "box" that holds ALL motif-related data
What it stores:
Rule Data:

rules - List of motif patterns, e.g., ['feature_1(x1)', 'citations(x1,x2)', 'feature_2(x2)']


functors - Function names in each rule


variables - Variable names (x1, x2, etc.)


states - Type of each atom (0=unary, 1=masked, 2=relation, 3=attribute)


values - Conditional probabilities from Bayesian Network

Database Data:

entities - Node tables (e.g., "papers" table)


relations - Edge tables (e.g., "citations" table)


keys - Primary keys for each table


attributes - Attribute columns

Computation Data:

matrices - Adjacency matrices for each relation (as PyTorch tensors)

indices - Lookup dictionaries for fast access

base_indices, mask_indices - Guides for matrix operations

sort_indices, stack_indices - Order for matrix multiplication

Feature Mappings:

entity_feature_columns - Which columns are features in entity tables

relation_feature_columns - Which columns are features in relation tables

feature_info_mapping - Detailed metadata about edge features


Key Methods:

save(file_path) - Saves everything to a pickle file

load(file_path) - Loads everything from a pickle file

to_device(device) - Moves all tensors to GPU/CPU


## 5. motif_reader.py - The Database Librarian
Role: Reads data from MySQL databases and fills the MotifStore

What it does:

Mode 1: Load from Pickle (Simple)

Opens the .pkl file

Deserializes all data

Creates a RuleBasedMotifStore object

Loads matrices to GPU/CPU

Returns filled store

Mode 2: Connect to Database (Complex)

Phase 1: Connect

Opens 3 database connections:

cora - Main data (papers table, citations table)

cora_setup - Metadata (what columns are keys, what are attributes)

cora_BN - Bayesian Network rules (motif patterns)

and so on ...



## 6. motif_counter.py - The Counting Engine
Role: Performs the actual motif counting using matrix operations
What it does:
High-Level Process:
```python
for each rule in rules:
    for each value combination in conditional_probability_table:
        # 1. Create matrices for each atom in rule
        unmasked_matrices = compute_unmasked_matrices()
        
        # 2. Apply constraints (masking)
        masked_matrices = compute_masked_matrices()
        
        # 3. Sort for multiplication
        sorted_matrices = compute_sorted_matrices()
        
        # 4. Multiply matrices together
        stacked_matrices = compute_stacked_matrices()
        
        # 5. Compute final result
        result = matrix_multiply_all(stacked_matrices)
        
        # 6. Sum to get count
        count = sum(result)
        
        motif_counts.append(count)
```
Detailed: Matrix Computation for Each State
State 0: Unary Predicates (Node Features)
python# Example: feature_1(x1) where feature_1=0
# Creates column vector: [1 if node has feature_1=0, else 0]

matrix = [
    [1],  # Node 0: has feature_1=0 ✓
    [0],  # Node 1: doesn't have feature_1=0
    [1],  # Node 2: has feature_1=0 ✓
    ...
]
State 1: Masked Variables (Connected Nodes)
python# Example: feature_1(x1) where x1 is connected via citations(x1,x2)
# Creates column or row vector depending on position

if variable_position == 'source':
    matrix = [[1], [0], [1], ...]  # Column vector
else:
    matrix = [[1, 0, 1, ...]]      # Row vector
State 2: Relations (Adjacency Matrices)
python# Example: citations(x1,x2)
# Uses pre-built adjacency matrix

matrix = [
    [0, 1, 0, 1, ...],  # Node 0 cites nodes 1 and 3
    [1, 0, 0, 0, ...],  # Node 1 cites node 0
    ...
]
State 3: Edge Features (Attribute Relations)
python# Example: citation_type(x1,x2) = 'direct'
# Creates matrix where entry is 1 if edge has attribute

matrix = [
    [0, 1, 0, 0, ...],  # Edge 0→1 is 'direct'
    [0, 0, 0, 1, ...],  # Edge 0→3 is 'direct'
    ...
]
Example: Complete Counting Process
Rule: "Papers in category 1 citing papers in category 2"

label(x1)=1, citations(x1,x2), label(x2)=2

Step 1: Compute Unmasked Matrices
pythonM1 = [1, 0, 1, 0, ...]  # Nodes with label=1 (column vector)
M2 = [[0,1,0,...],       # Citations adjacency
      [1,0,0,...],
      ...]
M3 = [0, 1, 0, 1, ...]  # Nodes with label=2 (column vector)
Step 2: Apply Masking
python# No masking needed in this simple example
masked = [M1, M2, M3]
Step 3: Sort for Multiplication
python# Transpose M1 to make it a row vector for multiplication
sorted = [M1.T, M2, M3]
# Shape: (1×2708) × (2708×2708) × (2708×1)
Step 4: Stack and Multiply
pythonresult = M1.T @ M2 @ M3
# Result shape: (1×1) - single number
Step 5: Extract Count
pythoncount = result[0][0] = 542
Interpretation: There are 542 instances where a paper in category 1 cites a paper in category 2.
Key Methods:

count(graph_data) - Main entry point

_iteration_function() - Loops through all rules

_compute_unmasked_matrices() - Creates initial matrices

_compute_state_zero() - Handles unary predicates

_compute_state_one() - Handles masked variables

_compute_state_two() - Handles relations

_compute_state_three() - Handles edge attributes

_compute_masked_matrices() - Applies constraints

_compute_sorted_matrices() - Arranges for multiplication

_compute_stacked_matrices() - Multiplies matrices

_compute_result() - Final multiplication

## 7. motif_dataset.py - The Dataset Wrapper
Role: Wraps graph data and adds motif counting capability

What it does:

Stores graph data - Holds adjacency, features, labels
Lazy evaluation - Only counts motifs when you ask for them
Caching - Remembers counts so it doesn't recompute
Provides interface - Easy access to augmented data

Key Class:

```python

class MotifAugmentedDataset:
    def __init__(self, base_data, motif_counter):
        self.base_data = base_data
        self.motif_counter = motif_counter
        self._motif_counts = None  # Cache
    
    @property
    def motif_counts(self):
        # Only compute once
        if self._motif_counts is None:
            self._motif_counts = self.motif_counter.count(self.base_data)
        return self._motif_counts
    
    def get_augmented_data(self):
        # Returns original data + motif counts
        return {
            **self.base_data,
            'motif_counts': self.motif_counts
        }

```
```
# Create dataset
dataset = MotifAugmentedDataset(graph_data, counter)

# First access - computes motifs
counts = dataset.motif_counts  # Takes time

# Second access - uses cache
counts = dataset.motif_counts  # Instant!

# Get everything
data = dataset.get_augmented_data()
# data = {
#     'adjacency_matrix': ...,
#     'features': ...,
#     'labels': ...,
#     'motif_counts': [542, 401, ...]  # NEW!
# }
```

**Why it exists:**
- **Standard pattern** - Like PyTorch Dataset classes
- **Lazy evaluation** - Saves computation
- **Caching** - Avoids redundant work
- **Clean interface** - Simple to use

---

## 🏛️ Complete Code Schema

### **Visual Architecture**
```
┌────────────────────────────────────────────────────────────────────────┐
│                         FULL SYSTEM ARCHITECTURE                        │
└────────────────────────────────────────────────────────────────────────┘

                              main.py
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼────┐   ┌──▼────┐   ┌──▼──────────┐
              │ data.py  │   │ args  │   │ ds_to_db.py │
              │DataLoader│   │parsing│   │  Feature    │
              └─────┬────┘   └───────┘   │  Reducer    │
                    │                    └──┬──────────┘
                    │                       │
                    │     ┌─────────────────┘
                    │     │
              ┌─────▼─────▼──────┐
              │   Graph Data      │
              │  ┌─────────────┐ │
              │  │ Adjacency   │ │
              │  │ Features    │ │
              │  │ Labels      │ │
              │  └─────────────┘ │
              └──────┬────────────┘
                     │
                     │
        ┌────────────▼───────────────┐
        │  motif_reader.py           │
        │  DatabaseMotifReader       │
        │                            │
        │  ┌──────────┐ ┌──────────┐│
        │  │ PKL Mode │ │ DB Mode  ││
        │  │          │ │          ││
        │  │  Load    │ │ Connect  ││
        │  │  .pkl    │ │ MySQL    ││
        │  │  file    │ │ databases││
        │  └────┬─────┘ └────┬─────┘│
        │       │            │      │
        │       └──────┬─────┘      │
        │              │            │
        │       ┌──────▼────────┐   │
        │       │ motif_store.py│   │
        │       │MotifStore     │   │
        └───────┴───────┬───────┴───┘
                        │
                        │
            ┌───────────▼──────────────┐
            │  RuleBasedMotifStore     │
            │  ┌────────────────────┐  │
            │  │ Rules              │  │
            │  │ Matrices           │  │
            │  │ Indices            │  │
            │  │ Functors           │  │
            │  │ Variables          │  │
            │  │ States             │  │
            │  │ Feature Mappings   │  │
            │  └────────────────────┘  │
            └───────────┬──────────────┘
                        │
                        │
               ┌────────▼─────────┐
               │ motif_counter.py │
               │RelationalMotif   │
               │     Counter      │
               │                  │
               │  Takes:          │
               │   - MotifStore   │
               │   - Graph Data   │
               │                  │
               │  Does:           │
               │   For each rule: │
               │     1. Create    │
               │        matrices  │
               │     2. Apply     │
               │        masks     │
               │     3. Sort      │
               │     4. Multiply  │
               │     5. Sum       │
               └────────┬─────────┘
                        │
                        │
              ┌─────────▼──────────┐
              │ motif_dataset.py   │
              │ MotifAugmented     │
              │      Dataset       │
              │                    │
              │  Wraps graph data  │
              │  Adds motif counts │
              │  Caches results    │
              └─────────┬──────────┘
                        │
                        │
                   ┌────▼─────┐
                   │  OUTPUT  │
                   │          │
                   │ Motif    │
                   │ Count    │
                   │ Vector   │
                   └──────────┘
```

### **Data Flow Diagram**
```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                 │
└─────────────────────────────────────────────────────────────────┘

Input: Command Line Arguments
  │
  ├─► --use_pkl True --pkl_path "file.pkl"
  │   OR
  └─► --connect_db True --save_pkl_after_db True
      │
      ▼
┌─────────────────────┐
│ main.py             │
│ parse_arguments()   │
│ validate_args()     │
└──────┬──────────────┘
       │
       ├──► Step 1: Load Graph
       │    ┌─────────────────────┐
       │    │ data.py             │
       │    │ DataLoader()        │
       │    │   .load_data()      │
       │    └──────┬──────────────┘
       │           │
       │           ▼
       │    Graph Data Dictionary:
       │    {
       │      adjacency: [2708×2708],
       │      features: [2708×1433],
       │      labels: [2708],
       │      edges: [10556×2]
       │    }
       │           │
       ├──────────►│
       │           │
       ├──► Step 2: Reduce Features
       │    ┌─────▼──────────────────┐
       │    │ ds_to_db.py            │
       │    │ reduce_node_features() │
       │    └──────┬─────────────────┘
       │           │
       │           ▼
       │    Reduced Features:
       │    [2708×5] + labels
       │    = [2708×6]
       │           │
       ├──────────►│
       │           │
       ├──► Step 3: Read Motif Rules
       │    ┌─────▼───────────────────┐
       │    │ motif_reader.py         │
       │    │ DatabaseMotifReader     │
       │    │                         │
       │    │ if mode == 'pkl':       │
       │    │   load from .pkl        │
       │    │ else:                   │
       │    │   connect to MySQL ───┐ │
       │    │   read entities       │ │
       │    │   read relations      │ │
       │    │   process BN rules    │ │
       │    │   create matrices     │ │
       │    │   save to .pkl ◄──────┘ │
       │    └──────┬──────────────────┘
       │           │
       │           ▼
       │    ┌──────────────────────┐
       │    │ motif_store.py       │
       │    │ RuleBasedMotifStore  │
       │    │                      │
       │    │ - rules[147]         │
       │    │ - matrices{}         │
       │    │ - indices{}          │
       │    │ - functors{}         │
       │    │ - variables{}        │
       │    │ - states[]           │
       │    │ - base_indices[]     │
       │    │ - mask_indices[]     │
       │    │ - sort_indices[]     │
       │    │ - stack_indices[]    │
       │    └──────┬───────────────┘
       │           │
       ├──────────►│
       │           │
       ├──► Step 4: Create Counter
       │    ┌─────▼──────────────────┐
       │    │ motif_counter.py       │
       │    │ RelationalMotifCounter │
       │    │   __init__(store)      │
       │    └──────┬─────────────────┘
       │           │
       ├──────────►│
       │           │
       ├──► Step 5: Augment Dataset
       │    ┌─────▼───────────────────┐
       │    │ motif_dataset.py        │
       │    │ MotifAugmentedDataset   │
       │    │   __init__(data,counter)│
       │    └──────┬──────────────────┘
       │           │
       ├──────────►│
       │           │
       └──► Step 6: Count Motifs
            ┌─────▼─────────────────────┐
            │ motif_counter.py          │
            │ .count(graph_data)        │
            │                           │
            │ For each rule (147 total):│
            │   ┌───────────────────┐   │
            │   │ Compute Unmasked  │   │
            │   │   M1, M2, M3,...  │   │
            │   ├───────────────────┤   │
            │   │ Apply Masking     │   │
            │   │   M1 × M2         │   │
            │   ├───────────────────┤   │
            │   │ Sort Matrices     │   │
            │   │   Transpose if    │   │
            │   │   needed          │   │
            │   ├───────────────────┤   │
            │   │ Stack & Multiply  │   │
            │   │   M1×M2×M3×...    │   │
            │   ├───────────────────┤   │
            │   │ Sum Result        │   │
            │   │   count = Σ(M)    │   │
            │   └───────────────────┘   │
            │                           │
            └──────┬────────────────────┘
                   │
                   ▼
            ┌──────────────┐
            │   OUTPUT     │
            │              │
            │ motif_counts │
            │ [542, 401,   │
            │  1023, ...]  │
            │              │
            │ 147 counts   │
            └──────────────┘
```

### **Class Hierarchy**
```
┌─────────────────────────────────────────────┐
│           CLASS RELATIONSHIPS                │
└─────────────────────────────────────────────┘

main.py
  └── main()
      ├── creates DataLoader
      ├── creates DatabaseMotifReader
      ├── creates RelationalMotifCounter
      └── creates MotifAugmentedDataset


data.py
  └── DataLoader
      ├── __init__()
      ├── load_data()
      ├── _create_adjacency_matrix()
      └── get_data() → Dict


motif_store.py
  └── RuleBasedMotifStore
      ├── __init__()
      ├── save(file_path)
      ├── load(file_path) [classmethod]
      ├── to_device(device)
      └── num_motifs [property]


motif_reader.py
  └── DatabaseMotifReader
      ├── __init__(dataset_name, args)
      ├── read(mode, pkl_path) → RuleBasedMotifStore
      │   ├── _load_from_pickle()
      │   └── _read_from_database()
      │       ├── _connect_to_databases()
      │       ├── _fetch_entities()
      │       ├── _fetch_relations()
      │       ├── _fetch_attributes()
      │       ├── _create_indices()
      │       ├── _create_mask_matrices()
      │       ├── _process_rules()
      │       │   ├── _create_sort_indices()
      │       │   └── _create_stack_indices()
      │       ├── _adjust_matrices()
      │       ├── _create_feature_info_mapping()
      │       └── _close_connections()


motif_counter.py
  └── RelationalMotifCounter
      ├── __init__(motif_store)
      ├── count(graph_data) → List[float]
      │   ├── _process_graph_data()
      │   └── _iteration_function()
      │       ├── _compute_unmasked_matrices()
      │       │   ├── _compute_state_zero()
      │       │   ├── _compute_state_one()
      │       │   │   ├── _compute_state_one_variable()
      │       │   │   └── _compute_state_one_variable_transpose()
      │       │   ├── _compute_state_two()
      │       │   └── _compute_state_three()
      │       ├── _compute_masked_matrices()
      │       ├── _compute_sorted_matrices()
      │       ├── _compute_stacked_matrices()
      │       └── _compute_result()


motif_dataset.py
  └── MotifAugmentedDataset
      ├── __init__(base_data, motif_counter)
      ├── motif_counts [property]
      ├── get_augmented_data()
      └── get_motif_vector()


ds_to_db.py
  └── reduce_node_features(x, y, seed, n_components)
```

### **State Diagram: Motif Counting**
```
┌──────────────────────────────────────────────────┐
│         MOTIF COUNTING STATE MACHINE              │
└──────────────────────────────────────────────────┘

START
  │
  ▼
┌────────────────────┐
│ For each RULE      │
│ (147 iterations)   │
└──────┬─────────────┘
       │
       ▼
┌──────────────────────────┐
│ For each VALUE in CP     │
│ (conditional probability)│
└──────┬───────────────────┘
       │
       ▼
┌────────────────────────────────┐
│ For each ATOM in rule          │
│ (e.g., feature_1(x1))          │
└──────┬─────────────────────────┘
       │
       ▼
┌──────────────────┐
│ What STATE?      │
│                  │
│ ┌──────────────┐ │
│ │ State 0?     │─┼─► Unary predicate
│ │ State 1?     │─┼─► Masked variable
│ │ State 2?     │─┼─► Relation
│ │ State 3?     │─┼─► Edge attribute
│ └──────────────┘ │
└────┬─────────────┘
     │
     ▼
┌─────────────────────┐
│ CREATE MATRIX       │
│                     │
│ State 0 → Column    │
│ State 1 → Row/Col   │
│ State 2 → Adjacency │
│ State 3 → Attribute │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│ COLLECT MATRICES    │
│ [M1, M2, M3, ...]   │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│ APPLY MASKING       │
│ M1 = M1 * M2        │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│ SORT for            │
│ MULTIPLICATION      │
│ Transpose if needed │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│ STACK & MULTIPLY    │
│ Result = M1×M2×M3   │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│ SUM RESULT          │
│ count = Σ(Result)   │
└────┬────────────────┘
     │
     ▼
┌─────────────────────┐
│ APPEND to           │
│ motif_counts[]      │
└────┬────────────────┘
     │
     ▼
   NEXT VALUE/RULE
     │
     ▼
   END → Return motif_counts
```

### **Database Schema (When Using DB Mode)**
```
┌────────────────────────────────────────────┐
│         DATABASE STRUCTURE                  │
└────────────────────────────────────────────┘

Database: cora
├── papers (Entity Table)
│   ├── paper_id (PRIMARY KEY)
│   ├── feature_1
│   ├── feature_2
│   ├── feature_3
│   ├── feature_4
│   ├── feature_5
│   └── label
│
└── citations (Relation Table)
    ├── citing_paper_id (FOREIGN KEY → papers)
    └── cited_paper_id (FOREIGN KEY → papers)

Database: cora_setup
├── EntityTables
│   ├── TABLE_NAME
│   └── COLUMN_NAME (primary key)
│
├── RelationTables
│   └── TABLE_NAME
│
├── ForeignKeyColumns
│   ├── TABLE_NAME
│   ├── COLUMN_NAME
│   └── REFERENCED_TABLE_NAME
│
└── AttributeColumns
    ├── COLUMN_NAME
    └── TABLE_NAME

Database: cora_BN
├── Final_Path_BayesNets_view
│   ├── child
│   └── parent
│
└── [rule_name]_CP (Conditional Probability Tables)
    └── One table for each rule
```

### **Pickle File Structure**
```
┌────────────────────────────────────────────┐
│     PICKLE FILE (.pkl) STRUCTURE            │
└────────────────────────────────────────────┘

{
  "entities": {
    "papers": DataFrame(...)
  },
  
  "relations": {
    "citations": DataFrame(...)
  },
  
  "keys": {
    "papers": "paper_id",
    "citations": ("citing_paper_id", "cited_paper_id")
  },
  
  "matrices": {
    "citations": Tensor(2708×2708)
  },
  
  "rules": [
    ["feature_1(x1)"],
    ["feature_2(x1)"],
    ["feature_1(x1)", "citations(x1,x2)", "feature_2(x2)"],
    ...  # 147 total
  ],
  
  "indices": {
    "paper_id": {0: 0, 1: 1, ...}
  },
  
  "attributes": {
    "feature_1": "papers",
    ...
  },
  
  "base_indices": [[0], [0], [0, 1, 2], ...],
  "mask_indices": [[], [], [[0, 1]], ...],
  "sort_indices": [...],
  "stack_indices": [...],
  
  "values": [  # Conditional probabilities
    [(0.0, 542, 2708), ...],
    ...
  ],
  
  "prunes": [...],  # If rule pruning enabled
  
  "functors": {
    0: {0: "feature_1"},
    1: {0: "feature_2"},
    2: {0: "feature_1", 1: "citations", 2: "feature_2"},
    ...
  },
  
  "variables": {...},
  "nodes": {...},
  "states": [[0], [0], [0, 2, 0], ...],
  "masks": {...},
  "multiples": [0, 0, 1, ...],
  
  "entity_feature_columns": {
    "papers": ["feature_1", "feature_2", ...]
  },
  
  "relation_feature_columns": {...},
  "feature_info_mapping": {...},
  "num_nodes_graph": 2708
}
```
