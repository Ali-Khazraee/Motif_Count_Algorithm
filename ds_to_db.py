from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from typing import Optional, Tuple, List
import torch
from torch import Tensor
import random
import numpy as np
from dgl.data import CoraGraphDataset
from pymysql import connect
import gc


def reduce_node_features(x, y, random_seed, n_components=5):
    """
    Reduce node features using ExtraTreesClassifier feature importance.
    
    Args:
        x: Node features (numpy array or torch tensor)
        y: Node labels (numpy array or torch tensor)
        random_seed: Random seed for reproducibility
        n_components: Number of top features to keep
        
    Returns:
        Reduced features and indices of important features
    """
    # Convert to numpy if needed
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    
    np.random.seed(random_seed)
    model = ExtraTreesClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_)
    important_feats = np.array(feat_importances.nlargest(n_components).index)
    x_reduced = x[:, important_feats]
    return x_reduced, important_feats


# Only run database creation when this script is run directly
if __name__ == "__main__":
    # Set random seed
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Load Cora dataset
    print("Loading Cora dataset...")
    dataset = CoraGraphDataset()
    graph = dataset[0]

    # Extract data from DGL graph
    features = graph.ndata['feat'].numpy()  # Node features
    labels = graph.ndata['label'].numpy()    # Node labels
    src, dst = graph.edges()
    edge_index = torch.stack([src, dst], dim=0)  # Edge indices

    print(f"Number of nodes: {graph.num_nodes()}")
    print(f"Number of features: {features.shape[1]}")
    print(f"Number of edges: {graph.num_edges()}")
    print(f"Number of classes: {dataset.num_classes}")

    # Binarize features
    features_binary = np.where(features > 0, 1, 0)

    # Reduce features using ExtraTreesClassifier
    print("Reducing features...")
    random_seed = 0
    x_reduced, important_feats = reduce_node_features(features_binary, labels, random_seed, n_components=5)
    x_reduced = torch.tensor(x_reduced, dtype=torch.float)

    # Add label as a node feature
    print("Adding label as node feature...")
    labels_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
    x_with_label = torch.cat([x_reduced, labels_tensor], dim=1)

    print(f"Features shape (with label): {x_with_label.shape}")

    # Database connection parameters
    print("\nDumping data to database")
    db_name = 'cora'
    connection_params = {
        'host': 'localhost',
        'user': 'fbuser',
        'password': ''
    }

    # Connect without specifying database first
    connection = connect(**connection_params)
    cursor = connection.cursor()

    # Check if database exists
    cursor.execute("SHOW DATABASES LIKE '%s'" % (db_name))
    db_exists = cursor.fetchone()

    if db_exists:
        print(f"Database '{db_name}' already exists. Using existing database.")
    else:
        print(f"Database '{db_name}' does not exist. Creating new database.")
        cursor.execute("CREATE DATABASE %s" % (db_name))

    # Now use the database
    cursor.execute("USE %s" % (db_name))

    # Create papers table (nodes) if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        paper_id INT PRIMARY KEY,
        feature_1 FLOAT,
        feature_2 FLOAT,
        feature_3 FLOAT,
        feature_4 FLOAT,
        feature_5 FLOAT,
        label FLOAT
    )
    """)

    # Create citations table (edges) if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS citations (
        citing_paper_id INT,
        cited_paper_id INT,
        PRIMARY KEY (citing_paper_id, cited_paper_id),
        FOREIGN KEY (citing_paper_id) REFERENCES papers(paper_id),
        FOREIGN KEY (cited_paper_id) REFERENCES papers(paper_id)
    )
    """)

    # Insert nodes into papers table
    print("Inserting papers into database...")

    # Check if papers table already has data
    cursor.execute("SELECT COUNT(*) FROM papers")
    paper_count = cursor.fetchone()[0]

    if paper_count > 0:
        print(f"Papers table already contains {paper_count} records. Skipping insertion.")
    else:
        for i, features in enumerate(x_with_label):
            # Convert tensor values to Python floats
            feature_values = [float(val) for val in features]
            
            cursor.execute(
                "INSERT INTO papers (paper_id, feature_1, feature_2, feature_3, feature_4, feature_5, label) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (i, *feature_values)
            )
        print("Done adding to papers table")

    # Insert edges into citations table
    print("Inserting citations into database...")

    # Check if citations table already has data
    cursor.execute("SELECT COUNT(*) FROM citations")
    citation_count = cursor.fetchone()[0]

    if citation_count > 0:
        print(f"Citations table already contains {citation_count} records. Skipping insertion.")
    else:
        for i in range(edge_index.shape[1]):
            cursor.execute(
                "INSERT INTO citations (citing_paper_id, cited_paper_id) VALUES (%s, %s)",
                (int(edge_index[0][i].item()), int(edge_index[1][i].item()))
            )
        print("Done adding to citations table")

    # Commit and close
    connection.commit()
    cursor.close()
    connection.close()

    print("\nDatabase creation completed successfully!")
    print(f"Database: {db_name}")
    print(f"Tables: papers, citations")
    print(f"Total papers: {graph.num_nodes()}")
    print(f"Total citations: {graph.num_edges()}")
