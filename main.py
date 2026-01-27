from data import DataLoader 
from ds_to_db import reduce_node_features
import numpy as np 
import torch
import random
from motif_count import Motif_Count
import argparse


parser = argparse.ArgumentParser(description='Kernel VGAE')

parser.add_argument('-e', dest="epoch_number", default=20000, help="Number of Epochs to train the model", type=int)
parser.add_argument('-v', dest="Vis_step", default=1000, help="at every Vis_step 'minibatch' the plots will be updated")
parser.add_argument('-redraw', dest="redraw", default=False, help="either update the log plot each step")
parser.add_argument('-lr', dest="lr", default=0.0003, help="model learning rate")
parser.add_argument('-dataset', dest="dataset", default="Cora_dgl",
                    help="possible choices are:   wheel_graph,PTC, FIRSTMM_DB, star, triangular_grid, multi_community, NCI1, ogbg-molbbbp, IMDbMulti, grid, community, citeseer, lobster, DD")  # citeceer: ego; DD:protein
parser.add_argument('-graphEmDim', dest="graphEmDim", default=1024, help="the dimention of graph Embeding LAyer; z")
parser.add_argument('-graph_save_path', dest="graph_save_path", default=None,
                    help="the direc to save generated synthatic graphs")
parser.add_argument('-f', dest="use_feature", default=True, help="either use features or identity matrix")
parser.add_argument('-PATH', dest="PATH", default="model",
                    help="a string which determine the path in wich model will be saved")
parser.add_argument('-decoder', dest="decoder", default="FC", help="the decoder type, FC is only option in this rep")
parser.add_argument('-encoder', dest="encoder_type", default="AvePool",
                    help="the encoder: only option in this rep is 'AvePool'")  # only option in this rep is "AvePool"
parser.add_argument('-batchSize', dest="batchSize", default=200,
                    help="the size of each batch; the number of graphs is the mini batch")
parser.add_argument('-UseGPU', dest="UseGPU", default=True, help="either use GPU or not if availabel")
parser.add_argument('-model', dest="model", default="GraphVAE-MM",
                    help="KernelAugmentedWithTotalNumberOfTriangles and kipf is the only option in this rep; NOTE KernelAugmentedWithTotalNumberOfTriangles=GraphVAE-MM and kipf=GraphVAE")
parser.add_argument('-device', dest="device", default="cuda:0", help="Which device should be used")
parser.add_argument('-task', dest="task", default="graphGeneration", help="only option in this rep is graphGeneration")
parser.add_argument('-BFS', dest="bfsOrdering", default=True, help="use bfs for graph permutations", type=bool)
parser.add_argument('-directed', dest="directed", default=True, help="is the dataset directed?!", type=bool)
parser.add_argument('-beta', dest="beta", default=None, help="beta coefiicieny", type=float)
parser.add_argument('-plot_testGraphs', dest="plot_testGraphs", default=True, help="shall the test set be printed",
                    type=float)
parser.add_argument('-ideal_Evalaution', dest="ideal_Evalaution" , default=False, help="if you want to comapre the 50%50 subset of dataset comparision?!", type=bool)



#-==-=-=-=--------------------------======================================================================================
parser.add_argument('--motif_loss', type=bool, default=True,
                    help='Enable motif loss term in objective function')

parser.add_argument('--rule_prune', type=bool, default=False,
                    help='Enable rule pruning in motif counting')

parser.add_argument('--rule_weight', type=bool, default=False,
                    help='Enable rule weighting (requires rule_prune=True)')

parser.add_argument('--graph_type', type=str, default='homogeneous',
                    choices=['homogeneous', 'heterogeneous'],
                    help='Graph type for motif counting')

parser.add_argument('--lambda_motif', type=float, default=1.0,
                    help='Weight coefficient for motif loss term')

parser.add_argument('--motif_count_eval', type=bool, default=True,
                    help='Enable motif count evaluation at end of training')

parser.add_argument(
    '--test_local_mults',type=bool,default=True,
    help='Enable validation of motif count number against FactorBase outputs'
)

parser.add_argument(
    '--test_small_graph_motifs',
    type=bool,
    default=True,
    help='Evaluate motif counts on small graphs with known ground truth'
)
#-==-=-=-=--------------------------======================================================================================

args = parser.parse_args()

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


loader = DataLoader()

loader.load_data()

data = loader.get_data()

x_reduced, important_feats = reduce_node_features(data['features'], data['labels'], random_seed= 0, n_components=5)
labels_column = data['labels'].numpy().reshape(-1, 1)
x_with_labels = np.concatenate([x_reduced, labels_column], axis=1)




CM = Motif_Count(args)
CM.setup_function()

reconstructed_x_slice, _ = CM.process_reconstructed_data(
    None, [data['adjacency_matrix']],
    x_with_labels,
    important_feats,
    None
)

motifs = CM.iteration_function(reconstructed_x_slice, None ,mode="test")

