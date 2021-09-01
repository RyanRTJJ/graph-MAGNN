import time
import argparse

import torch as th
import torch.nn.functional as F
import torch.sparse
import numpy as np
import dgl

from utils.pytorchtools import EarlyStopping
import utils.casedata as data
from model import MAGNN_gc
from sklearn import model_selection

# Params
out_dim = 14
dropout_rate = 0.05
lr = 0.005
weight_decay = 0.001

# TODO: improve on this list of etypes
etypes_lists = [
        [[1, 2], [3, 4], [2, 0], [4, 0], [6, 0]],
        [[1, 5]],
        [[2, 1], [4, 1], [6, 1]],
        [[2, 3], [4, 3], [6, 3]],
        []
    ]
metapaths = [
        [(0,2,0), (0,3,0), (2,0,0), (3,0,0), (4,0,0)],
        [(0,2,1)],
        [(2,0,2), (3,0,2), (4,0,2)],
        [(2,0,3), (3,0,3), (4,0,3)],
        []
    ]
metapath_to_edge_mapping = {
    (0, 0): 0,
    (0, 2): 1,
    (2, 0): 2,
    (0, 3): 3,
    (3, 0): 4,
    (2, 1): 5,
    (4, 0): 6
}
"""
@param nx_G_dict: [ { (0,2,0): nx_G, (0,2,1): nx_G }, 
                    {}, 
                    { (2,0,3): nx_G }, 
                    {}, 
                    {}
                  ]
@return: [
             [nx_G, nx_G, 

def nx_G_dict_to_list_mapped_to_metapath_universal(nx_G_dict):

    for i in range(len(nx_G_dict)): # len(nx_G_dict) MUST == len(metapaths)
        for j in range(metapaths[i]):
            if j in nx_G_dict[i]:
"""                

def get_net_inputs(device, data_by_graphs, graph_num):
    nx_G_lists = data_by_graphs[graph_num]['Gs']
    edge_metapath_indices_lists = data_by_graphs[graph_num]['idxs']
    features_list = data_by_graphs[graph_num]['features']
    adjM = data_by_graphs[graph_num]['adjM']
    type_mask = data_by_graphs[graph_num]['type_mask']
    features_list = [torch.FloatTensor(features.todense()).to(device) for features in features_list]
    target_node_indices = [True for _ in range(adjM.shape[0])]
    in_dims = [features.shape[1] for features in features_list] # all the same basically

    #convert edge_metapath_indices_list into a Tensor
    edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in edge_metapath_indices_lists]

    g_lists = [] # [[G,G,G,G,G], [G], [G,G,G], [G,G,G], []]
    for nx_G_list in nx_G_lists:
        g_lists.append([])
        for nx_G in nx_G_list:
            if (nx_G == None):
                g_lists[-1].append(None)
                continue
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(nx_G.number_of_nodes())
            g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
            g_lists[-1].append(g)

    return g_lists, features_list, type_mask, edge_metapath_indices_lists, target_node_indices

def get_missing_ntypes(type_mask):
    missing_ntypes = []
    for i in range(5):
        if i not in type_mask:
            missing_ntypes.append(i)
    return missing_ntypes

def interrim_debug(data_by_graphs):
    for gnum, g in data_by_graphs.items():
        type_mask = g['type_mask']
        print("graph {}: ".format(gnum) + str(type_mask))

"""
Helper function called in run_casesim.
Layers relevant to node types that don't exist in a training graph need to be frozen, otherwise nan.
@param missing_n_types: a list containing missing node types.
@param net: the MAGNN_gc net
@param freeze: True = freeze. False = unfreeze.
"""
def toggle_layers_of_missing_ntypes(missing_n_types, net, freeze = True):
    num_magnn_gc_layers = len(net.layers)
    for i in range(num_magnn_gc_layers - 1):
        magnn_gc = net.layers[i]
        num_cnl_layers = len(magnn_gc.ctr_ntype_layers)
        for j in range(num_cnl_layers):
            cnl = magnn_gc.ctr_ntype_layers[j]
            if j in missing_n_types:
                # cnl.fc1.weight.requires_grad = False
                if freeze:
                    cnl.fc2.weight.requires_grad = False
                else:
                    cnl.fc2.weight.requires_grad = True

"""
Helper function to make repeated training and evaluation neater.
@param device: torch.device('cuda:0')
@param net: the MAGNN_gc Object.
@param data_by_graphs: a dict{ graph_num: dict{feature_name: feature_value} }
@param train_idx: a list of training graph NUMBERS
@param val_idx: a list of val graph NUMBERS
"""
def train_and_evaluate_one_epoch(device, net, optimizer, data_by_graphs, graph_ground_truth, \
                                 train_idx, val_idx, epoch, best_val_loss=None):
    train_acc = 0
    num_train_elems = 0
    mean_train_loss = 0

    val_acc = 0
    num_val_elems = 0
    mean_val_loss = 0

    for graph_num in train_idx:
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_node_indices = \
            get_net_inputs(device, data_by_graphs, graph_num)
        missing_ntypes = get_missing_ntypes(type_mask)

        # If nodes are missing, freeze the NN layers of that type, otherwise nan gradient.
        toggle_layers_of_missing_ntypes(missing_ntypes, net, freeze=True)

        net.train()
        logits = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), \
                     target_node_indices)
        pred = logits.argmax(1) # [class_number]
        truth_label = torch.tensor([graph_ground_truth[graph_num]], dtype=torch.long, device=device)
        
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp, truth_label)
        mean_train_loss += train_loss / len(train_idx)

        if pred == truth_label: train_acc += 1
        num_train_elems += 1

        # Have to do SGD because of how the layer freezing happens specific to graph sample
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Unfreeze the layer gradients
        toggle_layers_of_missing_ntypes(missing_ntypes, net, freeze=False)

    # eval
    net.eval()
    with torch.no_grad():
        for graph_num in val_idx:
            g_lists, features_list, type_mask, edge_metapath_indices_lists, target_node_indices = \
                get_net_inputs(device, data_by_graphs, graph_num)
            logits = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), \
                         target_node_indices)

            pred = logits.argmax(1)
            truth_label = torch.tensor([graph_ground_truth[graph_num]], \
                                       dtype=torch.long, device=device)

            logp = F.log_softmax(logits, 1)
            val_loss = F.nll_loss(logp, truth_label)
            mean_val_loss += val_loss / len(val_idx)

            if pred == truth_label: val_acc += 1
            num_val_elems += 1

    # print info
    print("\nEpoch {:05d} | Train_Acc: {:3d}/{:3d} ({:.4f}) | Val_acc: {:3d}/{:3d} ({:.4f})".format(epoch, train_acc, num_train_elems, train_acc/num_train_elems, val_acc, num_val_elems, val_acc/num_val_elems))
    print("Mean Train Loss: {:.4f} | Mean Val Loss: {:.4f}".format(mean_train_loss, mean_val_loss))
    return mean_train_loss, mean_val_loss

def predict(device, net, data_by_graphs, graph_num):
    g_lists, features_list, type_mask, edge_metapath_indices_lists, target_node_indices = \
        get_net_inputs(device, data_by_graphs, graph_num)
    logits = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), \
                 target_node_indices)
    pred = logits.argmax(1)
    return pred

def predict_graph_nums_and_log(device, net, data_by_graphs, graph_ground_truth, val_idx, log_file):
    log = open(log_file, "a+")
    for graph_num in val_idx:
        pred = predict(device, net, data_by_graphs, graph_num)
        truth_label = torch.tensor([graph_ground_truth[graph_num]], dtype=torch.long, device=device)
        log.write("\n{:03d}: {:03d}".format(graph_num, pred[0].item()))
        if pred == truth_label:
            log.write(" CORRECT")
    log.close()

def run_model_casesim(num_layers, hidden_dim, num_heads, attn_vec_dim, rnn_type, num_epochs, patience, repeat, save_postfix, save_modelname):

    # Getting ready device settings
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Getting ready data.
    train_graph_nums = [i for i in range(1,106)]
    blacklisted_graphs = [4, 5, 6, 7, 9, 18, 25, 31, 42, 58, 67, 81, 102]
    for b in blacklisted_graphs:
        if b in train_graph_nums:
            train_graph_nums.remove(b)
    data_by_graphs = data.load_casesim_data(graphs=train_graph_nums)
    graph_ground_truth = data.load_graph_ground_truth()
    in_dims = [19,19,19,19,19]

    # Convergence split (1 split)
    train_idx = train_graph_nums
    val_idx = train_graph_nums
    loo_splits = {
        0: (train_idx, val_idx)
    }
    
    # Real LOO (92 splits)
    """
    loo_splits = {}
    for i in range(len(train_graph_nums)):
        train_idx = []
        val_idx = [train_graph_nums[i]]
        for j in train_graph_nums:
            if j == train_graph_nums[i]:
                continue
            train_idx.append(j)
        loo_splits[i] = (train_idx, val_idx)
    """

    #train_idx, val_idx = model_selection.train_test_split(train_graph_nums, train_size=0.7, random_state=45)
    
    # train MAGNN_gc repeat times
    for _, (train_idx, val_idx) in loo_splits.items():
        
        print("\n\nTraining starting!!!")
        print("\ntrain_idx: " + str(train_idx))
        print("\nval_idx: " + str(val_idx))
        net = MAGNN_gc(num_layers,
                       [5, 1, 3, 3, 0],
                       7,
                       etypes_lists,
                       in_dims,
                       hidden_dim,
                       out_dim,
                       num_heads,
                       attn_vec_dim,
                       rnn_type,
                       dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path="/scratch/jinjtan/ryan-exploration/MAGNN/checkpoint_{}.pt".format(save_postfix))

        best_val_loss = None

        local_save_modelname = save_modelname
        if len(val_idx) == 1:
            # different model name for LOO
            local_save_modelname += '_idx{}'.format(val_idx[0])
        for epoch in range(num_epochs):
            train_acc = 0
            num_elems = 0
            val_acc = 0
            num_val_elems = 0
            epoch_val_loss = None
            # TODO: This for loop should only happen for train_set

            mean_train_loss, mean_val_loss = train_and_evaluate_one_epoch(device,
                                                                          net,
                                                                          optimizer,
                                                                          data_by_graphs,
                                                                          graph_ground_truth,
                                                                          train_idx,
                                                                          val_idx,
                                                                          epoch)

            # save model
            if best_val_loss == None or best_val_loss > mean_val_loss:
                best_val_loss = mean_val_loss
                th.save(net.state_dict(), local_save_modelname + '_{}'.format(epoch))
                print("Saving model!!!")
                predict_graph_nums_and_log(device, net, data_by_graphs, graph_ground_truth, val_idx, "training_log_convergence_4_128")
                        
            # early stopping
            early_stopping(mean_val_loss, net)
            if early_stopping.early_stop:
                print("Early stopping!")
                break
                                              
# entry point
if __name__ == '__main__':
    # TODO: argparse

    run_model_casesim(
        num_layers = 2,
        hidden_dim = 512,
        num_heads = 8,
        attn_vec_dim = 1024, # should be double of hidden_dim
        rnn_type = 'RotatE1',
        num_epochs = 500,
        patience = 25,
        repeat = 1,
        save_postfix = 'casesim',
        save_modelname = 'saved_models/model_casesim'
    )
