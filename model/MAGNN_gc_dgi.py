import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.base_MAGNN import MAGNN_ctr_ntype_specific
from model.dgi import DGI

fc_switch = False
num_classes = 14

# multi-layer support
"""
Each instance of this layer accepts metapaths of a graph 
"""
class MAGNN_gc_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='RotatE1',
                 attn_drop=0):
        super(MAGNN_gc_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        if rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.ctr_ntype_layers = nn.ModuleList()
        for i in range(len(num_metapaths_list)):
            self.ctr_ntype_layers.append(MAGNN_ctr_ntype_specific(num_metapaths_list[i],
                                                                  etypes_lists[i],
                                                                  in_dim,
                                                                  num_heads,
                                                                  attn_vec_dim,
                                                                  rnn_type,
                                                                  r_vec,
                                                                  attn_drop,
                                                                  use_minibatch=False))

        # not fully connected here. Fully connected outside the MAGNN_gc_layers.
        self.fc = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists = inputs

        # ctr-ntype-specific layers
        h = torch.zeros(type_mask.shape[0], self.in_dim * self.num_heads, device=features.device)
        for i, (g_list, edge_metapath_indices_list, ctr_ntype_layer) in \
            enumerate(zip(g_lists, edge_metapath_indices_lists, self.ctr_ntype_layers)):

            # maybe one of these args is causing over-dimensionality
            h[np.where(type_mask == i)[0]] = \
                ctr_ntype_layer((g_list, features, type_mask, edge_metapath_indices_list, i))

        # not fully connected here. Fully connected outside the MAGNN_gc_layers.
        h_fc = self.fc(h)

        return h_fc, h

"""
Each instance of this class accepts the whole graph.
"""
class MAGNN_gc_dgi(nn.Module):
    def __init__(self,
                 num_layers,
                 num_metapaths_list, #[2,2,2] in IMDB
                 num_edge_type, #some int
                 etypes_lists, # [[[1, 2], [3, 4], [2, 0], [4, 0], [6, 0]], [[1,5]], [[..]], []]
                 feats_dim_list, #node features
                 hidden_dim, #some int
                 out_dim, #some int
                 num_heads,
                 attn_vec_dim,
                 rnn_type='RotatE1',
                 dropout_rate=0):
        super(MAGNN_gc, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after transformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_gc layers
        self.layers = nn.ModuleList()
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(MAGNN_gc_layer(num_metapaths_list,
                                               num_edge_type,
                                               etypes_lists,
                                               hidden_dim,
                                               hidden_dim,
                                               num_heads,
                                               attn_vec_dim,
                                               rnn_type,
                                               attn_drop = dropout_rate))
        # no output projection layer, just straight up feed node embedding to aggregator, then FC
        # no fc layer either; will feed directly into DGI (Ryan's modifications)

    def forward(self, inputs, target_node_indices):
        g_lists, features_list, type_mask, edge_metapath_indices_lists = inputs
        
        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
            h = self.feat_drop(transformed_features)

        # hidden layers
        for l in range(self.num_layers):
            h, _ = self.layers[l]((g_lists, h, type_mask, edge_metapath_indices_lists))
            h = F.elu(h)
        
        return h
