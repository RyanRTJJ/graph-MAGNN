import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.base_MAGNN import MAGNN_ctr_ntype_specific

fc_switch = False

# multi-layer support
class MAGNN_gc_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='RotatE0',
                 attn_drop=0):
        super(MAGNN_gc_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
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

            # type_mask here refers to node type (we have 5)
            h[np.where(type_mask == i)[0]] = \
                ctr_ntype_layers((g_list, features, type_mask, edge_metapath_indices_list))

        # not fully connected here. Fully connected outside the MAGNN_gc_layers.
