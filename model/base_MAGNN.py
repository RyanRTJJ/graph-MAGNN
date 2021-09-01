import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax

"""
one instance of this class deals with one metapath TYPE, not even one target node.
"""
class MAGNN_metapath_specific(nn.Module):
    def __init__(self,
                 etypes,
                 out_dim,
                 num_heads,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 alpha=0.01,
                 use_minibatch=False,
                 attn_switch=False):
        super(MAGNN_metapath_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rnn_type = rnn_type
        self.etypes = etypes
        self.r_vec = r_vec
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch

        # rnn-like metapath instance aggregator
        # consider multiple attention heads
        if rnn_type == 'gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim)
        elif rnn_type == 'bi-gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'bi-lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'max-pooling':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'neighbor-linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)

        # node-level attention
        # attention considers the center node embedding or not
        if self.attn_switch:
            self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
            self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        else:
            self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        # weight initialization
        if self.attn_switch:
            nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
            nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        else:
            nn.init.xavier_normal_(self.attn.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    """
    Just a helper function called in forward() to extract all the relevant node idxs from metapaths
    @param edge_metapath_indices: something like tensor([[2, 0, 6], [3, 0, 6]])
    """
    def get_sorted_relevant_nidxs(self, edge_metapath_indices):
        _relevant_node_idx_dict = {}
        _relevant_node_idx_list = []
        _emis_list = edge_metapath_indices.tolist()
        for _emis in _emis_list:
            _relevant_node_idx_dict[_emis[0]] = _relevant_node_idx_dict.get(_emis[0], 1)
            _relevant_node_idx_dict[_emis[-1]] = _relevant_node_idx_dict.get(_emis[-1], 1)
        for _n in _relevant_node_idx_dict.keys():
            _relevant_node_idx_list.append(_n)
        _relevant_node_idx_list.sort()
        return _relevant_node_idx_list

    """
    Just a helper function called in forward() to extract all target node idxs
    """
    def get_target_node_idxs(self, edge_metapath_indices):
        """
        _emis_list = edge_metapath_indices.tolist()
        if len(edge_metapath_indices) == 0:
            print("Not supposed to happen. base_MAGNN.py > get_target_node_idxs()")
        sample_target_node_idx = _emis_list[0][-1]
        target_node_type = type_mask[sample_target_node_idx]
        relevant_node_idxs = []
        for i in range(len(type_mask)):
            if type_mask[i] == target_node_type:
                relevant_node_idxs.append(i)
        return relevant_node_idxs
        """
        
        _emis_list = edge_metapath_indices.tolist()
        _relevant_node_idx_dict = {}
        _relevant_node_idx_list = []
        for _emis in _emis_list:
            _relevant_node_idx_dict[_emis[-1]] = _relevant_node_idx_dict.get(_emis[-1], 1)
        for _n in _relevant_node_idx_dict.keys():
            _relevant_node_idx_list.append(_n)
        _relevant_node_idx_list.sort() #not really needed
        return _relevant_node_idx_list
    
    def forward(self, inputs):
        # features: num_all_nodes x out_dim
        if self.use_minibatch:
            g, features, type_mask, edge_metapath_indices, target_idx = inputs
        else:
            g, features, type_mask, edge_metapath_indices, target_n_type = inputs

        
        nodes_of_current_target_type_idxs = np.where(type_mask == target_n_type)[0].tolist()
        ret = F.embedding(torch.tensor(nodes_of_current_target_type_idxs, dtype=torch.int32, device='cuda:0'), features)
        ret = torch.cat([ret] * self.num_heads, dim=1)
        ret = ret.unsqueeze(dim=0)
        ret = ret.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)

        if g == None: # means that edge_metapath_indices also = []
            # print("target_n_type: {}, None graph found. Returning...".format(target_n_type))
            return ret
        
        # For later: extract local idx of target node.
        # e.g. if edge_metapath_indices = [[2,0,6],[3,0,6]],
        # in this nxgraph it would correspond to [[1 --> 2],[3 --> 6], (intermediate nodes not saved)
        _relevant_node_idx_list = self.get_sorted_relevant_nidxs(edge_metapath_indices)
        _target_node_idxs = self.get_target_node_idxs(edge_metapath_indices)
        fullg_nidx_to_partialg_nidx_mapping = {}
        for _i, _n in enumerate(_relevant_node_idx_list):
            fullg_nidx_to_partialg_nidx_mapping[_n] = _i
        target_node_idxs_local = [fullg_nidx_to_partialg_nidx_mapping[tnode] for \
                                  tnode in _target_node_idxs]
            
        # Embedding layer
        # use torch.nn.functional.embedding or torch.embedding here
        # do not use torch.nn.embedding
        # edata: E x Seq x out_dim
        edata = F.embedding(edge_metapath_indices, features)

        # if (no_such_metapath_instances):
        #     edata = torch.empty((0, 3, 64), dtype=torch.float64, device='cuda:0')
        
        # apply rnn to metapath-based feature sequence
        # if no_such_metapath_instances:
        #     hidden = edata
        #     return hidden # zero tensors
        if self.rnn_type == 'gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'bi-gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'bi-lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'average':
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'linear':
            hidden = self.rnn(torch.mean(edata, dim=1))
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'max-pooling':
            hidden, _ = torch.max(self.rnn(edata), dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'TransE0' or self.rnn_type == 'TransE1':
            r_vec = self.r_vec
            if self.rnn_type == 'TransE0':
                r_vec = torch.stack((r_vec, -r_vec), dim=1)
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1])  # etypes x out_dim
            edata = F.normalize(edata, p=2, dim=2)
            for i in range(edata.shape[1] - 1):
                # consider None edge (symmetric relation)
                temp_etypes = [etype for etype in self.etypes[i:] if etype is not None]
                edata[:, i] = edata[:, i] + r_vec[temp_etypes].sum(dim=0)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'RotatE0' or self.rnn_type == 'RotatE1':
            r_vec = F.normalize(self.r_vec, p=2, dim=2)
            if self.rnn_type == 'RotatE0':
                r_vec = torch.stack((r_vec, r_vec), dim=1)
                r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)  # etypes x out_dim/2 x 2
            edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
            final_r_vec = torch.zeros([edata.shape[1], self.out_dim // 2, 2], device=edata.device)
            final_r_vec[-1, :, 0] = 1
            for i in range(final_r_vec.shape[0] - 2, -1, -1):
                # consider None edge (symmetric relation)
                if self.etypes[i] is not None: # removed "self." in front of all "etypes"
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] -\
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] +\
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
                else:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
            for i in range(edata.shape[1] - 1):
                temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] -\
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
                temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] +\
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
                edata[:, i, :, 0] = temp1
                edata[:, i, :, 1] = temp2
            edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor':
            hidden = edata[:, 0]
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor-linear':
            hidden = self.rnn(edata[:, 0])
            hidden = hidden.unsqueeze(dim=0)

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_di
        
        if self.attn_switch:
            center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  # E x out_dim
            a1 = self.attn1(center_node_feat)  # E x num_heads
            a2 = (eft * self.attn2).sum(dim=-1)  # E x num_heads
            a = (a1 + a2).unsqueeze(dim=-1)  # E x num_heads x 1
        else:
            a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1
        a = self.leaky_relu(a)
        g.edata.update({'eft': eft, 'a': a})
        # compute softmax normalized attention values
        self.edge_softmax(g)
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        new_nfeatures = g.ndata['ft'][target_node_idxs_local]  # E x num_heads x out_dim

        # all nodes of current type = [5, 6]
        # target_node_idxs = [6]
        
        fill_emb_idx = [nodes_of_current_target_type_idxs.index(i) for i in _target_node_idxs]
        ret = F.embedding(torch.tensor(nodes_of_current_target_type_idxs, dtype=torch.int32, device='cuda:0'), features)
        ret = torch.cat([ret] * self.num_heads, dim=1)
        ret = ret.unsqueeze(dim=0)
        ret = ret.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)
        ret[fill_emb_idx] = new_nfeatures
        
        # Want to return embeddings of all nodes of current node type
        # If all nodes of current type are [5, 6], but only [6] was involved in metapaths,
        # still want to return a data for [5, 6], where 5 is simply the original node data
        if self.use_minibatch:
            #idk what this should be (Ryan)
            pass
        else:
            return ret

"""
Each instance of this class accepts all metapaths of a target node type
"""
class MAGNN_ctr_ntype_specific(nn.Module):
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 use_minibatch=False):
        super(MAGNN_ctr_ntype_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch

        # metapath-specific layers
        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metapath_layers.append(MAGNN_metapath_specific(etypes_list[i],
                                                                out_dim,
                                                                num_heads,
                                                                rnn_type,
                                                                r_vec,
                                                                attn_drop=attn_drop,
                                                                use_minibatch=use_minibatch))

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.use_minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1, self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, target_idx, metapath_layer, in zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
        else:
            g_list, features, type_mask, edge_metapath_indices_list, target_n_type = inputs

            # metapath-specific layers
            # each of these layers correspond to a metapath type of same target node type
            # e.g.: (0,2,0), (0,3,0), (2,0,0)
            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_n_type)).view(-1, self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, metapath_layer in zip(g_list, edge_metapath_indices_list, self.metapath_layers)]

        beta = []
        if len(metapath_outs) == 0:
            nodes_of_current_target_type_idxs = np.where(type_mask == target_n_type)[0].tolist()
            ret = F.embedding(torch.tensor(nodes_of_current_target_type_idxs, dtype=torch.int32, device='cuda:0'), features)
            ret = torch.cat([ret] * self.num_heads, dim=1)
            ret = ret.unsqueeze(dim=0)
            ret = ret.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)
            metapath_outs = [F.elu(ret).view(-1, self.num_heads * self.out_dim)]

        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)

        # print("beta:")
        # print(beta)
        # print(len(beta))

        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h
