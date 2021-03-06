import torch as th
import numpy as np
import pandas as pd
import networkx as nx
import dgl
import math
import scipy.sparse
import scipy.io
import pathlib
import copy

"""
etypes_lists = [
    [[1, 2], [1, 5], [3, 4]],
    [],
    [[2, 0], [2, 1], [2, 3]],
    [[4, 0], [4, 1], [4, 3]],
    [[6, 0], [6, 1], [6, 3]],
]
"""
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
"""
metapaths = [
    [(0,2,0), (0,2,1), (0,3,0)],
    [],
    [(2,0,0), (2,0,2), (2,0,3)],
    [(3,0,0), (3,0,2), (3,0,3)],
    [(4,0,0), (4,0,2), (4,0,3)]
]
"""

def load_graph_ground_truth(truth_path='/scratch/jinjtan/ryan-exploration/MAGNN/data/truth.csv'):
    print("loading graph_ground_truth...")
    truth = pd.read_csv(truth_path, header=None, index_col=0)
    truth = truth.sort_values(by=0)
    class_mapping = {
        "Ponzi Scheme": 0,
        "Unregistered MSB": 1,
        "Unknown Source of Check Funding": 2,
        "Political Corruption": 3,
        "Shell Company": 4,
        "Vehicle Exporting": 5,
        "Structuring Utilizing OBCs": 6,
        "Employee Corruption": 7,
        "Illegal Import/Export": 8,
        "Phony Storefront": 9,
        "Tax Evasion": 10,
        "Dormant Account": 11,
        "Funnel Account": 12,
        "Human Trafficking": 13
    }

    out = {}
    for idx, value in truth.iterrows():
        out[idx] = class_mapping[value[1]] # idx here is already 1-indexed
    print("loaded graph_ground_truth.")
    print(out)
    return out
    

def deleteNodesNoEdge(nodes, edges):
    node_edges = set(edges['From Entity (Originator)'].unique()).union(set(edges['To Entity (Beneficiary)'].unique()))
    nodesToDelete = list(set(nodes.index.unique()).difference(node_edges))
    return nodes.drop(nodesToDelete, axis='index')

# Reads node_file.csv, edge_file.csv
# Splits edge properties (connection_type, amount), puts on sink / source nodes
# Returns df_nodes, df_edges
def loadData(node_file, edge_file):
    df_nodes_original = pd.read_csv(node_file).set_index('Node ID')
    df_edges_original = pd.read_csv(edge_file)

    # Drop edges without connections
    df_nodes = df_nodes_original.copy()
    df_edges = df_edges_original.copy()
    df_nodes = deleteNodesNoEdge(df_nodes, df_edges)

    # Only get relevant columns
    df_nodes = df_nodes[['Node Type', 'original_graph']]
    df_edges = df_edges[['From Entity (Originator)', 'To Entity (Beneficiary)', 'Connection Type', 'Amount', 'Date', 'original_graph']]

    # Add edge features to source and sink nodes
    edge_features_by_node = dict()
    for Node_ID, node_object in df_nodes.iterrows():
        
        # Create dictionary of binary features to store edge information for each node.
        edge_features_by_node[Node_ID] = {
            # Features of edges leading into this node
            'sink_edge_features': {
                'sink_type_Cash Transaction': 0,
                'sink_type_Check': 0,
                'sink_type_Electronic Transaction': 0,
                'sink_type_Has Account': 0,
                'sink_type_Has Address': 0,
                'sink_type_OBC': 0,
                'sink_Amount': 0,
                # TODO: 'date'
            },
            # Features of edges leading away from this node
            'source_edge_features': {
                'source_type_Cash Transaction': 0,
                'source_type_Check': 0,
                'source_type_Electronic Transaction': 0,
                'source_type_Has Account': 0,
                'source_type_Has Address': 0,
                'source_type_OBC': 0,
                'source_Amount': 0,
                # TODO: 'date'
            }
        }

    for idx, edge_object in df_edges.iterrows():
        # update sink node feature dictionary
        sink_Node_ID = edge_object['To Entity (Beneficiary)']
        connection_type = edge_object['Connection Type']
        connection_amount = edge_object['Amount']
        
        node_sink_feature_key = 'sink_type_' + connection_type
        edge_features_by_node[sink_Node_ID]['sink_edge_features'][node_sink_feature_key] += 1
        if not math.isnan(connection_amount):
            edge_features_by_node[sink_Node_ID]['sink_edge_features']['sink_Amount'] += connection_amount

        # do the same for source
        source_Node_ID = edge_object['From Entity (Originator)']
        node_source_feature_key = 'source_type_' + connection_type
        edge_features_by_node[source_Node_ID]['source_edge_features'][node_source_feature_key] += 1
        if not math.isnan(connection_amount):
            edge_features_by_node[source_Node_ID]['source_edge_features']['source_Amount'] += connection_amount

    edge_features_listed = {
        'sink': {
            'sink_type_Cash Transaction': [],
            'sink_type_Check': [],
            'sink_type_Electronic Transaction': [],
            'sink_type_Has Account': [],
            'sink_type_Has Address': [],
            'sink_type_OBC': [],
            'sink_Amount': []
        },
        'source': {
            'source_type_Cash Transaction': [],
            'source_type_Check': [],
            'source_type_Electronic Transaction': [],
            'source_type_Has Account': [],
            'source_type_Has Address': [],
            'source_type_OBC': [],
            'source_Amount': []
        }   
    }

    edge_types = ['Cash Transaction', 'Check', 'Electronic Transaction', 'Has Account', 'Has Address', 'OBC']

    for Node_ID in edge_features_by_node:
        for e_type in edge_types:
            # append all the sinks first
            sink_etype_key = 'sink_type_' + e_type
            edge_features_listed['sink'][sink_etype_key].append( \
                edge_features_by_node[Node_ID]['sink_edge_features'][sink_etype_key])
            # then append the sources
            source_etype_key = 'source_type_' + e_type
            edge_features_listed['source'][source_etype_key].append( \
                edge_features_by_node[Node_ID]['source_edge_features'][source_etype_key])
        edge_features_listed['sink']['sink_Amount'].append( \
            edge_features_by_node[Node_ID]['sink_edge_features']['sink_Amount'])
        edge_features_listed['source']['source_Amount'].append( \
            edge_features_by_node[Node_ID]['source_edge_features']['source_Amount'])

    for e_type in edge_types:
        e_type_key = 'sink_type_' + e_type
        df_nodes[e_type_key] = edge_features_listed['sink'][e_type_key]
    df_nodes['sink_Amount'] = edge_features_listed['sink']['sink_Amount']
    # add source features to df_nodes
    for e_type in edge_types:
        e_type_key = 'source_type_' + e_type
        df_nodes[e_type_key] = edge_features_listed['source'][e_type_key]
    df_nodes['source_Amount'] = edge_features_listed['source']['source_Amount']

    # convert Node Type to 1-hot categorical
    # convert Edge Type to 1-hot categorical (won't be used, but cleaning requires more work)
    df_nodes = pd.get_dummies(df_nodes, 'Node Type')
    df_edges = pd.get_dummies(df_edges, 'type', columns=['Connection Type'])
    print("df_edges.columns")
    print(df_edges.columns)

    df_nodes = df_nodes.sort_values(by=['original_graph'])
    return df_nodes, df_edges

# transforms graphs to:
# uses DGL
def transform_graph(df_nodes, df_edges):
    columns = [
        'Node Type_Account',
        'Node Type_Address',
        'Node Type_Customer',
        'Node Type_External Entity',
        'Node Type_Legal Entity',
        'sink_type_Cash Transaction',
        'sink_type_Check',
        'sink_type_Electronic Transaction',
        'sink_type_Has Account',
        'sink_type_Has Address',
        'sink_type_OBC',
        'sink_Amount',
        'source_type_Cash Transaction',
        'source_type_Check',
        'source_type_Electronic Transaction',
        'source_type_Has Account',
        'source_type_Has Address',
        'source_type_OBC',
        'source_Amount'
    ]
    # By limiting columm_edges to this list, dummified versions of other connection types like "Wendy" will be ignored
    columns_edges = [
        'type_Cash Transaction',
        'type_Check',
        'type_Electronic Transaction',
        'type_Has Account',
        'type_Has Address'
    ]

    dataset = {}
    for graph_nb in sorted(df_nodes['original_graph'].unique()):
        edges = df_edges[df_edges['original_graph'] == graph_nb].copy().reset_index()
        nodes = df_nodes[df_nodes['original_graph'] == graph_nb].copy()
        nodeIdMapping = {x: i for i, x in enumerate(nodes.index.to_list())}

        edges['From Entity (Originator)'] = edges['From Entity (Originator)'].map(nodeIdMapping)
        edges['To Entity (Beneficiary)'] = edges['To Entity (Beneficiary)'].map(nodeIdMapping)
        graph = dgl.graph((edges['From Entity (Originator)'], edges['To Entity (Beneficiary)']))

        # Node features
        graph.ndata['feat'] = th.tensor(nodes[columns].to_numpy(), dtype=th.float)
        # Graph to which the node belongs
        graph.ndata['original_graph'] = th.tensor(nodes['original_graph'])
        # Edge features
        graph.edata['feat'] = th.tensor(edges[columns_edges].to_numpy(), dtype=th.float)

        dataset[graph_nb] = graph

    return dataset

def get_type_mask(df_nodes):
    num_nodes = df_nodes.shape[0]

    type_mask = np.zeros((num_nodes), dtype=int)
    for node_idx, row in df_nodes.reset_index().iterrows():
        if row['Node Type_Account'] == 1:
            type_mask[node_idx] = 0
        elif row['Node Type_Address'] == 1:
            type_mask[node_idx] = 1
        elif row['Node Type_Customer'] == 1:
            type_mask[node_idx] = 2
        elif row['Node Type_External Entity'] == 1:
            type_mask[node_idx] = 3
        elif row['Node Type_Legal Entity'] == 1:
            type_mask[node_idx] = 4

    return type_mask

# used in get_metapath_neighbor_pairs(), NOT YET right now.
# same as bfs_helper, but backtraces from dst to src nodes instead of moving forward along edges
"""
@param adjM: numpy matrix, adjacency matrix (directed)
@param type_mask: list, where type_mask[i] = node type of node i
@param metapath: tuple containing node types, e.g. (2, 0 ,3) 
"""
def bfs_helper_reverse(adjM, type_mask, metapath, paths_so_far, d):
    if len(paths_so_far) == 0:
        return []
    if (len(paths_so_far[0]) - 1 == d):
        return paths_so_far
    extended_paths = []
    for path_so_far in paths_so_far:
        current_node = path_so_far[0]
        neighbors = np.nonzero(adjM[current_node, :])[0]
        neighbors_correct_type = []
        for neighbor in neighbors:
            if type_mask[neighbor] == metapath[len(path_so_far)]:
                neighbors_correct_type.append(neighbor)
        # can't go any further
        if len(neighbors_correct_type) == 0:
            continue
        for neighbor in neighbors_correct_type:
            extended_path = [neighbor]
            for n in path_so_far:
                extended_path.append(n)
            extended_paths.append(extended_path)
    return bfs_helper_reverse(adjM, type_mask, metapath, extended_paths, d)

# used in get_metapath_neighbor_pairs()
def bfs_helper(adjM, type_mask, metapath, paths_so_far, d):
    if len(paths_so_far) == 0:
        return []
    if (len(paths_so_far[0]) - 1 == d):
        return paths_so_far
    extended_paths = []
    for path_so_far in paths_so_far:
        current_node = path_so_far[-1]
        neighbors = np.nonzero(adjM[:, current_node])[0]
        neighbors_correct_type = []
        for neighbor in neighbors:
            if type_mask[neighbor] == metapath[len(path_so_far)]:
                neighbors_correct_type.append(neighbor)
        # Can't go any further
        if len(neighbors_correct_type) == 0:
            continue
        for neighbor in neighbors_correct_type:
            extended_path = []
            for n in path_so_far:
                extended_path.append(n)
            extended_path.append(neighbor)
            extended_paths.append(extended_path)
    return bfs_helper(adjM, type_mask, metapath, extended_paths, d)

def get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths):
    """
    @param adM: adjacency matrix (a 2D numpy array)
    @param type_mask: a 1D python array, where type_mask[i] = type(node i)
    @param expected_metapaths: a list of tuples describing metapaths starting with a certain type
    @return: a list a python disctionaries, key: metapath-based neighbor pairs, value: full paths
    """
    outs = []
    for metapath in expected_metapaths:
        d = len(metapath) - 1
        # Want to create mask that filters through the edges that are in metapath, in adjM
        mask = np.zeros(adjM.shape, dtype=bool)

        # FUTURE: don't assume symmetry of path
        for i in range(len(metapath) - 1):
        # ----------------------------------
        # CURRENT: assuming symmetry of path
        # for i in range((len(metapath) - 1) // 2):
            temp = np.zeros(adjM.shape, dtype=bool)
            temp[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])] = True
            temp[np.ix_(type_mask == metapath[i + 1], type_mask == metapath[i])] = True
            mask = np.logical_or(mask, temp)

        partial_adjM = adjM * mask
        partial_g_nx = nx.from_numpy_matrix(partial_adjM.astype(int))
        
        # We just got the metapath-specific adjacency matrix, time to
        # find paths from source to target using this adjacency matrix.
        metapath_neighbor_pairs = {}
        for source in (type_mask == metapath[0]).nonzero()[0]:
            #print("\nChecking source (node idx): " + str(source) + "...")
            current_node = source
            seed_path = [[current_node]]
            all_full_paths = bfs_helper(partial_adjM, type_mask, metapath, seed_path, d)
            for full_path in all_full_paths:
                source_target_pair = (full_path[0], full_path[-1])
                all_paths = metapath_neighbor_pairs.get(source_target_pair, [])
                all_paths.append(full_path)
                metapath_neighbor_pairs[source_target_pair] = all_paths
        outs.append(metapath_neighbor_pairs)
    return outs

def get_networkx_graph(neighbor_pairs, type_mask, ctr_ntype):
    all_relevant_node_indices = []
    all_idx_mappings = []
    for metapaths in neighbor_pairs:
        relevant_node_indices_hist = {}
        for source_target_pair in metapaths.keys():
            relevant_node_indices_hist[source_target_pair[0]] = \
                relevant_node_indices_hist.get(source_target_pair[0], 1)
            relevant_node_indices_hist[source_target_pair[1]] = \
            relevant_node_indices_hist.get(source_target_pair[1], 1)
        indices = []
        for relevant_node in relevant_node_indices_hist.keys():
            indices.append(relevant_node)
        indices.sort()
        # indices = np.where(type_mask == ctr_ntype)[0]
        idx_mapping = {}
        for i, idx in enumerate(indices):
            idx_mapping[idx] = i
        all_relevant_node_indices.append(indices)
        all_idx_mappings.append(idx_mapping)
    G_list = []
    for i, metapaths in enumerate(neighbor_pairs):
        # print("\nget_networkx_graph(): checking metapaths: " + str(metapaths))
        edge_count = 0
        sorted_metapaths = sorted(metapaths.items())
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(len(all_relevant_node_indices[i])))
        for (src, dst), paths in sorted_metapaths:
            for path in paths:
                # The below line doesn't make sense for path length > 3
                # print("get_networkx_graph(): adding edge ({}, {}), a.k.a ({}, {})...".format(src, dst, all_idx_mappings[i][src], all_idx_mappings[i][dst]))
                G.add_edge(all_idx_mappings[i][src], all_idx_mappings[i][dst])
                """
                print("path:")
                print(path)
                for i in range(len(path) - 1):
                    print("nodes: ")
                    print(path[i], path[i+1])
                    G.add_edge(idx_mapping[path[i]], idx_mapping[path[i + 1]])
                    edge_count += 1
                """
                
        G_list.append(G)
    return G_list

def get_edge_metapath_idx_array(neighbor_pairs):
    # print("\nget_edge_metapath_idx_array():")
    # print("neighbor_pairs: " + str(neighbor_pairs))
    all_edge_metapath_idx_array = []
    for metapath_neighbor_pairs in neighbor_pairs:
        sorted_metapath_neighbor_pairs = sorted(metapath_neighbor_pairs.items())
        edge_metapath_idx_array = []
        for _, paths in sorted_metapath_neighbor_pairs:
            # print("paths: " + str(paths))
            edge_metapath_idx_array.extend(paths)
        edge_metapath_idx_array = np.array(edge_metapath_idx_array, dtype=int)
        all_edge_metapath_idx_array.append(edge_metapath_idx_array)
    # print(all_edge_metapath_idx_array)
    return all_edge_metapath_idx_array


"""
@param adjM: a (2, 2) numpy array (adjacency matrix)
@param type_mask: a list, where list[i] = node type of node i
Returns a list of lists (by start node type) metapaths, like [[(0, 1, 2)], [(1, 4, 3)], ...]
"""
def profile_metapaths(adjM, type_mask, num_node_types):
    # print("adjM")
    # print(adjM)
    col_sums = adjM.sum(axis=0)
    adjM_normalized = adjM / col_sums
    adjM_normalized = np.nan_to_num(adjM_normalized, nan=0.0)
    # print("adjM_normalized")
    # print(adjM_normalized)

    # sanity check to make sure depth-2 is possible:
    _adj_row_summed = adjM.sum(axis=1)
    possible_1hop_neighbor_indices = np.where(_adj_row_summed != 0)[0]
    possibility_sum = 0
    for p1hni in possible_1hop_neighbor_indices:
        possibility_sum += col_sums[p1hni]
    if (possibility_sum == 0):
        return
    
    
    # compose random walks
    num_nodes = adjM.shape[0]
    random_walks_nodes = []
    k = 50
    d = 2
    i = 0
    while i < k:
        go_to_next_iteration = False
        random_walk = []
        root_node = np.random.randint(0, num_nodes)
        random_walk.append(root_node)
        current_node = root_node
        for j in range(d):
            choices = adjM_normalized[:, current_node]
            # uh oh.. no valid depth-2 path
            if choices.sum(axis=0) == 0:
                i -= 1
                go_to_next_iteration = True
                break
            choices = choices.reshape(num_nodes)
            current_node = np.random.choice(np.arange(0, num_nodes, step=1), 1, p=choices)
            random_walk.append(current_node[0])
        i += 1
        if (go_to_next_iteration):
            continue
        random_walks_nodes.append(random_walk)

    metapaths_histogram = {}
    for i in range(num_node_types):
        metapaths_histogram[i] = {}

    # print(metapaths_histogram)
    # print("RANDOM_WALKS_NODES:")
    # print(random_walks_nodes)
    for random_walk in random_walks_nodes:
        metapath_tuple = (type_mask[random_walk[0]], \
                          type_mask[random_walk[1]], \
                          type_mask[random_walk[2]])
        metapaths_histogram[type_mask[random_walk[2]]][metapath_tuple] = \
                            metapaths_histogram[type_mask[random_walk[2]]].get(metapath_tuple, 1)

    # convert metapaths_histogram to list
    metapaths_list = []
    for i in range(num_node_types):
        metapaths_list.append([])
        for metapath_tuple in metapaths_histogram[i].keys():
            metapaths_list[i].append(metapath_tuple)
    
    return metapaths_list

"""
@param df_nodes: df of the nodes of 1 graph
@param df_edges: df of the edges of 1 graph (unused)
@param num_node_types: num node types
@graph_num: integer (1-indexed) representing graph number as per the data csv
"""
def preprocess(df_nodes, df_edges, train_graph, num_node_types=5, graph_num=10):
    print("\nPreprocessing graph number {}...".format(graph_num))
    adjM = train_graph.adjacency_matrix().to_dense().numpy()
    num_nodes = train_graph.number_of_nodes()
    type_mask = get_type_mask(df_nodes)
    print("\ntype_mask:")
    print(type_mask)
    expected_metapaths = profile_metapaths(adjM, type_mask, num_node_types=5)
    #expected_metapaths = [
    #]
    print("EXPECTED_METAPATHS:")
    print(expected_metapaths)
    if (expected_metapaths == None):
        return
    # Create directories here
    save_prefix = "/scratch/jinjtan/ryan-exploration/MAGNN/data/casesim_preprocessed/"
    for i in range(num_node_types):
        pathlib.Path(save_prefix + '{}/'.format(graph_num) + '{}'.format(i)).mkdir(parents=True, exist_ok=True)

        
    # relevant nodes features type_mask
    # objective, get list of indices of indices to save
    # if metapaths of type (1) = (1, 0, 0), (2, 3, 2), (2, 4, 3), want to save (1, 2, 0, 3)
    # (all sources and targets)
    all_relevant_nodes_indices = []
    # Get metapath based neighbor pairs
    for i in range(num_node_types):
        neighbor_pairs = get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths[i])
        print("\nneighbor_pairs:")
        print(neighbor_pairs)
        G_list = get_networkx_graph(neighbor_pairs, type_mask, i)

        # Ryan
        # keep track of which nodes would like to keep (sources and dest node IDs of
        # metapath instances starting from node of current node type)
        relevant_nodes_indices_hist = {}

        # save data
        # format: save_prefix/graph_num/node_type/1-0-1.adjlist
        for G, metapath in zip(G_list, expected_metapaths[i]):
            nx.write_adjlist(G, save_prefix + '{}/'.format(graph_num) + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist')
            
        # node indices of edge metapaths
        # format: save_prefix/graph_num/node_type/1-0-1_idx.npy
        all_edge_metapath_idx_array = get_edge_metapath_idx_array(neighbor_pairs)
        for metapath, edge_metapath_idx_array in zip(expected_metapaths[i], all_edge_metapath_idx_array):
            np.save(save_prefix + '{}/'.format(graph_num) + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)

    # save data
    # all node adjacency matrix
    df_no_original_graph_col = df_nodes.loc[:, df_nodes.columns != 'original_graph']
    full_X_dense = df_no_original_graph_col.to_numpy()
    full_X = scipy.sparse.csr_matrix(full_X_dense)

    # save the graph's adjM
    # format: save_prefix/graph_num/adjM.npz
    scipy.sparse.save_npz(save_prefix + '{}/'.format(graph_num) + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
    for i in range(num_node_types):
        scipy.sparse.save_npz(save_prefix + '{}/'.format(graph_num) + 'features_{}.npz'.format(i), full_X[np.where(type_mask == i)[0]])

    # save the type_mask (all node types)
    np.save(save_prefix + '{}/'.format(graph_num) + 'node_types.npy', type_mask)

    # save the graph labels (do a function call to ___)
    # do train test split somewhere else.

def diagnostic_profile_all_graphs(df_nodes_grouped, df_edges, list_of_graphs, num_node_types):
    for i in range(len(list_of_graphs)):
        graph_num = i + 1
        preprocess(df_nodes_grouped.get_group(graph_num), df_edges, list_of_graphs[i], \
                   num_node_types=5, graph_num=graph_num)

def path_to_metapath_tuple(path_string, start, end):
    _s = path_string.rfind(start)
    _d = path_string.rfind(end)
    metapath_type_string = path_string[_s + 1: _d]
    metapath_type_list = metapath_type_string.split('-')
    for i in range(len(metapath_type_list)):
        metapath_type_list[i] = int(metapath_type_list[i]) # typecasting
    metapath_type_tuple = tuple(metapath_type_list)
    return metapath_type_tuple
        
def load_casesim_data(prefix='/scratch/jinjtan/ryan-exploration/MAGNN/data/casesim_preprocessed/', \
                      graphs=[10], num_node_types=5):

    res_by_graph_num = {}

    for graph_num in graphs:
        Gs = []
        idxs = []
        features = []
        for node_type in range(num_node_types):

            # Load all adjlists
            adjlist_paths = [adjlist_path for adjlist_path in pathlib.Path(prefix + '{}/'.format(graph_num) + '{}/'.format(node_type)).iterdir() if adjlist_path.suffix == '.adjlist']
            # Load all node_idx_paths
            node_idx_paths = [node_idx_path for node_idx_path in pathlib.Path(prefix + '{}/'.format(graph_num) + '{}/'.format(node_type)).iterdir() if node_idx_path.suffix == '.npy']

            node_type_specific_Gs = []
            node_type_specific_idxs = []
            
            # Convert each list to dict. key = metapath tuple, value = posix path
            adjlist_paths_dict = {}
            node_idx_paths_dict = {}
            for (a_path, node_idx_path) in zip(adjlist_paths, node_idx_paths):
                metapath_type_tuple = path_to_metapath_tuple(str(a_path), start='/', end='.')
                assert metapath_type_tuple == path_to_metapath_tuple(str(node_idx_path), start='/', end='_')
                adjlist_paths_dict[metapath_type_tuple] = a_path
                node_idx_paths_dict[metapath_type_tuple] = node_idx_path
                
            # Create dictionary of nx_graphs by metapath type
            # u_metapath = universal metapath; metapath that features at least once in some graph
            for u_metapath in metapaths[node_type]:
                # u_metapath = [(0,2,0), (0,2,1), (0,3,0)] for example
                if u_metapath not in adjlist_paths_dict.keys():
                    node_type_specific_Gs.append(None)
                    node_type_specific_idxs.append(np.array([]))
                else:
                    node_type_specific_Gs.append(nx.read_adjlist(str(adjlist_paths_dict[u_metapath]), create_using=nx.MultiDiGraph))
                    node_type_specific_idxs.append(np.load(str(node_idx_paths_dict[u_metapath])))
            Gs.append(node_type_specific_Gs)
            idxs.append(node_type_specific_idxs)

            features.append(scipy.sparse.load_npz(prefix + '{}/'.format(graph_num) + 'features_' + '{}'.format(node_type) + '.npz'))

        adjM = scipy.sparse.load_npz(prefix + '{}/'.format(graph_num) + 'adjM.npz')
        type_mask = np.load(prefix + '{}/'.format(graph_num) + 'node_types.npy')

        # print("idxs: " + str(idxs))
        res_dict = {
            'Gs': Gs,
            'idxs': idxs,
            'features': features,
            'adjM': adjM,
            'type_mask': type_mask,
        }
        res_by_graph_num[graph_num] = res_dict
    return res_by_graph_num
    

# df_nodes, df_edges = loadData("/scratch/jinjtan/ryan-exploration/garbage/exp_nodes.csv", "/scratch/jinjtan/ryan-exploration/garbage/exp_edges.csv")

df_nodes, df_edges = loadData("/scratch/jinjtan/case-similarity-master/data/graphsage/nodes.csv", \
                              "/scratch/jinjtan/case-similarity-master/data/graphsage/edges.csv")

print(df_nodes)
train_graphs = transform_graph(df_nodes, df_edges)
df_nodes_grouped = df_nodes.groupby(df_nodes["original_graph"])

# diagnostic_profile_all_graphs(df_nodes_grouped, df_edges, train_graphs, num_node_types=5) 

#graphs_to_preprocess = [i for i in range(1, 106)]
#blacklisted_graphs = [4, 5, 6, 7, 9, 18, 25, 31, 42, 58, 67, 81, 102]
#for b in blacklisted_graphs:
#    graphs_to_preprocess.remove(b)

#for gnum in graphs_to_preprocess:
#    preprocess(df_nodes_grouped.get_group(gnum), df_edges, train_graphs[gnum], num_node_types=5, graph_num=gnum)
#list_of_graphs = load_casesim_data(graphs=graphs_to_preprocess)
# print(list_of_graphs)

# ground_truth = load_graph_ground_truth()
