
import os
import json
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from utils.refs import sem_ref, joint_ref, cat_ref

class BaseDataset(Dataset):
    def __init__(self, hparams):
        super().__init__()
    
    def _cache_data(self):
        '''
        Function to cache data from json files.
        '''
        data_root = self.hparams.root
        files = []
        for model_id in self.model_ids:
            path = os.path.join(data_root, model_id, f'train.json')
            with open(path, 'r') as f:
                file = json.load(f)
            files.append(file)
        return files
    
    def _random_permute(self, graph, nodes):
        '''
        Function to randomly permute the nodes and update the graph and node attribute info.

        Args:
            graph: a dictionary containing the adjacency matrix, edge list, and root node
            nodes: a list of nodes
        Returns:
            graph_permuted: a dictionary containing the updated adjacency matrix, edge list, and root node
            nodes_permuted: a list of permuted nodes
        '''
        N = len(nodes)
        order = np.random.permutation(N)
        graph_permuted = self._reorder_nodes(graph, order)
        nodes_permuted = nodes[order, :]
        return graph_permuted, nodes_permuted

    def _build_graph(self, nodes):
        '''
        Function to build graph from the node list.
        
        Args:
            nodes: list of nodes
        Returns:
            adj: adjacency matrix, records the 1-ring relationship (parent+children) between nodes
            edge_list: list of edges, for visualization
            root_id: root node id, for visualization  
        '''
        K = self.hparams.K
        adj = np.zeros((K, K), dtype=np.float32)
        root_id = 0
        parents = []
        for node in nodes:
            # 1-ring relationship
            if node['parent'] != -1:
                adj[node['id'], node['parent']] = 1
                parents.append(node['parent'])
            else:
                adj[node['id'], node['id']] = 1
                root_id = node['id']
                parents.append(-1)
            for child_id in node['children']:
                adj[node['id'], child_id] = 1 

        return {
            'adj': adj,
            'root': root_id,
            'parents': np.array(parents, dtype=np.int8)
        }

    def _reorder_nodes(self, graph, order):
        '''
        Function to reorder nodes in the graph and 
        update the adjacency matrix, edge list, and root node.

        Args:
            graph: a dictionary containing the adjacency matrix, edge list, and root node
            order: a list of indices for reordering
        Returns:
            new_graph: a dictionary containing the updated adjacency matrix, edge list, and root node
        '''
        N = len(order)
        mapping = {i: order[i] for i in range(N)}
        mapping.update({i: i for i in range(N, self.hparams.K)})
        G = nx.from_numpy_array(graph['adj'], create_using=nx.Graph)
        G_ = nx.relabel_nodes(G, mapping)
        new_adj = nx.adjacency_matrix(G_, G.nodes).todense()
        return {
            'adj': new_adj.astype(np.float32),
            'root': mapping[graph['root']],
            'parents': graph['parents'][order],
        }
    
    def _prepare_node_data(self, node):
        # semantic label
        label = np.array([sem_ref['fwd'][node['name']]], dtype=np.float32) / 5. - 0.8 # (1,), range from -0.8 to 0.8
        # joint type
        joint_type = np.array([joint_ref['fwd'][node['joint']['type']] / 5.], dtype=np.float32) - 0.5 # (1,), range from -0.8 to 0.8
        # aabb
        aabb_center = np.array(node['aabb']['center'], dtype=np.float32) # (3,), range from -1 to 1
        aabb_size = np.array(node['aabb']['size'], dtype=np.float32) # (3,), range from -1 to 1
        aabb_max = aabb_center + aabb_size / 2
        aabb_min = aabb_center - aabb_size / 2
        # joint axis and range
        if node['joint']['type'] == 'fixed':
            axis_dir = np.zeros((3,), dtype=np.float32)
            axis_ori = np.zeros((3,), dtype=np.float32)
            joint_range = np.zeros((2,), dtype=np.float32)
        else:
            if node['joint']['type'] == 'revolute' or node['joint']['type'] == 'continuous':
                joint_range = np.array([node['joint']['range'][1]], dtype=np.float32) / 360. 
                joint_range = np.concatenate([joint_range, np.zeros((1,))], axis=0) # (2,) 
            elif node['joint']['type'] == 'prismatic' or node['joint']['type'] == 'screw':
                joint_range = np.array([node['joint']['range'][1]], dtype=np.float32) 
                joint_range = np.concatenate([np.zeros((1,)), joint_range], axis=0) # (2,) 
            axis_dir = np.array(node['joint']['axis']['direction'], dtype=np.float32) * 0.7 # (3,), range from -0.7 to 0.7
            # make sure the axis is pointing to the positive direction
            if np.sum(axis_dir > 0) < np.sum(-axis_dir > 0): 
                axis_dir = -axis_dir 
                joint_range = -joint_range
            axis_ori = np.array(node['joint']['axis']['origin'], dtype=np.float32) # (3,), range from -1 to 1
        node_data = np.concatenate([aabb_max, aabb_min, joint_type.repeat(6), axis_dir, axis_ori, joint_range.repeat(3), label.repeat(6)], axis=0)
        return node_data
    
    def _prepare_item(self, idx):
        file = self.files[idx]
        tree = file['diffuse_tree']
        K = self.hparams.K # max number of nodes
        cond = {} # conditional information and axillary data
        cond['parents'] = np.zeros(K, dtype=np.int8)

        # object category
        cond['cat'] = cat_ref[file['meta']['obj_cat']]

        # prepare node data
        nodes = []
        for node in tree:
            node_data = self._prepare_node_data(node) # (30,)     
            nodes.append(node_data) 
        nodes = np.array(nodes, dtype=np.float32)
        n_nodes = len(nodes)

        # prepare graph
        graph = self._build_graph(tree)
        if self.hparams.augment:
            graph, nodes = self._random_permute(graph, nodes)

        # pad the nodes to K with empty nodes
        empty_node = np.zeros((nodes[0].shape[0],))
        data = np.concatenate([nodes, [empty_node] * (K - n_nodes)], axis=0, dtype=np.float32) # (K, 30)
        data = data.reshape(K*5, 6) # (K * n_attr, 6)

        # record the original graph
        cond['adj'] = graph['adj']
        cond['root'] = graph['root']
        cond['parents'][:n_nodes] = graph['parents']
        cond['n_nodes'] = n_nodes

        # attr mask (for Local Attention)
        attr_mask = np.eye(K, K, dtype=np.float32)
        attr_mask = attr_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['attr_mask'] = attr_mask

        # key padding mask (for Global Attention)
        pad_mask = np.zeros((K*5, K*5), dtype=np.float32)
        pad_mask[:, :n_nodes*5] = 1
        cond['key_pad_mask'] = pad_mask.astype(np.float32)

        # adj mask (for Graph Relation Attention)
        adj_mask = cond['adj'].copy()
        adj_mask = adj_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['adj_mask'] = adj_mask.astype(np.float32)

        # axillary info
        cond['name'] = self.model_ids[idx]
        cond['obj_cat'] = file['meta']['obj_cat']
        cond['tree_hash'] = file['meta']['tree_hash']
        return data, cond
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.model_ids)

