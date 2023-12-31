
import os
import json
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from utils.refs import sem_ref, joint_ref

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
        mapping = {i: order[i] for i in range(N)}
        mapping.update({i: i for i in range(len(nodes), self.hparams.K)})
        graph_permuted = self._reorder_nodes(graph, mapping)
        nodes_permuted = nodes[order, :]
        graph_permuted['parents'] = graph['parents'][order]
        return graph_permuted, nodes_permuted

    def _build_graph(self, nodes):
        '''
        Function to build graph from the node list.
        
        Args:
            nodes: list of nodes
            K: size of the adjacency matrix
        Returns:
            adj: adjacency matrix, records the 1-ring relationship (parent+children) between nodes
            edge_list: list of edges, for visualization
            root_id: root node id, for visualization  
        '''
        K = self.hparams.K
        adj = np.zeros((K, K))
        root_id = 0
        parents = []
        for node in nodes:
            # 1-ring relationship
            if node['parent'] != -1:
                adj[node['id'], node['parent']] = 1
                parents.append(node['parent'])
            else:
                root_id = node['id']
                parents.append(-1)
            for child_id in node['children']:
                adj[node['id'], child_id] = 1 
              
        adj = adj.astype(np.float32)
        return {
            'adj': adj,
            'root': root_id,
            'parents': np.array(parents)
        }

    def _reorder_nodes(self, graph, mapping):
        '''
        Function to reorder nodes in the graph and 
        update the adjacency matrix, edge list, and root node.

        Args:
            graph: a dictionary containing the adjacency matrix, edge list, and root node
            mapping: a dictionary mapping the old node id to the new node id
        Returns:
            new_graph: a dictionary containing the updated adjacency matrix, edge list, and root node
        '''
        G = nx.from_numpy_array(graph['adj'], create_using=nx.Graph)
        G_ = nx.relabel_nodes(G, mapping)
        new_adj = nx.adjacency_matrix(G_, G.nodes).todense()
        return {
            'adj': new_adj.astype(np.float32),
            'root': mapping[graph['root']],
            'parents': graph['parents'],
        }
    
    def _prepare_node_data(self, node):
        # semantic label
        label = np.array([sem_ref['fwd'][node['name']]]) / 5. - 0.8 # (1,), range from -0.8 to 0.8
        # joint type
        joint_type = np.array([joint_ref['fwd'][node['joint']['type']] / 5.]) - 0.5 # (1,), range from -0.8 to 0.8
        # aabb
        aabb_center = np.array(node['aabb']['center']) # (3,), range from -1 to 1
        aabb_size = np.array(node['aabb']['size']) # (3,), range from -1 to 1
        aabb_max = aabb_center + aabb_size / 2
        aabb_min = aabb_center - aabb_size / 2
        # joint axis and range
        if node['joint']['type'] == 'fixed':
            axis_dir = np.zeros((3,))
            axis_ori = np.zeros((3,))
            joint_range = np.zeros((2,))
        else:
            if node['joint']['type'] == 'revolute' or node['joint']['type'] == 'continuous':
                joint_range = np.array([node['joint']['range'][1]]) / 360. 
                joint_range = np.concatenate([joint_range, np.zeros((1,))], axis=0) # (2,) 
            elif node['joint']['type'] == 'prismatic' or node['joint']['type'] == 'screw':
                joint_range = np.array([node['joint']['range'][1]]) 
                joint_range = np.concatenate([np.zeros((1,)), joint_range], axis=0) # (2,) 
            axis_dir = np.array(node['joint']['axis']['direction']) * 0.7 # (3,), range from -0.7 to 0.7
            # make sure the axis is pointing to the positive direction
            if np.sum(axis_dir > 0) < np.sum(-axis_dir > 0): 
                axis_dir = -axis_dir 
                joint_range = -joint_range
            axis_ori = np.array(node['joint']['axis']['origin']) # (3,), range from -1 to 1
        node_data = np.concatenate([aabb_max, aabb_min, joint_type.repeat(6), axis_dir, axis_ori, joint_range.repeat(3), label.repeat(6)], axis=0)
        return node_data
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.model_ids)

