
import json
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from datamodules.base import BaseDataset
from utils.refs import cat_ref

class TrainDataset(BaseDataset):
    def __init__(self, hparams, model_ids):    
        self.hparams = hparams
        self.model_ids = model_ids
        self.files = self._cache_data()
    
    def __getitem__(self, idx):

        file = self.files[idx]
        tree = file['diffuse_tree']
        K = self.hparams.K # max number of nodes
        cond = {} # conditional information and axillary data
        cond['parents'] = np.zeros(K).astype(np.int8)

        # object category
        cond['cat'] = cat_ref[file['meta']['obj_cat']]

        # prepare node data
        nodes = []
        for node in tree:
            node_data = self._prepare_node_data(node) # (30,)     
            nodes.append(node_data) 
        nodes = np.array(nodes)
        n_nodes = len(nodes)

        # prepare graph
        graph = self._build_graph(tree)
        if self.hparams.augment:
            graph, nodes = self._random_permute(graph, nodes)

        # pad the nodes to K with empty nodes
        empty_node = np.zeros((nodes[0].shape[0],))
        data = np.concatenate([nodes, [empty_node] * (K - n_nodes)], axis=0).astype(np.float32) # (K, 30)
        data = data.reshape(K*5, 6) # (K * n_attr, 6)

        # record the original graph
        cond['adj'] = graph['adj']
        cond['root'] = graph['root']
        cond['parents'][:n_nodes] = graph['parents']
        cond['n_nodes'] = n_nodes

        # attr mask (for Local Attention)
        attr_mask = np.eye(K, K)
        attr_mask = attr_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['attr_mask'] = attr_mask.astype(np.float32)

        # key padding mask (for Global Attention)
        pad_mask = np.zeros((K*5, K*5))
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


class IDPredDataset(BaseDataset):
    '''In-distribution prediction dataset'''
    def __init__(self, hparams, model_ids):
        self.hparams = hparams
        self.hparams.augment = False # turn off node permutation
        self.model_ids = model_ids
        self.files = self._cache_data()
            
    def _load_data(self, idx):
        file = self.files[idx]
        tree = file['diffuse_tree']
        K = self.hparams.K # max number of nodes
        cond = {} # conditional information and axillary data
        cond['parents'] = np.zeros(K)

        # object category
        cond['cat'] = cat_ref[file['meta']['obj_cat']]

        # prepare node data
        nodes = []
        for node in tree:
            node_data = self._prepare_node_data(node) # (30,)     
            nodes.append(node_data) 
        nodes = np.array(nodes)
        n_nodes = len(nodes)

        # prepare graph
        graph = self._build_graph(tree)
        if self.hparams.augment:
            graph, nodes = self._random_permute(graph, nodes)

        # pad the nodes to K with empty nodes
        empty_node = np.zeros((nodes[0].shape[0],))
        data = np.concatenate([nodes, [empty_node] * (K - n_nodes)], axis=0).astype(np.float32) # (K, 30)
        data = data.reshape(K*5, 6) # (K * n_attr, 6)

        # record the original graph
        cond['adj'] = graph['adj']
        cond['root'] = graph['root']
        cond['parents'][:n_nodes] = graph['parents']
        cond['n_nodes'] = n_nodes

        # attr mask (for Local Attention)
        attr_mask = np.eye(K, K)
        attr_mask = attr_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['attr_mask'] = attr_mask.astype(np.float32)

        # key padding mask (for Global Attention)
        pad_mask = np.zeros((K*5, K*5))
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

    def _load_graph_cat(self, idx):
        file = self.files[idx]
        K = self.hparams.K # max number of nodes
        data = np.zeros((K * 5, 6))
        tree = file['diffuse_tree']
        n_nodes = len(tree)
        cond = {}
        cond['parents'] = np.zeros(K)

        # object category
        cond['cat'] = cat_ref[file['meta']['obj_cat']]

        # record the original graph
        graph = self._build_graph(tree)
        cond['adj'] = graph['adj']
        cond['root'] = graph['root']
        cond['parents'][:n_nodes] = graph['parents']
        cond['n_nodes'] = n_nodes

        # key padding mask (for Global Attention)
        pad_mask = np.zeros((K*5, K*5))
        pad_mask[:, :n_nodes*5] = 1
        cond['key_pad_mask'] = pad_mask.astype(np.float32)

        # adj mask (for Graph Relation Attention)
        adj_mask = cond['adj'].copy()
        adj_mask = adj_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['adj_mask'] = adj_mask.astype(np.float32)

        # attr mask (for Local Attention)
        attr_mask = np.eye(K, K)
        attr_mask = attr_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['attr_mask'] = attr_mask.astype(np.float32)

        # axillary info
        cond['name'] = self.model_ids[idx]
        cond['obj_cat'] = file['meta']['obj_cat']
        cond['tree_hash'] = file['meta']['tree_hash']

        return data, cond

    def __getitem__(self, idx):
        if self.hparams.pred_mode == 'uncond' or self.hparams.pred_mode == 'cond_graph':
            data, cond = self._load_graph_cat(idx)
        else:
            raise NotImplementedError
            
        return data, cond 
    
    def __len__(self):
        return len(self.model_ids)

