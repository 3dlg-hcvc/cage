
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
        data, cond = self._prepare_item(idx)
        return data, cond

class IDPredDataset(BaseDataset):
    '''In-distribution prediction dataset'''
    def __init__(self, hparams, model_ids):
        self.hparams = hparams
        self.hparams.augment = False # turn off node permutation
        self.model_ids = model_ids
        self.files = self._cache_data()

    def _load_graph_cat(self, idx):
        '''load graph and category only'''
        file = self.files[idx]
        K = self.hparams.K # max number of nodes
        data = np.zeros((K * 5, 6), dtype=np.float32)
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
        pad_mask = np.zeros((K*5, K*5), dtype=np.float32)
        pad_mask[:, :n_nodes*5] = 1
        cond['key_pad_mask'] = pad_mask

        # adj mask (for Graph Relation Attention)
        adj_mask = cond['adj'].copy()
        adj_mask = adj_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['adj_mask'] = adj_mask.astype(np.float32)

        # attr mask (for Local Attention)
        attr_mask = np.eye(K, K, dtype=np.float32)
        attr_mask = attr_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['attr_mask'] = attr_mask

        # axillary info
        cond['name'] = self.model_ids[idx]
        cond['obj_cat'] = file['meta']['obj_cat']
        cond['tree_hash'] = file['meta']['tree_hash']

        return data, cond

    def __getitem__(self, idx):
        if self.hparams.pred_mode == 'uncond' or self.hparams.pred_mode == 'cond_graph':
            data, cond = self._load_graph_cat(idx)
        else: # conditional on node attributes
            data, cond = self._prepare_item(idx)
            
        return data, cond 
    
    def __len__(self):
        return len(self.model_ids)

class OODPredDataset(Dataset):
    '''Out-of-distribution prediction dataset'''
    def __init__(self, hparams, ref_file):
        self.hparams = hparams
        ref = json.load(open(ref_file, 'r'))
        tree = ref['diffuse_tree']
        self.cats, self.adjs, self.adjs_plot, self.hashes = [], [], [], []
        self.num_nodes = []
        for cat in tree:
            for edges in tree[cat]:
                adj, adj_plot = self.get_adj(edges)
                h, n_nodes = self.get_hashcode(edges)
                self.hashes.append(h)
                self.adjs.append(adj)
                self.adjs_plot.append(adj_plot)
                self.cats.append(cat)
                self.num_nodes.append(n_nodes)
    
    def get_hashcode(self, edges):
        G = nx.DiGraph()
        G.add_edges_from(edges)
        hashcode = nx.weisfeiler_lehman_graph_hash(G)
        n_nodes = len(G.nodes)
        return hashcode, n_nodes
        
    def get_adj(self, edges):
        K = self.hparams.K
        adj = np.zeros((K, K))
        adj_plot = np.zeros((K, K))
        for edge in edges:
            adj[edge[0], edge[1]] = 1
            adj[edge[1], edge[0]] = 1
            adj_plot[edge[1], edge[0]] = 1
        adj[0][0] = 1
        adj_plot[0][0] = 1
        return adj.astype(np.float32), adj_plot.astype(np.float32)

    def __getitem__(self, idx):
        K = self.hparams.K
        adj = self.adjs[idx]
        cat = self.cats[idx]
        adj_plot = self.adjs_plot[idx]
        h = self.hashes[idx]
        n_nodes = self.num_nodes[idx]
        cond = {}
        cond['obj_cat'] = cat
        cond['cat'] = cat_ref[cat]
        cond['adj'] = adj
        cond['adj_plot'] = adj_plot
        cond['n_nodes'] = n_nodes
        cond['tree_hash'] = h
        # key padding mask
        pad_mask = np.zeros((K*5, K*5))
        pad_mask[:, :n_nodes*5] = 1
        cond['key_pad_mask'] = pad_mask.astype(np.float32)
        # adj mask
        adj_mask = cond['adj'].copy()
        adj_mask = adj_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['adj_mask'] = adj_mask.astype(np.float32)
        # attr mask
        attr_mask = np.eye(K, K)
        attr_mask = attr_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['attr_mask'] = attr_mask.astype(np.float32)
        data = np.zeros((K * 5, 6)).astype(np.float32)
        return data, cond
    
    def __len__(self):
        return len(self.cats)
