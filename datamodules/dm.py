
import os
import sys
import json
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datamodules
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from datamodules.datasets import TrainDataset, IDPredDataset, OODPredDataset


@datamodules.register('cage')
class CAGEDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

    def prepare_data(self):
        pass

    def _prepare_predict(self):
        if self.hparams.pred_mode == 'uncond':
            # load all models in the testing set
            split_path = self.hparams.split_file
            splits = json.load(open(split_path, 'r'))
            model_ids = splits['test']
            model_ids += model_ids * 5 # generte samples 5 times larger than the testing set
            self.pred_dataset = IDPredDataset(self.hparams, model_ids)
            self.pred_size=len(model_ids)
        elif self.hparams.pred_mode == 'cond_graph':
            self.pred_size = 1
            hb_path = self.hparams.hash_table
            hb = json.load(open(hb_path, 'r'))
            model_ids = []
            for cat in hb.keys():
                for h in hb[cat]:
                    model_ids.append(hb[cat][h][0])
            self.pred_dataset = IDPredDataset(self.hparams, model_ids)
        else: # condition on node attributes
            self.pred_size = 1
            hb_path = self.hparams.hash_table
            hb = json.load(open(hb_path, 'r'))
            model_ids = []
            n_examples = 8
            for cat in hb.keys():
                for h in hb[cat]:
                    if len(hb[cat][h]) < n_examples:
                        model_ids += [i for i in hb[cat][h]]
                    else:
                        model_ids += [i for i in hb[cat][h][:n_examples]]
            self.pred_dataset = IDPredDataset(self.hparams, model_ids)
        
    def setup(self, stage=None):
        # called on every process in DDP
        if stage == 'fit' or stage is None or stage == 'validate':
            hp = copy.deepcopy(self.hparams)
            hp['augment'] = False
            val_ids = []
            splits = json.load(open(self.hparams.split_file, 'r'))
            train_ids = splits['train']
            val_idx = np.random.choice(np.arange(len(train_ids)), 10)
            for i in val_idx:
                val_ids.append(train_ids[i])
                
            self.train_ids = train_ids
            self.val_ids = val_ids
        
        if stage == 'fit' or stage is None:
            self.train_dataset = TrainDataset(self.hparams, self.train_ids)
            self.val_dataset = TrainDataset(hp, self.val_ids)
        elif stage == 'validate':
            self.val_dataset = TrainDataset(hp, self.val_ids)
        elif stage == 'predict':
            mode = self.hparams.pred_mode
            if mode == 'ood':
                # load out-of-distribution graphs (manually constructed)
                self.pred_dataset = OODPredDataset(self.hparams, 'datamodules/ood_graphs.json')
                self.pred_size = 1
            else:
                self._prepare_predict()
                
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.train_ids), shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.pred_size, shuffle=False)
