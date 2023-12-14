
import torch
import systems
import numpy as np
import lightning.pytorch as pl
import torch.nn.functional as F
from systems.base import BaseSystem
from diffusers.optimization import get_cosine_schedule_with_warmup

@systems.register('cage')
class CAGESystem(BaseSystem):
    def configure_optimizers(self):
        n_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = n_steps / self.trainer.max_epochs
        optim = torch.optim.AdamW(self.model.parameters(), **self.hparams.optimizer.args)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=20*steps_per_epoch, # warm up 20 epochs
            num_training_steps=n_steps,
        )
        return {'optimizer': optim, 'lr_scheduler': lr_scheduler}
    
    def forward(self, x, cat, t, pad_mask=None, graph_mask=None, attr_mask=None):
        return self.model(x, cat, t, pad_mask, graph_mask, attr_mask)

    def training_step(self, batch, batch_idx):
        x, c = batch
        
        cat = c['cat']
        key_pad_mask = c['key_pad_mask']
        graph_mask = c['adj_mask']
        attr_mask = c['attr_mask']

        # construct loss mask
        index_tensor = torch.arange(x.shape[1], device=x.device).unsqueeze(0) # (1, N)
        len_nodes = c['n_nodes'] * 5 # five attributes
        mask = index_tensor < len_nodes.unsqueeze(-1)  # This uses broadcasting
        mask_padding = mask.unsqueeze(-1).expand_as(x)

        # repeat the input for multiple samples
        n_repeat = 8
        x = x.repeat(n_repeat, 1, 1)
        cat = cat.repeat(n_repeat)
        key_pad_mask = key_pad_mask.repeat(n_repeat, 1, 1)
        graph_mask = graph_mask.repeat(n_repeat, 1, 1)
        attr_mask = attr_mask.repeat(n_repeat, 1, 1)
        mask_padding = mask_padding.repeat(n_repeat, 1, 1) 

        # Sample Gaussian noise
        noise = torch.randn(x.shape, device=x.device)
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (x.shape[0],),
            device=x.device
        ).long()
        # Add Gaussian noise to the input
        noisy_x = self.scheduler.add_noise(x, noise, timesteps)
        # Predict the noise given the noisy input
        noise_pred = self(noisy_x, cat, timesteps, key_pad_mask, graph_mask, attr_mask)
        
        loss = F.mse_loss(noise_pred * mask_padding, noise * mask_padding)

        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, c = batch
        cat = c['cat']
        key_pad_mask = c['key_pad_mask']
        graph_mask = c['adj_mask']
        attr_mask = c['attr_mask']

        # set scheduler
        n_t = 100
        self.scheduler.set_timesteps(n_t) 

        noisy_x = torch.randn(x.shape).to(x.device)
        for t in self.scheduler.timesteps:
            timesteps = torch.tensor([t], device=x.device)
            noise_pred = self(noisy_x, cat, timesteps, key_pad_mask, graph_mask, attr_mask)
            noisy_x = self.scheduler.step(noise_pred, t, noisy_x).prev_sample

        self.save_val_img(noisy_x, x, c) # save the image

    def predict_step(self, batch, batch_idx):
        mode = self.hparams.datamodule.pred_mode
        if mode == 'uncond':
            self.pred_uncond(batch, batch_idx)
        elif mode == 'cond_graph' or mode == 'ood':
            self.pred_cond_graph(batch, batch_idx)
        elif mode == 'cond_n_nodes':
            raise NotImplementedError
        elif mode.startswith('cond_'):
            self.pred_cond_attr(batch, batch_idx)
        else:
            raise NotImplementedError(f'Unknown pred_mode: {mode}')
    
    def pred_cond_attr(self, batch, batch_idx):
        x, c = batch
        cat = c['cat']
        key_pad_mask = c['key_pad_mask']
        graph_mask = c['adj_mask']
        attr_mask = c['attr_mask']
        # set scheduler
        n_t = 100
        self.scheduler.set_timesteps(n_t)
        # repeat the input for multiple samples
        n_repeat = self.hparams.datamodule.pred_n_samples
        x = x.repeat(n_repeat, 1, 1)
        cat = cat.repeat(n_repeat)
        key_pad_mask = key_pad_mask.repeat(n_repeat, 1, 1)
        graph_mask = graph_mask.repeat(n_repeat, 1, 1)
        attr_mask = attr_mask.repeat(n_repeat, 1, 1)
        # masking indices
        mode = self.hparams.datamodule.pred_mode
        if mode == 'cond_box':
            indices = torch.arange(0, 32*5, step=5) # 0 is box
        elif mode == 'cond_type':
            indices = torch.arange(1, 32*5, step=5) # 1 is type
        elif mode == 'cond_axis':
            indices = torch.arange(2, 32*5, step=5) # 2 is axis
        elif mode == 'cond_axis_type':
            indices = torch.cat([torch.arange(1, 32*5, step=5) , torch.arange(2, 32*5, step=5)], dim=0)
        else:
            raise NotImplementedError
        # init the noisy input
        noisy_x = torch.randn(x.shape, device=x.device)
        for t in self.scheduler.timesteps:
            noise = torch.randn(x.shape, device=x.device)
            timesteps = torch.tensor([t], device=x.device)
            gt_noised = self.scheduler.add_noise(x, noise, timesteps)
            noisy_x[:, indices, :] = gt_noised[:, indices, :]
            noise_pred = self(noisy_x, cat, timesteps, key_pad_mask, graph_mask, attr_mask)
            noisy_x = self.scheduler.step(noise_pred, t, noisy_x).prev_sample
        
        masked_pred = noisy_x.clone()
        masked_pred[:, indices, :] = x[:, indices, :]
        self.save_pred_cond_attr(masked_pred, batch)

    def pred_cond_graph(self, batch, batch_idx):
        x, c = batch
        cat = c['cat']
        key_pad_mask = c['key_pad_mask']
        graph_mask = c['adj_mask']
        attr_mask = c['attr_mask']

        n_t = 100
        # repeat the input for multiple samples
        n_repeat = self.hparams.datamodule.pred_n_samples
        x = x.repeat(n_repeat, 1, 1)
        cat = cat.repeat(n_repeat)
        key_pad_mask = key_pad_mask.repeat(n_repeat, 1, 1)
        graph_mask = graph_mask.repeat(n_repeat, 1, 1)
        attr_mask = attr_mask.repeat(n_repeat, 1, 1)

        noisy_x = torch.randn(x.shape, device=x.device)
        self.scheduler.set_timesteps(n_t)

        for t in self.scheduler.timesteps:
            timesteps = torch.tensor([t], device=x.device)
            noise_pred = self(noisy_x, cat, timesteps, key_pad_mask, graph_mask, attr_mask)
            noisy_x = self.scheduler.step(noise_pred, t, noisy_x).prev_sample
        
        self.save_pred_cond_graph(noisy_x, batch)

    def pred_uncond(self, batch, batch_idx):
        x, c = batch
        cat = c['cat']
        key_pad_mask = c['key_pad_mask']
        graph_mask = c['adj_mask']
        attr_mask = c['attr_mask']

        n_t = 100 # ddpm, total denoising steps

        noisy_x = torch.randn(x.shape, device=x.device)
        self.scheduler.set_timesteps(n_t)

        for t in self.scheduler.timesteps:
            timesteps = torch.tensor([t], device=x.device)
            noise_pred = self(noisy_x, cat, timesteps, key_pad_mask, graph_mask, attr_mask)
            noisy_x = self.scheduler.step(noise_pred, t, noisy_x).prev_sample
        self.save_pred_uncond(noisy_x, batch, batch_idx)

    
    