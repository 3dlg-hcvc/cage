
import torch
import models
from torch import nn
from models.utils import FinalLayer, PEmbeder, AAB

@models.register('denoiser')
class AABModel(nn.Module):
    '''
    Denoiser based on Attribute Attention Block (AAB)
    3 sequential attentions: local -> global -> graph
    '''
    def __init__(self, hparams):
        super(AABModel, self).__init__()
        self.hparams = hparams
        in_ch = hparams.in_ch
        attn_dim = hparams.attn_dim
        dropout = hparams.dropout
        n_head = hparams.n_head

        head_dim = attn_dim // n_head
        num_embeds_ada_norm = 6*attn_dim
        self.K = self.hparams.get('K', 32)
        
        self.x_embedding = nn.Linear(in_ch, attn_dim)
        self.pe_node = PEmbeder(self.K, attn_dim)
        self.pe_attr = PEmbeder(5, attn_dim)

        self.attn_layers = nn.ModuleList(
            [  # to do: refactor this block, customize the eps of layernorm if train with fp16
                AAB(dim=attn_dim, 
                    num_attention_heads=n_head,
                    attention_head_dim=head_dim,
                    dropout=dropout,
                    activation_fn="geglu",
                    num_embeds_ada_norm=num_embeds_ada_norm, 
                    attention_bias=False,
                    norm_elementwise_affine=True,
                    final_dropout=False, 
                    ) 
                for d in range(hparams.n_layers)
            ]
        )

        self.final_layer = FinalLayer(attn_dim, in_ch)
    
    def forward(self, x, cat, timesteps, key_padding_mask=None, graph_mask=None, attr_mask=None):
        # positional encoding for nodes and attributes
        idx_attr = torch.tensor([0,1,2,3,4], device=x.device).long().repeat(self.K)
        idx_node = torch.arange(self.K, device=x.device).long().repeat_interleave(5)
        x = self.pe_attr(self.pe_node(self.x_embedding(x), idx=idx_node), idx=idx_attr)
        # attention layers
        for attn_layer in self.attn_layers:
            x = attn_layer(hidden_states=x,
                           timestep=timesteps,
                           class_labels=cat,
                           pad_mask=key_padding_mask,
                           graph_mask=graph_mask,
                           attr_mask=attr_mask)
            
        x = self.final_layer(x, timesteps, cat)
        return x
