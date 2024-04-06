
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
        mid_dim = attn_dim // 2
        dropout = hparams.dropout
        n_head = hparams.n_head

        head_dim = attn_dim // n_head
        num_embeds_ada_norm = 6*attn_dim
        self.K = self.hparams.get('K', 32)

        self.aabb_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim)
        )

        self.jaxis_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim)
        )

        self.range_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim)
        )

        self.label_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim)
        )

        self.jtype_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim)
        )

        self.pe_node = PEmbeder(self.K, attn_dim)
        self.pe_attr = PEmbeder(5, attn_dim)

        self.attn_layers = nn.ModuleList(
            [ 
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
        B = x.shape[0]
        x = x.view(B, self.K, 5*6)

        # embedding layers for different attributes
        x_aabb = self.aabb_emb(x[..., :6])
        x_jtype = self.jtype_emb(x[..., 6:12])
        x_jaxis = self.jaxis_emb(x[..., 12:18])
        x_range = self.range_emb(x[..., 18:24])
        x_label = self.label_emb(x[..., 24:30])

        x_ = torch.cat([x_aabb, x_jtype, x_jaxis, x_range, x_label], dim=2) # (B, K, 5*attn_dim)
        x_ = x_.view(B, self.K* 5, self.hparams.attn_dim)

        # positional encoding for nodes and attributes
        idx_attr = torch.tensor([0,1,2,3,4], device=x.device, dtype=torch.long).repeat(self.K)
        idx_node = torch.arange(self.K, device=x.device, dtype=torch.long).repeat_interleave(5)
        x_ = self.pe_attr(self.pe_node(x_, idx=idx_node), idx=idx_attr)

        # attention layers
        for attn_layer in self.attn_layers:
            x_ = attn_layer(hidden_states=x_,
                           timestep=timesteps,
                           class_labels=cat,
                           pad_mask=key_padding_mask,
                           graph_mask=graph_mask,
                           attr_mask=attr_mask)
            
        y = self.final_layer(x_, timesteps, cat)
        return y
