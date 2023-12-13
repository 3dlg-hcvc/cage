
import torch
from torch import nn, Tensor
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings


class PositionalEncoding(nn.Module):
    """
    Positional encoding module from "Attention Is All You Need"

    Parameters:
        d_model (`int`): The number of channels in the input and output.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        max_len (`int`, *optional*, defaults to 5000): The maximum length of the input sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # type: ignore
        pe[:, 1::2] = torch.cos(position * div_term)  # type: ignore
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Apply positional encoding to input tensor.

        Parameters:
            x (torch.Tensor): The input tensor of shape `(batch_size, seq_len, embed_dim)`.

        Returns:
            torch.Tensor: The encoded tensor of shape `(batch_size, seq_len, embed_dim)`.
        """
        x = x + self.pe[: x.size(0), : x.size(1)]
        return self.dropout(x)

class FinalLayer(nn.Module):
    def __init__(self, in_ch, out_ch=None, dropout=0.):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.linear = nn.Linear(in_ch, out_ch)
        self.norm = AdaLayerNormTC(in_ch, 2*in_ch, dropout)
        
    def forward(self, x, t, cond=None):
        assert cond is not None
        x = self.norm(x, t, cond)
        x = self.linear(x)
        return x

class AdaLayerNormTC(nn.Module):
    """
    Norm layer modified to incorporate timestep and condition embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings, dropout):
        super().__init__()
        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim, dropout)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=torch.finfo(torch.float16).eps)

    def forward(self, x, timestep, cond):
        emb = self.linear(self.silu(self.emb(timestep, cond, hidden_dtype=None)))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class PEmbeder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.embed.weight, mode="fan_in")

    def forward(self, x, idx=None):
        if idx is None:
            idx = torch.arange(x.shape[1], device=x.device).long()
        return x + self.embed(idx)


class MyAdaLayerNormZero(nn.Module):
    """
    Adaptive layer norm zero (adaLN-Zero), borrowed from diffusers.models.attention.AdaLayerNormZero.
    Extended to incorporate scale parameters (gate_2, gate_3) for intermidate attention layers.
    """

    def __init__(self, embedding_dim, num_embeddings, class_dropout_prob):
        super().__init__()

        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim, class_dropout_prob)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 8 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, timestep, class_labels, hidden_dtype=None):
        emb_t_cls = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb_t_cls))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_2, gate_3 = emb.chunk(8, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_2, gate_3


class AAB(nn.Module):
    def __init__(self, 
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: int = None,
            attention_bias: bool = False,
            norm_elementwise_affine: bool = True,
            final_dropout: bool = False,
            class_dropout_prob: float = 0.0 # for classifier-free
        ):
        super().__init__()

        self.norm1 = MyAdaLayerNormZero(dim, num_embeds_ada_norm, class_dropout_prob)  
        
        self.global_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        
        self.attr_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        self.graph_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.norm4 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
    
    def forward(self, hidden_states, pad_mask, attr_mask, graph_mask, timestep, class_labels):
        norm_hidden_states, gate_1, shift_mlp, scale_mlp, gate_mlp, gate_2, gate_3 = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        attr_out = self.attr_attn(norm_hidden_states, attention_mask=attr_mask)
        attr_out = gate_1.unsqueeze(1) * attr_out
        hidden_states = hidden_states + attr_out

        norm_hidden_states = self.norm2(hidden_states)
        global_out = self.global_attn(norm_hidden_states, attention_mask=pad_mask)
        global_out = gate_2.unsqueeze(1) * global_out
        hidden_states = hidden_states + global_out 

        norm_hidden_states = self.norm3(hidden_states)
        graph_out = self.graph_attn(norm_hidden_states, attention_mask=graph_mask)
        graph_out = gate_3.unsqueeze(1) * graph_out
        hidden_states = hidden_states + graph_out

        norm_hidden_states = self.norm4(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        
        hidden_states = ff_output + hidden_states
        return hidden_states


