import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import einsum, rearrange
from tokenizer import *
import random

# https://github.com/lucidrains/PaLM-rlhf-pytorch/blob/main/palm_rlhf_pytorch/palm.py#L69
# https://arxiv.org/pdf/2104.09864.pdf
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, seq_len, scale_base=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        assert dim % 2 == 0
        
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

        # Calcular rotation step de cada pareja de dimensiones.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim//2, device=device).float() / (dim*2))) # (C/2)
        # Escalar la rotación por cada posición T.
        freqs = torch.arange(seq_len, device=device).unsqueeze(-1).expand(-1, inv_freq.shape[0]) * inv_freq # (T,C/2)
                
        # Get cosine elements (diagonal)
        cos = freqs.cos() # (T,C/2)
        cos = torch.cat((cos, cos), dim = -1) # (T,C)
        
        # Get sine elements (odd elements are negative)
        sin = freqs.sin() # (T,C)
        sin = torch.cat((-sin, sin), dim = -1) # (T,C)
                
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, seq_dim=-2):
        if seq_dim < -2:
            extra_dims = -(seq_dim+2)
            cos = self.cos[(slice(None),) * 1 + (None, ) * extra_dims][:x.shape[seq_dim]]
            cos = cos.expand(x.shape[seq_dim:])
            sin = self.sin[(slice(None),) * 1 + (None, ) * extra_dims][:x.shape[seq_dim]]
            sin = sin.expand(x.shape[seq_dim:])
        else:
            cos, sin = self.cos, self.sin
        
        cos = cos * x
        
        x1, x2 = x.chunk(2, dim=-1)
        x = torch.cat((x2, x1), dim=-1)
        sin = sin * x
        
        return sin + cos
