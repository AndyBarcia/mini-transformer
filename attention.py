import torch
import torch.nn as nn

from rotary_embeddings import *

class WindowAttention(nn.Module):
    """
    Efficient attention with a window of a given size and stride.
    There is a slight optimization in this implementation: each token
    never pays attention to itself (so the effective window size is
    the usual windows size plus 1).

    Implements both relative and rotary embeddings.
    """

    def __init__(
        self,
        n_embed,
        window_size,
        n_head,
        window_stride=1,
        head_size=None,
        encoding="relative",
        dropout=0.4
    ):

        super().__init__()
        self.head_size = n_embed // n_head if head_size is None else head_size
        self.n_head = n_head
        self.head_surface = self.n_head*self.head_size
        self.window_size = window_size
        self.window_stride = window_stride
        self.n_embed = n_embed
                
        self.encoding = encoding
        if encoding=="relative":
            self.queries_pos = nn.Parameter(torch.empty(self.window_size, self.n_head, self.head_size))
            nn.init.xavier_normal_(self.queries_pos)

            self.values_pos = nn.Parameter(torch.empty(self.window_size, self.n_head, self.head_size))
            nn.init.xavier_normal_(self.values_pos)
        elif encoding == "rotary":
            self.rotary_emb = RotaryEmbedding(self.head_size,257)

        # Key, query and value matrix.     
        self.kqv = nn.Linear(n_embed, self.head_size*n_head*3, bias=False)
        # Output head.
        self.out_head = nn.Linear(self.head_size*n_head, n_embed)
        
        # Triangular matrix to mask padding tokens.
        self.register_buffer('tril', torch.tril(torch.ones(self.window_size, self.window_size)).flip((1,)), persistent=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        
        # If the aren't enough tokens, no token can attend to anything, so we just skip.
        if T <= self.window_stride:
            return torch.zeros_like(x), None
        
        x = self.kqv(x).view(B,T,3,self.n_head,-1)
        if self.encoding == "rotary":
            # If using rotary embeddings, rotate the key and query
            # embeddings (the first and second indicis in the 3rd dimension)
            x[:,:,0:2] = self.rotary_emb(x[:,:,0:2], seq_dim=-3)
        x = x.view(B,T,-1)
        
        # Add padding tokens (necessary for leftmost tokens).
        pad = self.window_size*self.window_stride
        x = F.pad(x, (0,0,pad,0), "constant", 0) # (B,T+pad,(3*H*C))
        
        # Unfold windows
        size = (B,T,self.window_size,3,self.n_head,self.head_size)
        stride = (
            (T+pad)*3*self.head_surface, # B
            3*self.head_surface, # T
            3*self.head_surface*self.window_stride, # L
            self.head_surface, # 3
            self.head_size, # H
            1 # C
        )
        x = x.as_strided(size, stride)[:,self.window_stride:] # (B,T,L,3*H*C)
        
        # Extract Keys, Queries and Values of each individual head.
        tk, tq, tv = x[:,:,:,0], x[:,:,:,1], x[:,:,:,2] # (B,T,L,H,C)
        
        # Get a TxL matrix, with the attention scores of each token for their window.
        # Equivalent to einsum "b head key query c, b head key query c -> b head key query"
        wei = (tk*tq).sum(-1) # (B,T,L,H)
        
        # If using relative positioning, apply the effect of positioning to the queries.
        # Equivalent to einsum "head query c, b head key query c -> b head key query"
        if self.encoding == "relative":
            pos = self.queries_pos # (L,H,C)
            wei += (pos*tq).sum(-1) # (B,T,L,H)
                        
        # Scale, and apply softmax.
        wei = wei * self.head_size**-0.5 # (B,T,L,H)
        # Ignore padding tokens.        
        tril = self.tril[:(T-self.window_stride)].unsqueeze(-1).expand(-1,-1,self.n_head)
        wei[:,:self.window_size] = wei[:,:self.window_size].masked_fill(tril == 0, float('-inf'))        
        # Limit attention to sum up to one.
        wei = F.softmax(wei, dim=-2) # (B,T,L,H)
        
        # Block some random attentions.
        wei = self.dropout(wei) # (B,T,L,H)
        
        # Calculate values based on attention.
        out = (wei.unsqueeze(-1) * tv).sum(-3) # (B,T,H,C)
        
        # If using relative positioning, apply the effect of positioning to the values.
        if self.encoding == "relative":
            pos = self.values_pos # (L,H,C)
            out += (wei.unsqueeze(-1) * pos).sum(-3) # (B,T,H,C)
                
        # Join the output of each of the heads.
        out = out.view(B,T-self.window_stride, -1) # (B,T,H*C)
        
        # Recover first tokens that are not attended by anything.
        out = F.pad(out, (0,0,self.window_stride,0), "constant", 0)
                
        return self.out_head(out), wei