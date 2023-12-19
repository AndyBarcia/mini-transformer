import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import *
from attention import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 96
block_size = 256

# 256 tokenizer.
encode, decode, sp = create_tokenizer("unigram_256_simplified.model")

"""
Standard Feed Forward layer with ReLU activation
and 4 times as many hidden neurons as input neurons. 
"""
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

"""
Standard decoder-only Transformer block implementing window attention
and either relative or rotary positioning.

The Feed Forward layer is optional.
"""
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        n_embed,
        with_ffw=True,
        encoding="relative",
        dropout=0.4,
        **kwargs
    ):
        super().__init__()
        self.sa_heads = WindowAttention(n_embed=n_embed, encoding=encoding, dropout=dropout, **kwargs)
        self.ln1 = nn.LayerNorm(n_embed)
        self.with_ffw = with_ffw
        if with_ffw is True:
            self.ffw = FeedForward(n_embed, dropout)
            self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        dx, wei = self.sa_heads(self.ln1(x))
        x = x + dx
        
        if self.with_ffw is True:
            x = x + self.ffw(self.ln2(x))
        
        return x, wei

"""
Standard decoder-only Transformer  implementing window attention
and either relative or rotary positioning.

Each of the layers of the transformer is independently configurable.
"""
class Transformer(nn.Module):
    def __init__(
        self, 
        vocab_size,
        n_embed, 
        encoding="relative",
        n_layers=None,
        dropout=0.4,
        **layers
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.dropout = dropout
        
        # Setup configuration of each layer.
        self.n_layers = n_layers
        for layer_prop in layers:
            if isinstance(layers[layer_prop],tuple):
                if self.n_layers is None:
                    self.n_layers = len(layers[layer_prop])
                else:
                    assert self.n_layers == len(layers[layer_prop])
        
        self.layer_props = layers.copy()
        for layer_prop in layers:
            if not isinstance(layers[layer_prop],tuple):
                layers[layer_prop] = (layers[layer_prop],)*self.n_layers
        
        self.layers = []
        for i in range(self.n_layers):
            props = {}
            for layer_prop in layers:
                props[layer_prop] = layers[layer_prop][i]
            self.layers.append(props)
         
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.ln = nn.LayerNorm(n_embed)
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(n_embed, vocab_size)        
        self.start_token = nn.Parameter(torch.randn(n_embed))
        self.encoding = encoding
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                n_embed, 
                encoding=encoding, 
                dropout=dropout, 
                **self.layers[i]
            ) for i in range(self.n_layers)
        ])
    
    def guardar_parametros(self, fichero):
        torch.save(self.state_dict(), fichero)
    
    def cargar_parametros(self, fichero):
        self.load_state_dict(torch.load(fichero), strict=False)
    
    def numero_parametros(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, idx, targets=None, save_layer_activations=False, save_attention=False):        
        # If the inputs are in tokens, turn them into embeddings.
        if idx.dtype == torch.long:
            B,T = idx.shape
            token_embed = self.token_embedding_table(idx) # (B,T,C)
        else:
            B,T,C = idx.shape
            token_embed = idx
        
        # Append start token.
        x = torch.cat([self.start_token[None, None, :].repeat(B,1,1), token_embed], dim=1)
                
        # Pass embeddings thrhogh all layer blocks.
        if save_layer_activations:
            layers = x.unsqueeze(1) # (B,L,T,C)
            attentions = []
            for block in self.blocks:
                x = block(x)  # (B,T,C)
                layers = torch.cat([layers,x.unsqueeze(1)],dim=1) # (B,L,T,C)
        else:
            layers = None
            attentions = []
            for block in self.blocks:
                x, wei = block(x) # (B,T,C)
                if save_attention:
                    attentions.append(wei)
        x = self.ln(x) # (B,T,C)

        # Turn embeddings of the last layer into logits
        logits = self.lm_head(x) # (B,T,vocab_size)             
        results = {
            "logits": logits,
            "layer_activations": layers,
            "attentions": attentions,
            "output_embeddings": x,
        }        
        
        # Calculate loss if target output is given.
        # Otherwise, just return the logits.
        if targets is None: 
            return results, None
        else:            
            logits = logits[:,:,:].view(-1,self.vocab_size)
            targets = targets.flatten()
            mean_loss = F.cross_entropy(logits, targets, reduction="mean")
            losses = { "mean": mean_loss }
            return results, losses
    
    @torch.no_grad()
    def estimate_loss(self, data_splits, test_iters=5, loss_type="mean"):
        out = {}
        self.eval()
        with torch.inference_mode():
            for name, split in data_splits.items():
                losses = torch.zeros(test_iters, device=device)

                split = iter(split)
                for k in range(test_iters):
                    batch = next(split)
                    x,y = batch[:,:-1], batch[:,:]
                    _, loss = self(x,y)
                    losses[k] = loss[loss_type].item()
                out[name] = losses.mean().item()
        self.train()
        return out
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, repetition_penalty=1.0, speculate=True):
        if isinstance(idx, str):
            idx = encode(idx).to(device).unsqueeze(0)
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)
        if idx.device != device:
            idx = idx.to(device)
        
        self.eval()
        with torch.inference_mode():
            for i in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                results, loss = self(idx_cond)
                
                logits = results["logits"][:,-1] # (B,vocab_size)
                logits[:, idx[:,-32:]] /= repetition_penalty
                                
                probs = F.softmax(logits / temperature, dim=-1) # (C)                                
                idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
                yield idx
        self.train()
