import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import einsum, rearrange
import random
import sys
from tqdm import tqdm
from torchinfo import summary

from tokenizer import *
from rotary_embeddings import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 96
block_size = 256

# 256 tokenizer.
encode, decode, sp = create_tokenizer("unigram_256_simplified.model")

# Efficient attention with a window of a given size.
# Implements both relative and rotary embeddings.
class WindowAttention(nn.Module):
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
                    
        self.kqv = nn.Linear(n_embed, self.head_size*n_head*3, bias=False)
        
        self.out_head = nn.Linear(self.head_size*n_head, n_embed)
        
        # Triangular matrix to mask padding tokens.
        # Convertir esto a matriz (L,L,H). De esta manera también es fácil
        # limitar el tamaño de varios heads, a ver qué pasa.
        self.register_buffer('tril', torch.tril(torch.ones(self.window_size, self.window_size)).flip((1,)), persistent=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        
        # If the aren't enough tokens, no token can attend to anything, so we just skip.
        if T <= self.window_stride:
            return torch.zeros_like(x), None
        
        #print(f"B {B}; H {self.n_head}; T {T}; L {self.window_size}; S {self.window_stride}; C {self.head_size}")
        
        x = self.kqv(x).view(B,T,3,self.n_head,-1) # (B,T,(3*H*C))
        
        if self.encoding == "rotary":
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
        wei = (tk*tq).sum(-1) # "b head key query c, b head key query c -> b head key query" (B,T,L,H)
        
        # Effect of position: "head query c, b head key query c -> b head key query"
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
        
        # Bloquear aleatoriamente algunas comunicaciones entre tokens.
        wei = self.dropout(wei) # (B,T,L,H)
        
        # Valores de salida de los valores de los tokens.
        out = (wei.unsqueeze(-1) * tv).sum(-3) # (B,T,H,C)
        
        # Valores de salida en base a las posiciones relativas de los tokens.
        if self.encoding == "relative":
            pos = self.values_pos # (L,H,C)
            out += (wei.unsqueeze(-1) * pos).sum(-3) # (B,T,H,C)
                
        # Juntar las distintas cabezas de cada token otra vez.
        out = out.view(B,T-self.window_stride, -1) # (B,T,H*C)
        
        # Recoperar primeros tokens que no atienden a nada.
        out = F.pad(out, (0,0,self.window_stride,0), "constant", 0)
                
        return self.out_head(out), wei

# Una simple capa oculta de una red neuronal, con 
# una función de activación ReLU.
# La capa tiene 4 veces más neuronas que la entrada
# y salida.
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            # Aplicar transformación lineal.
            nn.Linear(n_embed, 4*n_embed),
            # Aplicar función de activación.
            nn.ReLU(),
            # Devolver a camino residual.
            nn.Linear(4*n_embed, n_embed),
            # Bloquear (dropout) ciertas neuronas.
            # Previente overfitting
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

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
        
        # Propiedades que se pasarán a cada una de las capas.
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
        
        # Rotate embeddings
        #if self.encoding == "rotary":
        #    x = self.rotary_emb.rotate_queries_or_keys(x)
        
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
                
        if targets is None: 
            logits = self.lm_head(x) # (B,T,vocab_size)
                        
            results = {
                "logits": logits,
                "layer_activations": layers,
                "attentions": attentions,
                "output_embeddings": x,
            }
            
            return results, None
        else:
            logits = self.lm_head(x) # (B,T,vocab_size)
                        
            results = {
                "logits": logits,
                "layer_activations": layers,
                "attentions": attentions,
                "output_embeddings": x,
            }
            
            logits = logits[:,:,:].view(-1,self.vocab_size)
            targets = targets.flatten()
            mean_loss = F.cross_entropy(logits, targets, reduction="mean")
            losses = {
                "mean": mean_loss,
            }
            
            return results, losses
    
    @torch.no_grad()
    def estimate_loss(self, data_splits, test_iters=5, loss_type="mean"):
        out = {}
        self.eval()
        with torch.inference_mode():
            for name, split in data_splits.items():
            #for name, split in [("train", data_train), ("test", data_test)]:
            #    # TEST
            #    split = get_batch(name)
            #    # TEST
                
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
                # TODO: calcular cuál es la cantidad de tokens que podemos
                # utilizar que tenga sentido (el límite de propagación de información).
                idx_cond = idx[:, -block_size:]
                # Obtener predicciones
                results, loss = self(idx_cond)
                
                logits = results["logits"][:,-1] # (B,vocab_size)
                logits[:, idx[:,-32:]] /= repetition_penalty
                                
                probs = F.softmax(logits / temperature, dim=-1) # (C)
                
                #v,i = torch.sort(probs)
                #for b in range(probs.shape[0]):
                #    probs[b][i[b][torch.cumsum(v[b], -1) < 0.1]] = 0.0
                                
                idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
                yield idx
        self.train()
