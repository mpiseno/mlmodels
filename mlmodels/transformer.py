import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = self.d_v = self.d_model // self.n_heads

        scale = self.d_model ** -0.5
        self.heads = {
            i: (
                nn.Parameter(scale * torch.randn(self.d_model, self.d_k)),
                nn.Parameter(scale * torch.randn(self.d_model, self.d_k)),
                nn.Parameter(scale * torch.randn(self.d_model, self.d_v))
            )
            for i in range(self.n_heads)
        }
        self.W_O = nn.Parameter(scale * torch.randn(self.n_heads * self.d_v, self.d_model))

    def attention(self, Q, K, V):
        return F.softmax(Q @ K.permute(0, 2, 1) / math.sqrt(self.d_k), dim=-1) @ V

    def forward(self, x):
        head_outputs = []
        for i in range(self.n_heads):
            W_Q, W_K, W_V = self.heads[i]
            Q_i = x @ W_Q
            K_i = x @ W_K
            V_i = x @ W_V
            head_i = self.attention(Q_i, K_i, V_i)
            head_outputs.append(head_i)
        
        head_outputs = torch.cat(head_outputs, dim=-1)
        return head_outputs @ self.W_O


class TransformerLayer(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim

        self.ln_mhsa = nn.LayerNorm(self.dim)
        self.mhsa = MultiHeadSelfAttention(d_model=self.dim, n_heads=n_heads)
        self.ln_mlp = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim)
        )

    def forward(self, x):
        x = x + self.mhsa(self.ln_mhsa(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(self, n_layers, n_heads, dim):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.encoder = nn.Sequential(*[TransformerLayer(self.dim, self.n_heads) for _ in range(self.n_layers)])
    
    def forward(self, x):
        return self.encoder(x)