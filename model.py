
import math
import torch
import torch.nn as nn


# ── Positional Encoding ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ── Multi-Head Attention ────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, L, D = q.shape
        # project and reshape to (B, n_heads, seq_len, head_dim)
        q = self.q_proj(q).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)

        # combine heads
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


# ── Encoder Layer ───────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x))          # self-attention + residual
        x = self.norm2(x + self.ff(x))                   # feed-forward + residual
        return x


# ── Decoder Layer ───────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask):
        x = self.norm1(x + self.self_attn(x, x, x, mask=tgt_mask))   # masked self-attention
        x = self.norm2(x + self.cross_attn(x, memory, memory))       # cross-attention
        x = self.norm3(x + self.ff(x))                               # feed-forward
        return x


# ── Full Transformer ────────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=256):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)         # shared src/tgt embedding
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        x = self.pos_enc(self.embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, tgt, memory):
        causal_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device, dtype=torch.bool), diagonal=1)
        x = self.pos_enc(self.embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask=causal_mask)
        return self.out_proj(x)

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src))