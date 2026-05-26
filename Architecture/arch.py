import torch
import torch.nn as nn
import torch.nn.functional as F

"""# Global Vars"""

dtype = torch.float16
vocab_size = 32000
embed_dim = 768
n_head = 12
n_layer = 12
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 12
epochs = 1

max_seq_len = 1024

weight_decay = 0.1
b1 = 0.9
b2 = 0.95
peak_lr = 3e-4
gradient_accumulation_steps = 4
BASE_DIR = "/kaggle/working/"
update_checkpoint_steps = 100
update_plot_steps = 100

"""# RoPE"""


class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, d: int, base: int = 10_000):

        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        seq_len = x.shape[1]  # B, T, heads, d -> T is dim 1

        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return

        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(
            x.device
        )
        seq_idx = torch.arange(seq_len, device=x.device).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta2.cos()[None, :, None, :]  # (1, T, 1, d)
        self.sin_cached = idx_theta2.sin()[None, :, None, :]  # (1, T, 1, d)

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        # x: (B, T, num_heads, head_dim)
        self._build_cache(x)
        neg_half_x = self._neg_half(x)
        x_rope = (x * self.cos_cached[:, : x.shape[1]]) + (
            neg_half_x * self.sin_cached[:, : x.shape[1]]
        )
        return x_rope


"""# Attention"""


class ScaledDotProductAttention(nn.Module):
    def __init__(self, per_head_embed_dim):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.d = per_head_embed_dim

    def forward(self, Q, K, V):
        return nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)


"""# Layer"""


class Llm_layer(nn.Module):
    def __init__(self, n_head, embed_dim, dropout_ratio=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.linear1 = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_layer = ScaledDotProductAttention(
            per_head_embed_dim=embed_dim // n_head
        )
        self.lyr_norm1 = nn.LayerNorm(embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.lyr_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.rope = RotaryPositionalEmbeddings(d=embed_dim // n_head)
        self.linear_dropout = nn.Dropout(p=dropout_ratio)
        self.ffn_dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        x0 = self.linear1(self.lyr_norm1(x))
        Q, K, V = x0.split(self.embed_dim, dim=-1)

        Q = self.rope(
            Q.reshape(
                Q.shape[0], Q.shape[1], self.n_head, self.embed_dim // self.n_head
            )
        ).transpose(1, 2)
        K = self.rope(
            K.reshape(
                K.shape[0], K.shape[1], self.n_head, self.embed_dim // self.n_head
            )
        ).transpose(1, 2)
        V = V.reshape(
            V.shape[0], V.shape[1], self.n_head, self.embed_dim // self.n_head
        ).transpose(1, 2)

        res = self.attn_layer(Q, K, V)
        a, b, c, d = res.shape
        res = res.transpose(1, 2).reshape(a, c, self.embed_dim)

        x = x + self.linear_dropout(self.linear2(res))

        x = x + self.ffn_dropout(self.ffn(self.lyr_norm2(x)))
        return x


"""# LLM"""


# @title LLM
class LLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layer, n_head, dropout_ratio=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(p=dropout_ratio)
        self.llm_layers = nn.ModuleList(
            [Llm_layer(n_head=n_head, embed_dim=embed_dim) for i in range(n_layer)]
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.linear.weight = self.embedding.weight
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.embed_dropout(self.embedding(x))
        for llm_layer in self.llm_layers:
            x = llm_layer(x)
        x = self.linear(self.layer_norm1(x))

        return x
