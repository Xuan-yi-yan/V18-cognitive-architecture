"""
P7 跨句词级路由 (128D + 64头)
===============================
Q = A句词序列(各128D)
K,V = B词表(128D)
64头交叉注意力 → 256D B句向量
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossSentenceRouter(nn.Module):
    def __init__(self, word_dim=2048, attn_dim=2048, sent_dim=4096, heads=64):
        super().__init__()
        self.word_dim = word_dim
        self.attn_dim = attn_dim
        self.sent_dim = sent_dim
        self.heads = heads
        self.head_dim = attn_dim // heads  # 4
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(word_dim, attn_dim, bias=False)  # 128→256
        self.k_proj = nn.Linear(word_dim, attn_dim, bias=False)  # 128→256
        self.v_proj = nn.Linear(word_dim, attn_dim, bias=False)  # 128→256

        self.sent_fuse = nn.Linear(attn_dim, sent_dim, bias=False)  # 256→256

        self.explore_state = nn.Parameter(torch.randn(sent_dim) * 0.01)
        self.meta_fc = nn.Sequential(
            nn.Linear(sent_dim, sent_dim, bias=False), nn.Tanh())

    def forward(self, A_word_vecs, B_word_table, last_loss=1.0):
        nA = A_word_vecs.shape[0]
        nB = B_word_table.shape[0]

        q = self.q_proj(A_word_vecs).view(nA, self.heads, self.head_dim)
        k = self.k_proj(B_word_table).view(nB, self.heads, self.head_dim)
        v = self.v_proj(B_word_table).view(nB, self.heads, self.head_dim)

        q = q.transpose(0, 1)  # [heads, nA, 4]
        k = k.transpose(0, 1)  # [heads, nB, 4]
        v = v.transpose(0, 1)  # [heads, nB, 4]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)  # [heads, nA, 4]

        attn_out = attn_out.transpose(0, 1).contiguous().view(nA, self.attn_dim)  # [nA, 256]
        sent_raw = attn_out.mean(dim=0)  # [256]
        sent_vec = self.sent_fuse(sent_raw)  # [256]

        loss_factor = min(last_loss * 20.0, 1.0)
        mod = self.meta_fc(self.explore_state * loss_factor)
        sent_vec = sent_vec + mod

        return sent_vec, attn  # [256], [heads, nA, nB]
