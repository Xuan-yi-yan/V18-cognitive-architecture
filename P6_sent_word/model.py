"""
P6 句子→词序列 反向解码 (128D统一版)
======================================
输入: P5句子向量(256D)
输出: 还原各词向量 [词1(128D), 词2(128D), 词3(128D)]

架构: 256D → encoder(256→256) → 3平行头 → 3×128D
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SentToWordsDecoder(nn.Module):
    def __init__(self, sent_dim=256, word_dim=128, hidden_dim=256, max_words=3):
        super().__init__()
        self.sent_dim = sent_dim
        self.word_dim = word_dim
        self.max_words = max_words

        self.encoder = nn.Sequential(
            nn.Linear(sent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.word_heads = nn.ModuleList([
            nn.Linear(hidden_dim, word_dim) for _ in range(max_words)
        ])

        self.explore_state = nn.Parameter(torch.randn(word_dim) * 1.0)
        self.meta_fc = nn.Sequential(
            nn.Linear(word_dim, word_dim, bias=False))

    def forward(self, sent_vec, last_loss=1.0):
        b = sent_vec.shape[0]
        h = self.encoder(sent_vec)  # [b, 256]

        words = []
        for head in self.word_heads:
            w = head(h)  # [b, 128]
            loss_factor = min(last_loss * 20.0, 1.0)
            mod = self.meta_fc(self.explore_state * loss_factor)
            w = w + mod.unsqueeze(0).expand(b, -1)
            words.append(w)

        return torch.stack(words, dim=1)  # [b, 3, 128]


def decode_loss(pred_words, true_words, mask):
    sims = F.cosine_similarity(pred_words, true_words, dim=-1)  # [b, max]
    masked_sims = sims * mask
    n_valid = mask.sum() + 1e-8
    avg_sim = masked_sims.sum() / n_valid
    return (1.0 - avg_sim), avg_sim.item()
