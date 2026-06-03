"""
P4 主语原型 → 主语词检索
==========================
输入: P3主语原型(32D)
输出: 检索到的所有主语词列表
Loss: 原型解码向量与各主语词的Pearson r
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def pearson_r(x, y):
    xm = x.mean(dim=-1, keepdim=True); ym = y.mean(dim=-1, keepdim=True)
    xc = x - xm; yc = y - ym
    num = (xc * yc).sum(dim=-1)
    den = torch.sqrt((xc**2).sum(dim=-1) * (yc**2).sum(dim=-1) + EPS)
    return num / den


class ProtoToWordsDecoder(nn.Module):
    """主语原型 → 词语检索解码器"""
    def __init__(self, word_dim=32, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(word_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_dim, word_dim)

        self.explore_state = nn.Parameter(torch.randn(word_dim) * 0.01)
        self.meta_fc = nn.Sequential(
            nn.Linear(word_dim, word_dim, bias=False), nn.Tanh())

    def forward(self, prototype, last_loss=1.0):
        """prototype: [32] → decoded: [32]"""
        h = self.encoder(prototype.unsqueeze(0))
        out = self.decoder(h).squeeze(0)

        loss_factor = min(last_loss * 20.0, 1.0)
        mod = self.meta_fc(self.explore_state * loss_factor)
        out = out + mod

        return out

    @torch.no_grad()
    def retrieve(self, prototype, all_word_vecs, idx2word, threshold=0.5):
        """原型 → 检索所有匹配词"""
        self.eval()
        decoded = self.forward(prototype, last_loss=0.0)  # [32]
        scores = pearson_r(decoded.unsqueeze(0).expand(all_word_vecs.shape[0], -1), all_word_vecs)
        matches = [(idx2word[i], scores[i].item()) for i in range(len(scores)) if scores[i] > threshold]
        matches.sort(key=lambda x: -x[1])
        return matches, scores


def retrieval_loss(decoded_vec, family_word_vecs, non_family_vecs, margin=0.3):
    """
    decoded应与主语族词高Pearson r, 与非主语词低Pearson r
    """
    pos_r = pearson_r(decoded_vec.unsqueeze(0).expand(family_word_vecs.shape[0], -1), family_word_vecs)
    neg_r = pearson_r(decoded_vec.unsqueeze(0).expand(non_family_vecs.shape[0], -1), non_family_vecs)

    pos_l = F.relu(1.0 - margin - pos_r).mean()
    neg_l = F.relu(neg_r - margin).mean()

    return pos_l + neg_l, pos_r.mean().item(), neg_r.mean().item()
