"""
P1 字符→词语 (全链路128D统一版)
===============================
内部128D = 8D位置 + 120D语义, 64头×256D交叉注意力
输出128D词向量, 无投影层 — 直接兼容下游P2-P9
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.config import (POS_DIM, CONTENT_DIM, CHAR_DIM, WORD_DIM, MOD_DIM,
                          ATTN_HEADS, ATTN_DIM, ATTN_HEAD_DIM, ATTN_DROPOUT)


class PositionEncoding(nn.Module):
    def __init__(self, dim=POS_DIM, max_len=2):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, positions):
        return self.pe[positions]


class CrossAttention(nn.Module):
    """64头×256D: Q=两字符(256D), K,V=词表(128D) → 128D输出"""
    def __init__(self, char_dim=CHAR_DIM, word_dim=WORD_DIM, d_model=ATTN_DIM, heads=ATTN_HEADS):
        super().__init__()
        self.char_dim = char_dim
        self.word_dim = word_dim
        self.d_model = d_model
        self.heads = heads
        self.head_dim = ATTN_HEAD_DIM
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(char_dim * 2, d_model, bias=False)   # 256→256
        self.W_k = nn.Linear(word_dim, d_model, bias=False)        # 128→256
        self.W_v = nn.Linear(word_dim, d_model, bias=False)        # 128→256
        self.W_o = nn.Linear(d_model, word_dim, bias=False)        # 256→128
        self.dropout = nn.Dropout(ATTN_DROPOUT)

    def forward(self, char_vectors, word_table, return_weights=False):
        b = char_vectors.shape[0]; N = word_table.shape[0]
        q_input = torch.cat([char_vectors[:, 0, :], char_vectors[:, 1, :]], dim=-1)
        q = self.W_q(q_input).view(b, self.heads, self.head_dim)
        k = self.W_k(word_table).view(N, self.heads, self.head_dim)
        v = self.W_v(word_table).view(N, self.heads, self.head_dim)
        k = k.transpose(0,1).unsqueeze(0).expand(b,-1,-1,-1)
        v = v.transpose(0,1).unsqueeze(0).expand(b,-1,-1,-1)
        q = q.unsqueeze(2)
        attn_scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.squeeze(2).contiguous().view(b, self.d_model)
        output = self.W_o(attn_out)
        if return_weights: return output, attn_weights
        return output


class PosFusion(nn.Module):
    def forward(self, pos_vectors):
        out = pos_vectors[:, 0, :] + pos_vectors[:, 1, :]
        return F.normalize(out, dim=-1)


class ExplorationZone(nn.Module):
    def __init__(self, dim=MOD_DIM):
        super().__init__()
        self.pos_basis = nn.Parameter(torch.randn(dim) * 0.02)
        self.neg_basis = nn.Parameter(torch.randn(dim) * 0.02)
        self.strength_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 2), nn.Tanh())

    def forward(self, last_loss):
        lt = torch.tensor([last_loss], device=self.pos_basis.device)
        scales = self.strength_mlp(lt)
        modulation = self.pos_basis * scales[0] + self.neg_basis * scales[1]
        loss_factor = min(last_loss * 20.0, 1.0)
        return modulation * loss_factor


class MetaLearningZone(nn.Module):
    def __init__(self, dim=MOD_DIM):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Tanh(),
            nn.Linear(dim, dim, bias=False), nn.Tanh())

    def forward(self, explore_modulation):
        return self.transform(explore_modulation)


class CharToWordModel(nn.Module):
    """P1: 128D内部语义 + 64头×256D注意力 → 128D词向量 (无投影层)"""
    def __init__(self, num_chars, num_words):
        super().__init__()
        self.num_chars = num_chars
        self.num_words = num_words
        self.char_content = nn.Embedding(num_chars, CONTENT_DIM)  # 120D
        nn.init.xavier_uniform_(self.char_content.weight)
        self.word_table = nn.Parameter(torch.randn(num_words, WORD_DIM) * 0.01)  # 128D

        self.pos_encoder = PositionEncoding()
        self.cross_attn = CrossAttention()
        self.pos_fusion = PosFusion()
        self.explore_zone = ExplorationZone()
        self.meta_zone = MetaLearningZone()

    def get_char_vectors(self, char_ids):
        b = char_ids.shape[0]
        positions = torch.arange(2, device=char_ids.device).unsqueeze(0).repeat(b, 1)
        pos = self.pos_encoder(positions)       # [b, 2, 8]
        content = self.char_content(char_ids)   # [b, 2, 120]
        full = torch.cat([pos, content], dim=-1)  # [b, 2, 128]
        return pos, content, full

    def forward(self, char_ids, last_loss=1.0, return_details=False):
        pos, content, full = self.get_char_vectors(char_ids)

        # 交叉注意力: Q=字符对, K,V=词表
        attn_out, attn_weights = self.cross_attn(full, self.word_table, return_weights=True)  # [b, 128]

        # 调制链
        explore_mod = self.explore_zone(last_loss)
        meta_mod = self.meta_zone(explore_mod)
        meta_exp = meta_mod.unsqueeze(0).expand(pos.shape[0], -1)
        modulated_attn = attn_out + meta_exp  # [b, 128]

        # 位置融合: 8D
        pos_out = self.pos_fusion(pos)  # [b, 8]

        # 语义融合: 120D (字符内容求和 + 注意力语义区)
        base_sem = content[:, 0, :] + content[:, 1, :]         # [b, 120]
        attn_sem = modulated_attn[:, POS_DIM:]                  # [b, 120]
        content_out = F.normalize(base_sem + attn_sem, dim=-1)  # [b, 120]

        word_vector = torch.cat([pos_out, content_out], dim=-1)  # [b, 128]

        if return_details:
            return word_vector, {
                "pos_raw": pos, "content_raw": content, "char_full": full,
                "attn_weights": attn_weights, "attn_out_raw": attn_out,
                "modulated_attn": modulated_attn,
                "explore_mod": explore_mod, "meta_mod": meta_mod,
                "pos_out": pos_out, "content_out": content_out,
            }
        return word_vector

    def get_word_target(self, word_ids):
        return self.word_table[word_ids]

    @torch.no_grad()
    def get_all_reference_vectors(self, device):
        self.eval()
        return self.word_table.clone().to(device)

    @torch.no_grad()
    def predict_word_ids(self, char_ids):
        self.eval()
        pos, content, full = self.get_char_vectors(char_ids)
        _, attn_weights = self.cross_attn(full, self.word_table, return_weights=True)
        avg_attn = attn_weights.squeeze(2).mean(dim=1)
        return avg_attn.argmax(dim=-1), avg_attn
