"""
上下文缓存+自适应检索窗口+多轮对话拼接
============================================
每句→sent_vec→三层缓存→滑动窗口检索→拼接上下文
窗口自适应: 相关性高缩小(精准), 低扩大(召回)
"""
import torch, torch.nn.functional as F, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P7_cross_sent.model import P7WordRouter2048

# ── 模型 ──
p1_ckpt = torch.load(os.path.join(SAVE_DIR, 'P1_best.pt'), map_location=DEVICE)
p1 = CharToWordModel(p1_ckpt['num_chars'], p1_ckpt['num_words']).to(DEVICE)
p1.load_state_dict(p1_ckpt['model_state_dict']); p1.eval()
c2i = p1_ckpt['char2idx']

p7 = P7WordRouter2048().to(DEVICE)
p7_ckpt = torch.load(os.path.join(SAVE_DIR, 'V18_full_chain.pt'), map_location=DEVICE)
p7.load_state_dict(p7_ckpt['p7'], strict=False); p7.eval()

def enc_sentence(text):
    words = text.split()
    ids_list = []
    for w in words:
        c1, c2 = w[0], w[0] if len(w) == 1 else w[1]
        if c1 in c2i and c2 in c2i: ids_list.append([c2i[c1], c2i[c2]])
    if not ids_list: return None
    with torch.no_grad():
        wv = p1(torch.tensor(ids_list, device=DEVICE), last_loss=0.0)
        _, sv, _, _, _ = p7(wv, wv, last_loss=0.0)
        return F.normalize(sv.unsqueeze(0), dim=-1)


# ── 对话上下文管理器 ──
class DialogueContext:
    """自适应窗口 + 多轮拼接 + 防跑题"""

    def __init__(self, init_window=5, min_window=3, max_window=20):
        self.history = []        # [(text, vec_gpu)]
        self.window_size = init_window
        self.min_win = min_window
        self.max_win = max_window
        self.topic_vec = None    # 话题中心向量
        self.topic_decay = 0.9   # 话题衰减系数
        self.drift_threshold = 0.3  # 跑题阈值: cos低于此值警告

    def add(self, text):
        """添加一句话到对话历史"""
        vec = enc_sentence(text)
        if vec is None: return
        self.history.append((text, vec))

        # 窗口连贯性: 跟最近K句分别比, 全部低→跑题
        K = min(5, len(self.history))
        if K > 0:
            recent_vecs = torch.cat([self.history[-i-1][1] for i in range(K)], dim=0)
            recent_cos = torch.mm(vec, recent_vecs.T).squeeze(0)  # [K]
            max_cos = recent_cos.max().item()
            avg_cos = recent_cos.mean().item()

            # 自适应窗口: 最近句相关性高→缩窗, 低→扩窗
            if avg_cos > 0.8:
                self.window_size = max(self.min_win, self.window_size - 1)
            elif avg_cos < 0.5:
                self.window_size = min(self.max_win, self.window_size + 1)

            # 跑题: 最近5句没一个cos>0.5
            self.is_drift = max_cos < 0.5
        else:
            avg_cos, max_cos = 1.0, 1.0
            self.is_drift = False

    def retrieve_context(self, query_text):
        """检索最相关的上下文, 返回拼接后的输入"""
        query_vec = enc_sentence(query_text)
        if query_vec is None or not self.history: return query_text, 0, []

        # 跟所有历史句计算相关性
        hist_vecs = torch.cat([v for _, v in self.history], dim=0)  # [N, 256]
        cos_sim = torch.mm(query_vec, hist_vecs.T).squeeze(0)       # [N]

        # 选窗口大小个最相关句
        k = min(self.window_size, len(self.history))
        top_k = torch.topk(cos_sim, k)

        # 跑题检测: 跟最近K句的max_cos
        K = min(5, len(self.history))
        max_cos = 1.0
        if K > 0:
            recent_vecs = torch.cat([self.history[-i-1][1] for i in range(K)], dim=0)
            max_cos = torch.mm(query_vec, recent_vecs.T).max().item()

        # 拼接上下文: [相关历史句] + [当前查询]
        context_parts = []
        retrieved = []
        for idx in top_k.indices:
            idx = idx.item()
            text, _ = self.history[idx]
            retrieved.append((idx, text, cos_sim[idx].item()))
            context_parts.append(text)
        context_parts.append(query_text)
        full_context = " | ".join(context_parts)

        return full_context, max_cos, retrieved

    def status(self):
        return f"历史:{len(self.history)}句 窗口:{self.window_size} 话题cos:{self.topic_vec[0,0].item():.3f}" if self.topic_vec is not None else "空"


# ── 测试 ──
print("=== 多轮对话防跑题测试 ===\n")
ctx = DialogueContext()

# 模拟对话
conversation = [
    ("用户", "你喜欢什么电影"),
    ("助手", "我喜欢科幻电影"),
    ("用户", "谁种了科幻电影"),
    ("助手", "那位老教练种了科幻电影"),
    ("用户", "他什么时候种的"),
    ("助手", "昨天下午在院子里种的"),
    ("用户", "今天天气怎么样"),  # ← 故意跑题
    ("助手", "今天晴天适合出门"),
    ("用户", "那部电影好看吗"),  # ← 回到电影话题
]
for speaker, text in conversation:
    ctx.add(text)
    full, _, retrieved = ctx.retrieve_context(text)
    drift = "PAOTI" if ctx.is_drift else "OK"
    print(f"[{speaker}] {text}")
    print(f"  drift={drift} | win={ctx.window_size}")
    if retrieved:
        print(f"  检索窗口(top{len(retrieved)}):")
        for idx, t, c in retrieved:
            print(f"    [{idx}] cos={c:.4f} | {t[:40]}")
    print()

# 最终检索: 回到电影话题
print("--- 最终查询 ---")
query = "那部科幻电影谁导演的"
full, max_cos, retrieved = ctx.retrieve_context(query)
print(f"查询: {query}")
print(f"最近cos={max_cos:.3f} drift={ctx.is_drift} | 窗口={ctx.window_size}")
print(f"拼接上下文:")
for idx, t, c in retrieved:
    print(f"  [{idx}] cos={c:.4f} | {t}")
print(f"\n完整输入: {full[:200]}...")
