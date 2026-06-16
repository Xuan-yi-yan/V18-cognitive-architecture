"""
V19 全链路训练 — 多数据集 + 三集分离 + 维度翻倍 + 从头训
==========================================================
数据: v5(71对) + 5k(4544对) → 合并 → 训练80% / 测试10% / 考试10%
维度: P7(128→256) + P6(256→512) + sent_vec(256→512)
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, random, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P7_cross_sent.model import P7WordRouter2048
from P6_sent_word.model import SentToWordsDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000); parser.add_argument("--display", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--data", type=str, default="auto", help="auto=本地v5+5k, public=公开集, 或文件路径")
args = parser.parse_args()

LOG_PATH = os.path.join(BASE_DIR, "logs", f"v19_{time.strftime('%H%M%S')}.txt")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
log = open(LOG_PATH, "w", encoding="utf-8")
def w(s): log.write(s+"\n"); log.flush(); print(s, flush=True)

w("="*70)
w(f"  V19 维度翻倍训练: 128→256(P7) 256→512(P6) sent_vec 512D")
w(f"  epochs={args.epochs} display={args.display} lr={args.lr}")
w("="*70)

# ── P1: 分批GPU编码 ──
w("\n[P1] 加载+编码...")
p1_ckpt = torch.load(os.path.join(SAVE_DIR, 'P1_best.pt'), map_location=DEVICE)
p1 = CharToWordModel(p1_ckpt['num_chars'], p1_ckpt['num_words']).to(DEVICE)
p1.load_state_dict(p1_ckpt['model_state_dict']); p1.eval()
c2i = p1_ckpt['char2idx']

# 加载数据
all_pairs = []
if args.data == "auto":
    for fname in ['data_p7_v5.txt', 'data_p7_5k.txt']:
        with open(f'C:/ai/P7_cross_sent/{fname}', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                a, b = line.split('|')
                all_pairs.append((a.split(), b.split()))
    w(f"  数据: {len(all_pairs)}对 (v5+5k)")
else:
    import re
    SENT_SPLIT = re.compile(r'[。！？；\n]')
    path = "C:/ai/data/public/public_combined.txt" if args.data=="public" else args.data
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            a, b = line.split('\t', 1)
            # 长文本按标点断句, 逐句配对
            sa = [p.strip() for p in SENT_SPLIT.split(a) if len(p.strip())>=3]
            sb = [p.strip() for p in SENT_SPLIT.split(b) if len(p.strip())>=3]
            for xa, xb in zip(sa, sb):
                wa = [c for c in xa if c.strip() and ord(c)>32]; wb = [c for c in xb if c.strip() and ord(c)>32]
                if 3<=len(wa)<=80 and 3<=len(wb)<=80:
                    all_pairs.append((wa, wb))
    w(f"  数据: {len(all_pairs)}对 (断句后, {path})")

# 编码所有词(分批50) — UNK兜底用上全部词
all_w = set()
for A, B in all_pairs: all_w.update(A); all_w.update(B)
# 找P1最大索引做UNK
unk_id = max(c2i.values()) if c2i else 0
word_cache = {}
batch_w, batch_ids = [], []
for word in sorted(all_w):
    c1, c2 = word[0], word[0] if len(word) == 1 else word[1]
    id1 = c2i.get(c1, unk_id); id2 = c2i.get(c2, unk_id)
    batch_w.append(word); batch_ids.append([id1, id2])
    if len(batch_ids) >= 50:
        with torch.no_grad():
            vecs = p1(torch.tensor(batch_ids, device=DEVICE), last_loss=0.0)
        for bw, bv in zip(batch_w, vecs): word_cache[bw] = bv
        batch_w, batch_ids = [], []; torch.cuda.empty_cache()
if batch_ids:
    with torch.no_grad():
        vecs = p1(torch.tensor(batch_ids, device=DEVICE), last_loss=0.0)
    for bw, bv in zip(batch_w, vecs): word_cache[bw] = bv
del p1; torch.cuda.empty_cache()
w(f"  词编码: {len(word_cache)}词, GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

# 编码句对
encoded = []
for A, B in all_pairs:
    if not all(w in word_cache for w in A + B): continue
    Av = torch.stack([word_cache[w].clone() for w in A])
    Bv = torch.stack([word_cache[w].clone() for w in B])
    encoded.append((Av, Bv, A, B))
w(f"  有效句对: {len(encoded)}")

# 三集分离: 80/10/10
random.shuffle(encoded)
n = len(encoded)
t80 = int(n * 0.80); t10 = int(n * 0.10)
train_set = encoded[:t80]
test_set = encoded[t80:t80+t10]
exam_set = encoded[t80+t10:]
# 保存考试集(绝对隔离, 训练完才碰)
exam_path = os.path.join(SAVE_DIR, "V19_exam_set.pt")
torch.save(exam_set, exam_path)
w(f"  训练: {len(train_set)} | 测试: {len(test_set)} | 考试: {len(exam_set)} (已隔离)")

# 展示样本从测试集取
long_idx = max(range(len(test_set)), key=lambda i: len(test_set[i][2]))
A_v, B_v, A_w, B_w = test_set[long_idx]
w(f"  展示(测试集): {len(A_w)}→{len(B_w)}词 | A={A_w} | B={B_w}")

# ── 模型 (维度翻倍) ──
w("\n[模型] V19 维度翻倍...")
p7 = P7WordRouter2048(max_len=128).to(DEVICE)

p6 = SentToWordsDecoder(max_words=128).to(DEVICE)

class UnifiedExplore(nn.Module):
    def __init__(self, in_dim=12, hidden=128, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden*2), nn.GELU(),
            nn.Linear(hidden*2, out_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1); nn.init.zeros_(layer.bias)
    def forward(self, x): return torch.tanh(self.net(x.unsqueeze(0)))

class UnifiedMeta(nn.Module):
    def __init__(self, dim=256):
        super().__init__(); self.bias = nn.Parameter(torch.randn(dim)*0.1)
    def forward(self, x): return torch.sigmoid(self.bias + x.squeeze(0))

explore = UnifiedExplore().to(DEVICE)
meta = UnifiedMeta().to(DEVICE)

opt_p7 = torch.optim.Adam(p7.parameters(), lr=args.lr*1.5)
opt_p6 = torch.optim.Adam(p6.parameters(), lr=args.lr)
opt_gate = torch.optim.Adam(list(explore.parameters())+list(meta.parameters()), lr=args.lr*2)

tp = sum(p.numel() for p in list(p7.parameters())+list(p6.parameters())+list(explore.parameters())+list(meta.parameters()))
w(f"  P7: {sum(p.numel() for p in p7.parameters()):,} | P6: {sum(p.numel() for p in p6.parameters()):,}")
w(f"  Explore+Meta: {sum(p.numel() for p in explore.parameters())+sum(p.numel() for p in meta.parameters()):,} | 总: {tp:,}")

# ── 训练 ──
w(f"\n{'='*70}\n  训练 {args.epochs}轮 (从零开始)\n{'='*70}\n")
total_t0 = time.time()

for ep in range(1, args.epochs+1):
    t0 = time.time(); total_loss, n = 0.0, 0

    for Av, Bv, _, _ in train_set:
        prev_loss = getattr(p7, '_loss_vec', None)
        gate = meta(explore(prev_loss)) if prev_loss is not None else None
        _, sv, _, _, _ = p7(Av, Bv, last_loss=0.0)
        nB = Bv.shape[0]
        pred_w = p6(sv.unsqueeze(0), gate=gate)[:, :nB, :]
        cos = F.cosine_similarity(pred_w, Bv.unsqueeze(0), dim=-1).mean()
        loss = 1.0 - cos
        loss_vec = torch.zeros(12, device=DEVICE); loss_vec[0] = cos.item(); loss_vec[1] = loss.item()
        p7._loss_vec = loss_vec
        for o in [opt_p7, opt_p6, opt_gate]: o.zero_grad()
        loss.backward()
        for o in [opt_p7, opt_p6, opt_gate]: o.step()
        total_loss += loss.item(); n += 1

    elapsed = time.time()-t0; avg_loss = total_loss/max(n,1)

    if ep<=3 or ep%args.display==0 or ep==args.epochs:
        with torch.no_grad():
            _, sv, _, _, _ = p7(A_v, B_v, last_loss=0.0)
            pred_w = p6(sv.unsqueeze(0))[:, :len(B_w), :]
            pred_n = F.normalize(pred_w.squeeze(0), dim=-1)
            true_n = F.normalize(B_v, dim=-1)
            c = torch.mm(pred_n, true_n.T).argmax(dim=-1)
            pred = [B_w[c[i].item()] for i in range(len(B_w))]
            ok = sum(1 for p,t in zip(pred, B_w) if p==t)
            per_cos = F.cosine_similarity(pred_w.squeeze(0), B_v, dim=-1).tolist()
            per_cos_str = " ".join([f"{c:.3f}" for c in per_cos])
        ttl = time.time()-total_t0
        eta = ttl/ep*(args.epochs-ep) if ep>0 else 0
        w(f"E{ep:5d}/{args.epochs} | loss={avg_loss:.4f} | pred={pred}")
        w(f"  测试集 per-cos=[{per_cos_str}] | 正确={ok}/{len(B_w)} | {elapsed:.0f}s | ETA {eta/60:.0f}min")
        w(f"{'='*70}")
    elif ep%200==0:
        ttl = time.time()-total_t0; eta = ttl/ep*(args.epochs-ep)
        w(f"  E{ep:5d} | loss={avg_loss:.4f} | ETA {eta/3600:.1f}h")

# ── 保存 + 考试集评测 ──
save_path = os.path.join(SAVE_DIR, "V19_full.pt")
torch.save({"p7":p7.state_dict(),"p6":p6.state_dict(),"explore":explore.state_dict(),"meta":meta.state_dict()}, save_path)
w(f"\n[保存] {save_path}")

if exam_set:
    w("\n[考试集评测] (绝对隔离数据)...")
    t0 = time.time(); total_ok, total_n = 0, 0
    for Av, Bv, A, B in exam_set:
        with torch.no_grad():
            _, sv, _, _, _ = p7(Av, Bv, last_loss=0.0)
            pred_w = p6(sv.unsqueeze(0))[:, :len(B), :]
            pred_n = F.normalize(pred_w.squeeze(0), dim=-1)
            true_n = F.normalize(Bv, dim=-1)
            c = torch.mm(pred_n, true_n.T).argmax(dim=-1)
            total_ok += sum(1 for i in range(len(B)) if B[c[i].item()]==B[i])
            total_n += len(B)
    w(f"  考试集词准确率: {total_ok/total_n*100:.1f}% ({total_ok}/{total_n})")
    w(f"  评测时间: {time.time()-t0:.0f}s")
w(f"[日志] {LOG_PATH}")
