"""
V18 基准评测脚本
===============
支持: 自建数据 / LCQMC / 任意A|B格式
输出: 准确率, per-cos均值, sent_vec相似度分布
"""
import torch, torch.nn.functional as F, os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P7_cross_sent.model import P7WordRouter2048
from P6_sent_word.model import SentToWordsDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="5k", help="5k/v5/或文件路径")
parser.add_argument("--max_samples", type=int, default=500)
args = parser.parse_args()

# ── 加载模型 ──
print("加载模型...")
p1_ckpt = torch.load(os.path.join(SAVE_DIR, 'P1_best.pt'), map_location=DEVICE)
p1 = CharToWordModel(p1_ckpt['num_chars'], p1_ckpt['num_words']).to(DEVICE)
p1.load_state_dict(p1_ckpt['model_state_dict']); p1.eval()
c2i = p1_ckpt['char2idx']

ckpt = torch.load(os.path.join(SAVE_DIR, 'V18_full_chain.pt'), map_location=DEVICE)
p7 = P7WordRouter2048().to(DEVICE); p7.load_state_dict(ckpt['p7'], strict=False); p7.eval()
p6 = SentToWordsDecoder(max_words=16).to(DEVICE); p6.load_state_dict(ckpt['p6'], strict=False); p6.eval()

# ── 编码 ──
@torch.no_grad()
def enc_batch(words):
    ids = []
    for w in words:
        c1, c2 = w[0], w[0] if len(w) == 1 else w[1]
        if c1 in c2i and c2 in c2i: ids.append([c2i[c1], c2i[c2]])
    if not ids: return None
    return p1(torch.tensor(ids, device=DEVICE), last_loss=0.0)

# ── 加载数据 ──
print("加载数据...")
data_map = {"5k": "data_p7_5k.txt", "v5": "data_p7_v5.txt"}
path = data_map.get(args.data, args.data)
if not os.path.isabs(path):
    path = f"C:/ai/P7_cross_sent/{path}"

pairs = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split('|')
        if len(parts) != 2: continue
        pairs.append((parts[0].split(), parts[1].split()))
        if len(pairs) >= args.max_samples: break

print(f"评测样本: {len(pairs)}")

# ── 评测 ──
total_correct = 0
total_words = 0
sent_cos_list = []
per_cos_list = []
t0 = time.time()

for idx, (A, B) in enumerate(pairs):
    Av = enc_batch(A)
    Bv = enc_batch(B)
    if Av is None or Bv is None: continue

    _, sv, _, _, _ = p7(Av, Bv, last_loss=0.0)
    nB = Bv.shape[0]
    pred_w = p6(sv.unsqueeze(0))[:, :nB, :]

    # 词级准确率
    pred_n = F.normalize(pred_w.squeeze(0), dim=-1)
    true_n = F.normalize(Bv, dim=-1)
    cos_mat = torch.mm(pred_n, true_n.T)
    pred_idx = cos_mat.argmax(dim=-1)
    correct = sum(1 for i in range(nB) if i < len(B) and B[pred_idx[i].item()] == B[i])
    total_correct += correct
    total_words += nB

    # per-cos
    per_cos = F.cosine_similarity(pred_w.squeeze(0), Bv, dim=-1).tolist()
    per_cos_list.extend(per_cos)

    # sent_vec相似度
    B_sent = p7.sent_proj(Bv.mean(dim=0, keepdim=True))
    sv_cos = (F.normalize(sv.unsqueeze(0), dim=-1) * F.normalize(B_sent, dim=-1)).sum().item()
    sent_cos_list.append(sv_cos)

    if (idx + 1) % 100 == 0:
        elapsed = time.time() - t0
        print(f"  {idx+1}/{len(pairs)} | 词准确率={total_correct/total_words*100:.1f}% | {elapsed:.0f}s")

# ── 总结 ──
elapsed = time.time() - t0
word_acc = total_correct / max(total_words, 1)
avg_per_cos = sum(per_cos_list) / max(len(per_cos_list), 1)
avg_sent_cos = sum(sent_cos_list) / max(len(sent_cos_list), 1)

print(f"\n{'='*50}")
print(f"评测完成 ({elapsed:.0f}s)")
print(f"  样本数: {len(pairs)}")
print(f"  词级准确率: {word_acc*100:.1f}%")
print(f"  平均per-cos: {avg_per_cos:.4f}")
print(f"  sent_vec cos: {avg_sent_cos:.4f}")
print(f"{'='*50}")
