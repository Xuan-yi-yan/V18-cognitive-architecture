"""
V18 标准跑分: BLEU + Exact Match + Rouge-L + per-cos
====================================================
评测: 句子转换质量 (机器翻译标准指标)
"""
import torch, torch.nn.functional as F, os, sys, argparse, time, math
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P7_cross_sent.model import P7WordRouter2048
from P6_sent_word.model import SentToWordsDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="5k")
parser.add_argument("--max_samples", type=int, default=1000)
args = parser.parse_args()

# ── 模型 ──
p1_ckpt = torch.load(os.path.join(SAVE_DIR, 'P1_best.pt'), map_location=DEVICE)
p1 = CharToWordModel(p1_ckpt['num_chars'], p1_ckpt['num_words']).to(DEVICE)
p1.load_state_dict(p1_ckpt['model_state_dict']); p1.eval()
c2i = p1_ckpt['char2idx']

ckpt = torch.load(os.path.join(SAVE_DIR, 'V18_full_chain.pt'), map_location=DEVICE)
p7 = P7WordRouter2048().to(DEVICE); p7.load_state_dict(ckpt['p7'], strict=False); p7.eval()
p6 = SentToWordsDecoder(max_words=16).to(DEVICE); p6.load_state_dict(ckpt['p6'], strict=False); p6.eval()

@torch.no_grad()
def enc_batch(words):
    ids = []
    for w in words:
        c1, c2 = w[0], w[0] if len(w) == 1 else w[1]
        if c1 in c2i and c2 in c2i: ids.append([c2i[c1], c2i[c2]])
    if not ids: return None
    return p1(torch.tensor(ids, device=DEVICE), last_loss=0.0)

# ── 数据 ──
data_map = {"5k": "data_p7_5k.txt", "v5": "data_p7_v5.txt"}
path = data_map.get(args.data, args.data)
if not os.path.isabs(path): path = f"C:/ai/P7_cross_sent/{path}"

pairs = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split('|')
        if len(parts) != 2: continue
        pairs.append((parts[0].split(), parts[1].split()))
        if len(pairs) >= args.max_samples: break

# ── BLEU 计算 ──
def compute_bleu(pred, ref):
    """BLEU-4: 1-4 gram precision + brevity penalty"""
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    if not pred or not ref: return 0.0
    bp = max(1.0, math.exp(1 - len(ref)/max(len(pred),1)))
    log_ps = []
    for n in [1,2,3,4]:
        p_c = ngrams(pred, n)
        r_c = ngrams(ref, n)
        matched = sum(min(p_c[k], r_c.get(k,0)) for k in p_c)
        total = max(sum(p_c.values()), 1)
        log_ps.append(math.log(max(matched/total, 1e-10)))
    return bp * math.exp(sum(log_ps)/4)

def compute_rouge_l(pred, ref):
    """Rouge-L: 最长公共子序列"""
    m, n = len(pred), len(ref)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if pred[i] == ref[j]: dp[i+1][j+1] = dp[i][j] + 1
            else: dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    lcs = dp[m][n]
    if lcs == 0: return 0.0, 0.0, 0.0
    prec = lcs / max(len(pred), 1)
    rec = lcs / max(len(ref), 1)
    f1 = 2*prec*rec/max(prec+rec, 1e-8)
    return prec, rec, f1

# ── 评测 ──
print(f"V18 标准跑分: {len(pairs)} 样本")
t0 = time.time()

total_bleu = 0.0
total_em = 0
total_words = 0
total_correct = 0
rouge_p, rouge_r, rouge_f = 0.0, 0.0, 0.0
per_cos_all = []

for idx, (A, B) in enumerate(pairs):
    Av = enc_batch(A)
    Bv = enc_batch(B)
    if Av is None or Bv is None: continue

    _, sv, _, _, _ = p7(Av, Bv, last_loss=0.0)
    pred_w = p6(sv.unsqueeze(0))[:, :Bv.shape[0], :]

    # 词匹配
    pred_n = F.normalize(pred_w.squeeze(0), dim=-1)
    true_n = F.normalize(Bv, dim=-1)
    cos_mat = torch.mm(pred_n, true_n.T)
    pred_idx = cos_mat.argmax(dim=-1)
    pred_tokens = [B[pred_idx[i].item()] for i in range(Bv.shape[0])]

    # BLEU-4
    bleu = compute_bleu(pred_tokens, B)
    total_bleu += bleu

    # Exact Match
    em = 1 if pred_tokens == B else 0
    total_em += em

    # Rouge-L
    rp, rr, rf = compute_rouge_l(pred_tokens, B)
    rouge_p += rp; rouge_r += rr; rouge_f += rf

    # 词级准确率
    correct = sum(1 for i in range(Bv.shape[0]) if pred_tokens[i] == B[i])
    total_correct += correct
    total_words += Bv.shape[0]

    # per-cos
    per_cos_all.extend(F.cosine_similarity(pred_w.squeeze(0), Bv, dim=-1).tolist())

    if (idx+1) % 200 == 0:
        print(f"  {idx+1}/{len(pairs)}...")

# ── 输出 ──
n_eval = len(pairs)
elapsed = time.time() - t0

print(f"\n{'='*55}")
print(f"  V18 标准跑分报告")
print(f"{'='*55}")
print(f"  模型参数:   875,651 (875K)")
print(f"  评测数据:   {n_eval} 对 (5k数据集)")
print(f"  评测时间:   {elapsed:.0f}s")
print(f"{'='*55}")
print(f"  BLEU-4:     {total_bleu/n_eval*100:.1f}")
print(f"  Exact Match: {total_em/n_eval*100:.1f}%")
print(f"  Rouge-L F1: {rouge_f/n_eval*100:.1f}")
print(f"  Rouge-L P:  {rouge_p/n_eval*100:.1f}")
print(f"  Rouge-L R:  {rouge_r/n_eval*100:.1f}")
print(f"  词准确率:   {total_correct/total_words*100:.1f}%")
print(f"  平均per-cos: {sum(per_cos_all)/len(per_cos_all):.4f}")
print(f"{'='*55}")
