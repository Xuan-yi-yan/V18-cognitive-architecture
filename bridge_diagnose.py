"""
Bridge 专修诊断: P8+P6 长训 500轮, 每5轮完整诊断
==================================================
固定 P1/P5, 只训练 P8+P6, 找 78% 天花板的原因
"""
import torch, torch.nn.functional as F, time, os, sys, re, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P5_sentence.model import SentenceSynthesis
from P6_sent_word.model import SentToWordsDecoder
from P8_char_sent.model import CharToSent

SEED = 789; LOG_EVERY = 5; TOTAL_EPOCHS = 500
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

SAVE_DIR = os.path.join(BASE_DIR, "P1_char_word", "checkpoints")

# ============================================================
# 加载数据
# ============================================================
SENT_FILE = os.path.join(BASE_DIR, "P5_sentence", "sentences_v2.txt")
WORD_LIST = os.path.join(DATA_DIR, "word_list_v2.txt")

def load_word_list(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            m = re.match(r'!([^@]+)@(.)@(.)', line)
            if m: entries.append((m.group(1), m.group(2), m.group(3)))
            elif len(line) == 2: entries.append((line, line[0], line[1]))
    return entries

entries = load_word_list(WORD_LIST)
chars_set = set()
for _, c1, c2 in entries: chars_set.add(c1); chars_set.add(c2)
char_list = sorted(chars_set)
char2idx = {c: i for i, c in enumerate(char_list)}
word2idx = {w: i for i, (w, _, _) in enumerate(entries)}
idx2word = {i: w for w, i in word2idx.items()}

def load_sentences(path):
    role_map = {"subj": 0, "verb": 1, "obj": 2, "adj": 3, "adv": 4, "comp": 5}
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("|")
            if len(parts) != 2: continue
            words, roles = [], []
            for item in parts[1].split():
                try:
                    w, r = item.split(":")
                    words.append(w); roles.append(role_map.get(r, 0))
                except: pass
            if words: data.append((words, roles))
    return data

sentences = load_sentences(SENT_FILE)
print(f"Bridge Diagnose | Seed={SEED} | {TOTAL_EPOCHS}epochs | log/{LOG_EVERY}")

# ============================================================
# 加载冻结模型
# ============================================================
ckpt = torch.load(os.path.join(SAVE_DIR, "P1_best.pt"), map_location=DEVICE)
p1 = CharToWordModel(ckpt["num_chars"], ckpt["num_words"]).to(DEVICE)
p1.load_state_dict(ckpt["model_state_dict"])
for p in p1.parameters(): p.requires_grad = False; p1.eval()
print(f"P1: Top-1={ckpt.get('top1','?')}")

p5 = SentenceSynthesis().to(DEVICE)
p5_ckpt = torch.load(os.path.join(SAVE_DIR, "P5_best.pt"), map_location=DEVICE)
if "model_state_dict" in p5_ckpt: p5.load_state_dict(p5_ckpt["model_state_dict"])
for p in p5.parameters(): p.requires_grad = False; p5.eval()
print(f"P5: gap={p5_ckpt.get('avg_gap','?')}")

# ============================================================
# 构建桥接数据
# ============================================================
@torch.no_grad()
def enc_word(w):
    c1, c2 = w[0], w[0] if len(w)==1 else w[-1]
    if c1 not in char2idx or c2 not in char2idx: return None
    return p1(torch.tensor([[char2idx[c1], char2idx[c2]]], device=DEVICE), last_loss=0.0)[0]

def get_char_vecs(text):
    vecs = []
    for c in text:
        if c not in char2idx: return None
        content = p1.char_content(torch.tensor([char2idx[c]], device=DEVICE))
        pos = p1.pos_encoder.pe[0:1]
        internal = torch.cat([pos[0], content[0]], dim=-1)
        vecs.append(p1.project_char(internal))
    return torch.stack(vecs)

# 桥接数据: 句子→(词向量, 角色) 对, 字符向量在训练时实时计算
bridge_data = []
for words, roles in sentences:
    wvs = [enc_word(w) for w in words]
    if None in wvs: continue
    wvs_s = torch.stack(wvs)
    sv = p5(wvs_s, torch.tensor(roles, device=DEVICE), last_loss=0.0)
    bridge_data.append((words, sv, wvs_s))  # 存原始words, 训练时实时get_char_vecs
print(f"Bridge data: {len(bridge_data)} groups (char vecs computed on-the-fly)")

# ============================================================
# P8 + P6
# ============================================================
# 解冻P1投影层 — Bridge联合微调的关键!
for p in p1.output_proj.parameters():
    p.requires_grad = True
p1.train()  # 仅投影层有梯度

p8 = CharToSent(max_len=15).to(DEVICE)
p6 = SentToWordsDecoder(max_words=5).to(DEVICE)
n_proj = sum(p.numel() for p in p1.output_proj.parameters())
n_bridge = sum(p.numel() for m in [p8,p6] for p in m.parameters())
print(f"P8+P6: {n_bridge:,} + P1_proj: {n_proj:,} = {n_bridge+n_proj:,} params")
print(f"P1 projection UNFROZEN for joint fine-tuning (lr=0.0002)")

opt = torch.optim.Adam([
    {'params': list(p8.parameters()) + list(p6.parameters()), 'lr': 0.002},
    {'params': p1.output_proj.parameters(), 'lr': 0.0002},
], weight_decay=1e-5)
last_loss = 1.0

# ============================================================
# 诊断函数
# ============================================================
@torch.no_grad()
def diagnose(epoch, loss_avg):
    d = {}
    p8.eval(); p6.eval()

    # Sample 30 sentences
    sample = random.sample(bridge_data, min(30, len(bridge_data)))
    sent_cos = []; word_cos = []
    all_wp = []; all_wt = []

    for words, sv_t, wvs_t in sample:
        cvs = get_char_vecs(''.join(words))
        if cvs is None: continue
        sp = p8(cvs, last_loss=0.0)
        sent_cos.append(F.cosine_similarity(sp.unsqueeze(0), sv_t.unsqueeze(0), dim=-1).item())
        wp = p6(sp.unsqueeze(0), last_loss=0.0)[0]
        nw = min(len(wp), len(wvs_t))
        word_cos.append(F.cosine_similarity(wp[:nw], wvs_t[:nw], dim=-1).mean().item())
        all_wp.append(wp[:nw])
        all_wt.append(wvs_t[:nw])

    d['sent_cos'] = sum(sent_cos)/len(sent_cos)
    d['word_cos'] = sum(word_cos)/len(word_cos)

    # Output distribution
    wp_cat = torch.cat([w for w in all_wp])
    d['wp_norm'] = wp_cat.norm(dim=-1).mean().item()
    d['wp_mean'] = wp_cat.mean().item(); d['wp_std'] = wp_cat.std().item()

    # Collapse check: pairwise cosine among different sentence predictions
    sp_all = torch.stack([p8(get_char_vecs(''.join(w)), last_loss=0.0) for w, _, _ in sample[:15] if get_char_vecs(''.join(w)) is not None])
    sp_n = F.normalize(sp_all, dim=-1)
    pair_cos = torch.mm(sp_n, sp_n.T)
    mask = 1 - torch.eye(len(sp_n), device=DEVICE)
    d['sent_self_sim'] = (pair_cos * mask).sum().item() / max(mask.sum().item(), 1)

    # Gradient norms
    d['grad_p8'] = sum(p.grad.norm().item() for p in p8.parameters() if p.grad is not None)
    d['grad_p6'] = sum(p.grad.norm().item() for p in p6.parameters() if p.grad is not None)

    # P8 explore/meta state
    if hasattr(p8, 'explore_state'):
        es = p8.explore_state
        d['p8_explore_norm'] = es.norm().item()
        d['p8_meta_out'] = p8.meta_fc(es * min(last_loss*20,1.0)).norm().item()

    # P6 explore/meta state
    if hasattr(p6, 'explore_state'):
        es6 = p6.explore_state
        d['p6_explore_norm'] = es6.norm().item()

    # Top-k analysis: what fraction of word vectors have cos>0.8 with target?
    d['word_cos_gt08'] = sum(1 for c in word_cos if c > 0.8) / len(word_cos)
    d['word_cos_gt09'] = sum(1 for c in word_cos if c > 0.9) / len(word_cos)

    p8.train(); p6.train()
    return d

# ============================================================
# 训练
# ============================================================
t0 = time.time(); history = []; best_wc = 0.0; no_improve = 0

for epoch in range(1, TOTAL_EPOCHS + 1):
    el = 0.0; nb = 0
    random.shuffle(bridge_data)
    for words, sv_t, wvs_t in bridge_data:
        cvs = get_char_vecs(''.join(words))  # 实时计算, 投影层梯度可流通
        if cvs is None: continue
        sp = p8(cvs, last_loss=last_loss)
        wp = p6(sp.unsqueeze(0), last_loss=last_loss)[0]
        sl = (1.0 - F.cosine_similarity(sp.unsqueeze(0), sv_t.unsqueeze(0), dim=-1)).mean()
        nw = min(len(wp), len(wvs_t))
        wl = (1.0 - F.cosine_similarity(wp[:nw], wvs_t[:nw], dim=-1)).mean()
        loss = sl + wl
        opt.zero_grad(); loss.backward(); opt.step()
        el += (sl+wl).item(); nb += 1; last_loss = (sl+wl).item()
    torch.cuda.empty_cache()

    if epoch % LOG_EVERY == 0 or epoch == 1:
        d = diagnose(epoch, el/nb)
        d['epoch'] = epoch; d['loss'] = el/nb
        history.append(d)

        if d['word_cos'] > best_wc:
            best_wc = d['word_cos']; no_improve = 0
            torch.save({"p8": p8.state_dict(), "p6": p6.state_dict(),
                       "epoch": epoch, "word_cos": d['word_cos'], "sent_cos": d['sent_cos']},
                       os.path.join(SAVE_DIR, "Bridge_diag_best.pt"))
        else: no_improve += LOG_EVERY

        improved = "↑" if len(history) >= 2 and d['word_cos'] > history[-2]['word_cos'] else ("=" if len(history) >= 2 else " ")
        print(f"E{epoch:4d} | Loss={d['loss']:.4f} | sent={d['sent_cos']:.4f} word={d['word_cos']:.4f} {improved} | "
              f"wp_norm={d['wp_norm']:.2f} | sent_self={d['sent_self_sim']:.3f} | "
              f"gP8={d['grad_p8']:.3f} gP6={d['grad_p6']:.3f} | "
              f"wc>.8={d['word_cos_gt08']:.2f} wc>.9={d['word_cos_gt09']:.2f} | "
              f"best={best_wc:.4f}")

        if no_improve >= 150:
            print(f"Early stop @ {epoch} (no improvement for 150 epochs)")
            break

elapsed = time.time() - t0

# ============================================================
# 最终报告
# ============================================================
print(f"\n{'='*60}")
print(f"Bridge Diagnose Complete ({elapsed:.0f}s, {len(history)} snapshots)")
print(f"{'='*60}")
print(f"Best word_cos: {best_wc:.4f}")
print(f"Final word_cos: {history[-1]['word_cos']:.4f}")

# Trend analysis
wc_trend = [h['word_cos'] for h in history]
print(f"Word_cos trend: {wc_trend[0]:.4f} -> {wc_trend[-1]:.4f} " +
      (f"(+{wc_trend[-1]-wc_trend[0]:.4f})" if wc_trend[-1] > wc_trend[0] else f"({wc_trend[-1]-wc_trend[0]:.4f})"))

# Improvement rate
if len(wc_trend) >= 5:
    early = sum(wc_trend[:5])/5
    late = sum(wc_trend[-5:])/5
    rate = (late - early) / len(wc_trend)
    if rate > 0.0001:
        print(f"Still improving: {rate:.6f}/snapshot — more epochs might help")
    elif rate > 0:
        print(f"Nearly plateaued: {rate:.6f}/snapshot — diminishing returns")
    else:
        print(f"Plateaued or declining — structural bottleneck, not epoch count")

# Save full history
torch.save({"history": history, "best_wc": best_wc, "p8": p8.state_dict(), "p6": p6.state_dict()},
           os.path.join(SAVE_DIR, "Bridge_diag_full.pt"))
print(f"\nSaved: {SAVE_DIR}/Bridge_diag_full.pt")
