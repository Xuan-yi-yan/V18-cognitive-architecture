"""
V18 v7.0: MetaModulator — 头温度调制 (改信息路径, 不改幅度)
==============================================================
v5.1 Staged(*/->+-) 主干 + MetaModulator 控制注意力温度

MetaModulator:
  输入: loss_ema(平滑loss) + loss_delta(loss变化趋势)
  输出: head temperature factor ∈ [0.95, 1.05]
  作用: attn_scores = attn_scores / temperature (改softmax锐度)

  温度>1: 注意力更均匀 (探索模式)
  温度<1: 注意力更集中 (利用模式)

层级分配:
  P1: v5.1 Staged, 不调制 (底层保协议)
  P2: 标准, 不调制
  P3: 标准, 不调制
  P5: 头温度调制 (中层软门控)
  Bridge: 头温度调制 (重点试验场)
  P7: 标准, 不调制 (验收)
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, os, sys, re, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P2_word_char.model import WordToCharDecoder
from P3_word_attr.model import SubjectBindingModel, margin_loss
from P5_sentence.model import SentenceSynthesis, contrastive_loss
from P6_sent_word.model import SentToWordsDecoder
from P8_char_sent.model import CharToSent
from P7_cross_sent.model import CrossSentenceRouter

# ============================================================
# MetaModulator: loss状态 → 头温度
# ============================================================
class MetaModulator(nn.Module):
    """根据训练状态输出注意力温度调制因子 [0.95, 1.05]"""
    def __init__(self, hidden=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Tanh()
        )
        self.loss_ema = 1.0  # 平滑后的loss

    def update_ema(self, loss, alpha=0.1):
        self.loss_ema = alpha * loss + (1 - alpha) * self.loss_ema

    def forward(self, loss, device):
        loss_delta = loss - self.loss_ema
        x = torch.tensor([[self.loss_ema, loss_delta]], device=device)
        factor = 1.0 + 0.05 * self.mlp(x).squeeze()  # [0.95, 1.05]
        return factor

# ============================================================
# 配置
# ============================================================
WORD_LIST = os.path.join(DATA_DIR, "word_list_v2.txt")
SENT_FILE = os.path.join(BASE_DIR, "P5_sentence", "sentences_v2.txt")

P1_MINI_BATCH = 64; P1_ACCUM = 25
P1_EPOCHS = 300; P2_EPOCHS = 200; P3_EPOCHS = 200
P5_EPOCHS = 300; BRIDGE_EPOCHS = 500; P7_EPOCHS = 300
PATIENCE = 30

print(f"{'='*60}")
print(f"V18 v7.0: MetaModulator + Head Temperature")
print(f"{'='*60}")

# ============================================================
# 数据加载
# ============================================================
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
idx2char = {i: c for c, i in char2idx.items()}
word2idx = {w: i for i, (w, _, _) in enumerate(entries)}
idx2word = {i: w for w, i in word2idx.items()}
all_pairs = [[char2idx[c1], char2idx[c2]] for _, c1, c2 in entries]
NUM_CHARS = len(char_list); NUM_WORDS = len(entries)

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
print(f"[数据] {NUM_WORDS}词 | {NUM_CHARS}字 | {len(sentences)}句")

# ============================================================
# 工具
# ============================================================
def pearson_loss(pred, target):
    pm = pred.mean(dim=0, keepdim=True); tm = target.mean(dim=0, keepdim=True)
    pc = pred - pm; tc = target - tm
    num = (pc * tc).sum()
    den = torch.sqrt((pc**2).sum() * (tc**2).sum() + PEARSON_EPSILON)
    return 1.0 - num / den

def cleanup(*objs):
    for obj in objs:
        if obj is not None: del obj
    torch.cuda.empty_cache()

# ============================================================
# P1: v5.1 Staged (不变)
# ============================================================
print(f"\n{'#'*60}")
print(f"# P1: v5.1 StagedProjection ({P1_EPOCHS}epochs)")
print(f"{'#'*60}")

p1 = CharToWordModel(NUM_CHARS, NUM_WORDS).to(DEVICE)
n_p1 = sum(p.numel() for p in p1.parameters())
print(f"[P1] {n_p1:,} params")

opt = torch.optim.Adam(p1.parameters(), lr=0.005, weight_decay=WEIGHT_DECAY)
last_loss = 1.0; best_top1 = 0.0; no_improve = 0

@torch.no_grad()
def eval_p1():
    p1.eval()
    ref = p1.get_all_reference_vectors(DEVICE); ref_n = F.normalize(ref, dim=-1)
    correct = 0; total = len(all_pairs)
    for i in range(0, total, 100):
        end = min(i+100, total)
        batch_p = torch.tensor([all_pairs[j] for j in range(i, end)], device=DEVICE)
        preds_n = F.normalize(p1(batch_p, last_loss=0.0), dim=-1)
        sims = torch.mm(preds_n, ref_n.T)
        correct += (sims.argmax(dim=-1) == torch.arange(i, end, device=DEVICE)).sum().item()
    p1.train()
    return correct / total

t0 = time.time()
for epoch in range(1, P1_EPOCHS + 1):
    epoch_loss = 0.0; nb = 0
    opt.zero_grad()
    perm = torch.randperm(NUM_WORDS)
    for acc_step in range(P1_ACCUM):
        start = acc_step * P1_MINI_BATCH
        end = min(start + P1_MINI_BATCH, NUM_WORDS)
        if start >= NUM_WORDS: break
        idxs = perm[start:end]
        pair_ids = torch.tensor([all_pairs[i] for i in idxs], device=DEVICE)
        word_ids = torch.tensor([i for i in idxs], device=DEVICE)
        pred, _ = p1(pair_ids, last_loss=last_loss, return_details=True)
        target = p1.get_word_target(word_ids)
        loss = pearson_loss(pred, target) / P1_ACCUM
        loss.backward()
        epoch_loss += loss.item() * P1_ACCUM; nb += 1
        last_loss = loss.item() * P1_ACCUM
    torch.nn.utils.clip_grad_norm_(p1.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    torch.cuda.empty_cache()
    if epoch % 5 == 0 or epoch == 1:
        top1 = eval_p1()
        if top1 > best_top1: best_top1 = top1; no_improve = 0
        else: no_improve += 5
        torch.save({"epoch": epoch, "model_state_dict": p1.state_dict(),
                   "num_chars": NUM_CHARS, "num_words": NUM_WORDS,
                   "char2idx": char2idx, "idx2char": idx2char,
                   "word2idx": word2idx, "idx2word": idx2word, "top1": top1},
                   os.path.join(SAVE_DIR, "P1_best.pt"))
        print(f"  E{epoch:4d} | Loss={epoch_loss/max(nb,1):.6f} | Top-1={top1:.4%} | best={best_top1:.4%} | {time.time()-t0:.0f}s")
        if no_improve >= PATIENCE: break

print(f"[P1] DONE: best={best_top1:.4%}")
ckpt = torch.load(os.path.join(SAVE_DIR, "P1_best.pt"), map_location=DEVICE)
p1.load_state_dict(ckpt["model_state_dict"])
for p in p1.parameters(): p.requires_grad = False; p1.eval()
cleanup(opt)

# ============================================================
# P2: 标准 (不调制)
# ============================================================
print(f"\n{'#'*60}")
print(f"# P2: 标准 ({P2_EPOCHS}epochs)")
print(f"{'#'*60}")

p2 = WordToCharDecoder().to(DEVICE)
print(f"[P2] {sum(p.numel() for p in p2.parameters()):,} params")
opt = torch.optim.Adam(p2.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
last_loss = 1.0; best_p2 = 0.0

t0 = time.time()
for epoch in range(1, P2_EPOCHS + 1):
    el = 0.0; s1 = 0.0; s2 = 0.0; nb = 0
    perm = torch.randperm(NUM_WORDS)
    for bs in range(0, NUM_WORDS, BATCH_SIZE):
        be = min(bs + BATCH_SIZE, NUM_WORDS)
        idxs = perm[bs:be]
        pair_ids = torch.tensor([all_pairs[i] for i in idxs], device=DEVICE)
        with torch.no_grad():
            _, _, full = p1.get_char_vectors(pair_ids)
            real_c1 = p1.project_char(full[:, 0, :])
            real_c2 = p1.project_char(full[:, 1, :])
            word_vec = p1(pair_ids, last_loss=last_loss)
        pred_c1, pred_c2 = p2(word_vec)
        sim1 = F.cosine_similarity(pred_c1, real_c1, dim=-1).mean()
        sim2 = F.cosine_similarity(pred_c2, real_c2, dim=-1).mean()
        loss = (1.0 - (sim1 + sim2) / 2.0)
        opt.zero_grad(); loss.backward(); opt.step()
        el += loss.item(); s1 += sim1.item(); s2 += sim2.item(); nb += 1
        last_loss = loss.item()
    torch.cuda.empty_cache()
    if epoch % 10 == 0 or epoch == 1:
        avg = (s1 + s2) / (2 * nb)
        if avg > best_p2: best_p2 = avg
        print(f"  E{epoch:4d} | Loss={el/nb:.6f} | cos={avg:.4%} | best={best_p2:.4%}")

torch.save({"model_state_dict": p2.state_dict(), "avg_cos": best_p2}, os.path.join(SAVE_DIR, "P2_best.pt"))
cleanup(p2, opt)
print(f"[P2] DONE: best={best_p2:.4%}")

# ============================================================
# P3: 七属性 (标准)
# ============================================================
print(f"\n{'#'*60}")
print(f"# P3: 七属性 ({P3_EPOCHS}epochs/attr)")
print(f"{'#'*60}")

ATTRS = ["主语", "谓语", "宾语", "定语", "状语", "补语", "虚词"]
attr_data = {}
for f in sorted(os.listdir(SAVE_DIR)):
    if not f.startswith('P3_') or not f.endswith('_best.pt'): continue
    try:
        cd = torch.load(os.path.join(SAVE_DIR, f), map_location='cpu', weights_only=False)
        a = cd.get('attr', ''); fw = cd.get('family_words', [])
        for attr in ATTRS:
            if attr in a and attr not in attr_data and len(fw) > 5:
                attr_data[attr] = fw; break
    except: pass

def _get_chars(w):
    if len(w) == 1: return w[0], w[0]
    return w[0], w[-1]

p3_results = {}
for attr in ATTRS:
    raw = attr_data.get(attr)
    if not raw: print(f"  [P3-{attr}] skip"); continue
    attr_words = [w for w in raw if _get_chars(w)[0] in char2idx and _get_chars(w)[-1] in char2idx]
    non_attr = [w for w in idx2word.values() if w not in set(attr_words) and w[0] in char2idx]
    @torch.no_grad()
    def encode(ws):
        vecs = []
        for w in ws:
            c1, c2 = _get_chars(w)
            cids = torch.tensor([[char2idx[c1], char2idx[c2]]], device=DEVICE)
            vecs.append(p1(cids, last_loss=0.0)[0])
        return torch.stack(vecs) if vecs else torch.zeros(0, WORD_DIM, device=DEVICE)
    family_p1 = encode(attr_words); family_proto = family_p1.mean(dim=0)
    p3_w2id = {w: i for i, w in enumerate(idx2word.values())}
    pos_ids = torch.tensor([p3_w2id[w] for w in attr_words if w in p3_w2id], device=DEVICE)
    neg_ids = torch.tensor([p3_w2id[w] for w in non_attr if w in p3_w2id], device=DEVICE)
    model = SubjectBindingModel(len(p3_w2id)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    last_loss = 1.0; best_acc = 0.0; half = BATCH_SIZE // 2
    for epoch in range(1, P3_EPOCHS + 1):
        el = 0.0; nb = 0
        if len(pos_ids) > 0 and len(neg_ids) > 0:
            pi = pos_ids[torch.randperm(len(pos_ids))[:half]]
            ni = neg_ids[torch.randperm(len(neg_ids))[:half]]
            ids = torch.cat([pi, ni])
            is_pos = torch.tensor([True]*len(pi)+[False]*len(ni), device=DEVICE)
            out, attn, q_raw = model(ids, family_p1, last_loss=last_loss)
            loss, _ = margin_loss(out, family_proto, is_pos, q_raw=q_raw)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1; last_loss = loss.item()
        if epoch % 40 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                ps = model.binding_score(pos_ids, family_p1) if len(pos_ids)>0 else torch.tensor([0.])
                ns = model.binding_score(neg_ids, family_p1) if len(neg_ids)>0 else torch.tensor([0.])
            model.train()
            pm, nm = ps.mean().item(), ns.mean().item()
            th = (pm+nm)/2
            acc = ((ps>th).float().mean()+(ns<=th).float().mean())/2 if len(ps)*len(ns)>0 else 0
            if acc > best_acc: best_acc = acc
            print(f"  P3-{attr} E{epoch:4d} | Loss={el/max(nb,1):.6f} | Acc={acc:.2%}")
    p3_results[attr] = (pm, nm, best_acc)
    cleanup(model, opt); torch.cuda.empty_cache()
    print(f"  [P3-{attr}] DONE: Acc={best_acc:.2%}")

# ============================================================
# P5: 中层MetaModulator — 头温度调制
# ============================================================
print(f"\n{'#'*60}")
print(f"# P5: MetaModulator 头温度 ({P5_EPOCHS}epochs)")
print(f"{'#'*60}")

@torch.no_grad()
def enc_word(w):
    c1, c2 = w[0], w[0] if len(w)==1 else w[-1]
    if c1 not in char2idx or c2 not in char2idx: return None
    return p1(torch.tensor([[char2idx[c1], char2idx[c2]]], device=DEVICE), last_loss=0.0)[0]

encoded = []
for words, roles in sentences:
    vecs = [enc_word(w) for w in words]
    if None not in vecs:
        encoded.append((torch.stack(vecs), torch.tensor(roles, device=DEVICE)))
print(f"[P5] {len(encoded)} valid")

def scrambled(wv, roles):
    idx = list(range(len(roles))); random.shuffle(idx)
    return torch.stack([wv[i] for i in idx]), torch.tensor([roles[i] for i in idx], device=DEVICE)

p5 = SentenceSynthesis().to(DEVICE)
meta_p5 = MetaModulator().to(DEVICE)
print(f"[P5] {sum(p.numel() for p in p5.parameters()):,} params + MetaModulator")

opt = torch.optim.Adam(list(p5.parameters()) + list(meta_p5.parameters()), lr=0.002, weight_decay=1e-5)
last_loss = 1.0; best_gap = -999; no_improve = 0

t0 = time.time()
for epoch in range(1, P5_EPOCHS + 1):
    el = 0.0; nb = 0
    random.shuffle(encoded)
    for wv, roles in encoded:
        # MetaModulator: 根据loss状态输出温度因子
        temp = meta_p5(last_loss, DEVICE)

        correct = p5(wv, roles, last_loss=last_loss)
        sv, sr = scrambled(wv, roles)
        scr = p5(sv, sr, last_loss=last_loss)

        # 头温度调制: 在对比损失前微调句子向量
        # 温度>1 → 向量更平滑, 温度<1 → 向量更锐利
        correct_mod = F.normalize(correct * temp, dim=-1)
        scr_mod = F.normalize(scr * temp, dim=-1)

        loss, cos_c, cos_s = contrastive_loss(correct_mod, scr_mod.unsqueeze(0))
        opt.zero_grad(); loss.backward(); opt.step()
        el += loss.item(); nb += 1
        meta_p5.update_ema(loss.item())
        last_loss = loss.item()
    torch.cuda.empty_cache()
    if epoch % 10 == 0 or epoch == 1:
        gaps = []
        for wv, roles in encoded[:30]:
            temp = meta_p5(last_loss, DEVICE)
            c = p5(wv, roles, last_loss=0.0)
            sv, sr = scrambled(wv, roles)
            s = p5(sv, sr, last_loss=0.0)
            c_mod = F.normalize(c * temp, dim=-1)
            s_mod = F.normalize(s * temp, dim=-1)
            cc = F.cosine_similarity(c_mod.detach(), c_mod.detach(), dim=-1).item()
            cs = F.cosine_similarity(s_mod, c_mod.detach(), dim=-1).item()
            gaps.append(cc - cs)
        avg_g = sum(gaps)/len(gaps)
        if avg_g > best_gap: best_gap = avg_g; no_improve = 0
        else: no_improve += 10
        print(f"  E{epoch:4d} | Loss={el/nb:.6f} | gap={avg_g:.4f} | temp={temp.item():.4f} | best={best_gap:.4f}")
        if no_improve >= PATIENCE: break

for p in p5.parameters(): p.requires_grad = False; p5.eval()
cleanup(opt)
print(f"[P5] DONE: best_gap={best_gap:.4f}")

# ============================================================
# Bridge: MetaModulator 头温度 (重点试验场)
# ============================================================
print(f"\n{'#'*60}")
print(f"# Bridge: MetaModulator 头温度 ({BRIDGE_EPOCHS}epochs)")
print(f"{'#'*60}")

@torch.no_grad()
def get_char_vecs(text):
    vecs = []
    for c in text:
        if c not in char2idx: return None
        content = p1.char_content(torch.tensor([char2idx[c]], device=DEVICE))
        pos = p1.pos_encoder.pe[0:1]
        internal = torch.cat([pos[0], content[0]], dim=-1)
        vecs.append(p1.project_char(internal))
    return torch.stack(vecs)

bridge_data = []
for words, roles in sentences:
    wvs = [enc_word(w) for w in words]
    if None in wvs: continue
    wvs_s = torch.stack(wvs)
    sv = p5(wvs_s, torch.tensor(roles, device=DEVICE), last_loss=0.0)
    cvs = get_char_vecs(''.join(words))
    if cvs is None: continue
    bridge_data.append((cvs, sv, wvs_s))
print(f"[Bridge] {len(bridge_data)} groups")

p8 = CharToSent(max_len=15).to(DEVICE)
p6 = SentToWordsDecoder(max_words=3).to(DEVICE)
meta_bridge = MetaModulator().to(DEVICE)
n_b = sum(p.numel() for m in [p8, p6, meta_bridge] for p in m.parameters())
print(f"[P8+P6+Meta] {n_b:,} params")

opt = torch.optim.Adam(list(p8.parameters()) + list(p6.parameters()) + list(meta_bridge.parameters()),
                       lr=0.002, weight_decay=1e-5)
last_loss = 1.0; best_wc = 0.0; no_improve = 0

t0 = time.time()
for epoch in range(1, BRIDGE_EPOCHS + 1):
    el = 0.0; nb = 0
    random.shuffle(bridge_data)
    for cvs, sv_t, wvs_t in bridge_data:
        temp = meta_bridge(last_loss, DEVICE)

        sp = p8(cvs, last_loss=last_loss)
        wp = p6(sp.unsqueeze(0), last_loss=last_loss)[0]

        # 头温度调制: 影响P8输出方向
        sp_mod = F.normalize(sp * temp, dim=-1)

        sl = (1.0 - F.cosine_similarity(sp_mod.unsqueeze(0), sv_t.unsqueeze(0), dim=-1)).mean()
        nw = min(len(wp), len(wvs_t))
        wl = (1.0 - F.cosine_similarity(wp[:nw], wvs_t[:nw], dim=-1)).mean()
        loss = sl + wl
        opt.zero_grad(); loss.backward(); opt.step()
        el += (sl+wl).item(); nb += 1
        meta_bridge.update_ema((sl+wl).item())
        last_loss = (sl+wl).item()
    torch.cuda.empty_cache()
    if epoch % 20 == 0 or epoch == 1:
        p8.eval(); p6.eval()
        sc = []; wc = []
        with torch.no_grad():
            for cvs, sv_t, wvs_t in bridge_data[:30]:
                temp = meta_bridge(last_loss, DEVICE)
                sp = p8(cvs, last_loss=0.0)
                sp_mod = F.normalize(sp * temp, dim=-1)
                sc.append(F.cosine_similarity(sp_mod.unsqueeze(0), sv_t.unsqueeze(0), dim=-1).item())
                wp = p6(sp.unsqueeze(0), last_loss=0.0)[0]
                nw = min(len(wp), len(wvs_t))
                wc.append(F.cosine_similarity(wp[:nw], wvs_t[:nw], dim=-1).mean().item())
        p8.train(); p6.train()
        avg_sc = sum(sc)/len(sc); avg_wc = sum(wc)/len(wc)
        if avg_wc > best_wc: best_wc = avg_wc; no_improve = 0
        else: no_improve += 20
        print(f"  E{epoch:4d} | Loss={el/nb:.4f} | sent={avg_sc:.4f} word={avg_wc:.4f} | temp={temp.item():.4f} | best={best_wc:.4f} | {time.time()-t0:.0f}s")
        if no_improve >= PATIENCE: break

cleanup(p8, p6, opt)
print(f"[Bridge] DONE: best_word={best_wc:.4f}")

# ============================================================
# P7: 跨句 (标准)
# ============================================================
print(f"\n{'#'*60}")
print(f"# P7: 跨句 ({P7_EPOCHS}epochs)")
print(f"{'#'*60}")

p7_data_path = os.path.join(BASE_DIR, "P7_cross_sent", "data_p7.txt")
p7_pairs = []
if os.path.exists(p7_data_path):
    with open(p7_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("|")
            if len(parts) == 2: p7_pairs.append((parts[0].split(), parts[1].split()))

if len(p7_pairs) > 0:
    B_vocab = sorted(set(w for _, B in p7_pairs for w in B))
    B_vecs = [enc_word(w) for w in B_vocab if enc_word(w) is not None]
    if len(B_vecs) >= 3:
        B_table = torch.stack(B_vecs)
        p7_enc = []
        for A, B in p7_pairs:
            Av = [enc_word(w) for w in A]; Bv = [enc_word(w) for w in B]
            if None in Av or None in Bv: continue
            B_sent = p5(torch.stack(Bv), torch.arange(len(Bv), device=DEVICE)%3, last_loss=0.0)
            p7_enc.append((torch.stack(Av), B_sent))
        print(f"[P7] {len(p7_enc)} valid pairs")
        if len(p7_enc) >= 3:
            p7 = CrossSentenceRouter().to(DEVICE)
            opt = torch.optim.Adam(p7.parameters(), lr=0.003, weight_decay=1e-5)
            last_loss = 1.0; best_p7 = 0.0
            for epoch in range(1, P7_EPOCHS + 1):
                el = 0.0; nb = 0
                for Av, Bt in p7_enc:
                    Bp, _ = p7(Av, B_table, last_loss=last_loss)
                    cos = F.cosine_similarity(Bp.unsqueeze(0), Bt.unsqueeze(0), dim=-1)
                    loss = (1.0 - cos).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    el += loss.item(); nb += 1; last_loss = loss.item()
                if epoch % 100 == 0 or epoch == 1:
                    p7.eval(); coses = []
                    with torch.no_grad():
                        for Av, Bt in p7_enc:
                            Bp, _ = p7(Av, B_table, last_loss=0.0)
                            coses.append(F.cosine_similarity(Bp.unsqueeze(0), Bt.unsqueeze(0), dim=-1).item())
                    p7.train()
                    avg_c = sum(coses)/len(coses)
                    if avg_c > best_p7: best_p7 = avg_c
                    print(f"  E{epoch:4d} | Loss={el/nb:.6f} | cos={avg_c:.4f} | best={best_p7:.4f}")
            cleanup(p7, opt)
            print(f"[P7] DONE: best={best_p7:.4f}")

# ============================================================
# 总结
# ============================================================
print(f"\n{'#'*60}")
print(f"# V18 v7.0 完成")
print(f"{'#'*60}")
print(f"P1: Top-1={best_top1:.4%}")
print(f"P2: cos={best_p2:.4%}")
for attr, (pm, nm, acc) in p3_results.items():
    print(f"P3-{attr}: Acc={acc:.2%}")
print(f"P5: gap={best_gap:.4f} (MetaModulator)")
print(f"Bridge: word_cos={best_wc:.4f} (MetaModulator)")
if 'best_p7' in dir(): print(f"P7: cos={best_p7:.4f}")
a = torch.cuda.memory_allocated(DEVICE)/1024**2
print(f"GPU: {a:.0f}MB")
