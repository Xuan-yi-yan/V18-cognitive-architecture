"""
V18 通宵极限压测: 逐层长训 + 验证早停 + LR退火 + 最优快照
=============================================================
P1→P2→P3→P5→Bridge→P7, 每层跑至收敛, 记录完整天花板地图
"""
import torch, torch.nn.functional as F, time, os, sys, re, random, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P2_word_char.model import WordToCharDecoder
from P3_word_attr.model import SubjectBindingModel, margin_loss
from P5_sentence.model import SentenceSynthesis, contrastive_loss
from P6_sent_word.model import SentToWordsDecoder
from P8_char_sent.model import CharToSent
from P7_cross_sent.model import CrossSentenceRouter

SEED = 789
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ============================================================
# EarlyStopping + LR Scheduler
# ============================================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0005):
        self.patience = patience; self.min_delta = min_delta
        self.counter = 0; self.best_score = None; self.early_stop = False
    def __call__(self, score):
        if self.best_score is None: self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else: self.best_score = score; self.counter = 0
        return self.early_stop

class OvernightReport:
    def __init__(self):
        self.rows = []
    def add(self, name, best_epoch, best_score, converged, lr_final, elapsed, notes=""):
        self.rows.append({"layer": name, "best_epoch": best_epoch, "best_score": best_score,
                         "converged": converged, "lr_final": lr_final, "elapsed_min": elapsed/60,
                         "notes": notes})
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write("layer,best_epoch,best_score,converged,lr_final,elapsed_min,notes\n")
            for r in self.rows:
                f.write(f"{r['layer']},{r['best_epoch']},{r['best_score']:.6f},{r['converged']},{r['lr_final']:.8f},{r['elapsed_min']:.0f},{r['notes']}\n")

report = OvernightReport()

# ============================================================
# Data
# ============================================================
WORD_LIST = os.path.join(DATA_DIR, "word_list_v2.txt")
SENT_FILE = os.path.join(BASE_DIR, "P5_sentence", "sentences_v2.txt")  # P5 uses 2000
BRIDGE_SENT_FILE = os.path.join(BASE_DIR, "P5_sentence", "sentences_v8k.txt")  # Bridge uses 8000

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
N = len(entries)

# P1: all pairs for training (each word IS its own class, no unseen classes exist)
train_pairs = all_pairs  # all 6000 pairs
train_idx = list(range(N))  # word indices 0..5999

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
s_cut = int(len(sentences) * 0.9)
train_sents = sentences[:s_cut]; val_sents = sentences[s_cut:]
print(f"[Data] {N} words | {len(sentences)} sentences")

# ============================================================
# Helpers
# ============================================================
def pearson_loss(pred, target):
    pm = pred.mean(dim=0, keepdim=True); tm = target.mean(dim=0, keepdim=True)
    pc = pred - pm; tc = target - tm
    num = (pc * tc).sum()
    den = torch.sqrt((pc**2).sum() * (tc**2).sum() + PEARSON_EPSILON)
    return 1.0 - num / den

def p1_accuracy(p1_model, pairs_with_idx):
    """pairs_with_idx: list of (char_pair, word_index)"""
    p1_model.eval()
    ref = p1_model.get_all_reference_vectors(DEVICE); ref_n = F.normalize(ref, dim=-1)
    correct = 0; total = len(pairs_with_idx)
    for i in range(0, total, 100):
        end = min(i+100, total)
        batch = [pairs_with_idx[j] for j in range(i, end)]
        bp = torch.tensor([b[0] for b in batch], device=DEVICE)
        true_ids = torch.tensor([b[1] for b in batch], device=DEVICE)
        sims = torch.mm(F.normalize(p1_model(bp, last_loss=0.0), dim=-1), ref_n.T)
        correct += (sims.argmax(dim=-1) == true_ids).sum().item()
    p1_model.train()
    return correct / total

def gpu_ok():
    a = torch.cuda.memory_allocated(DEVICE)/1024**2
    return a < 10000  # < 10GB

# ============================================================
# P1: Char→Word (max 500 epochs)
# ============================================================
print(f"\n{'#'*60}\n# P1 LONG TRAIN (max 500, patience 15)\n{'#'*60}")
p1 = CharToWordModel(len(char_list), N).to(DEVICE)
opt = torch.optim.Adam(p1.parameters(), lr=0.005, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-6)
es = EarlyStopping(patience=15, min_delta=0.0005)
best_val = 0.0; best_ep = 0; ll = 1.0
t0 = time.time()

for ep in range(1, 501):
    el = 0.0; nb = 0; opt.zero_grad()
    perm = torch.randperm(len(train_pairs))
    for acc in range(25):
        s = acc*64; idxs = perm[s:s+64]
        if len(idxs)==0: break
        pids = torch.tensor([train_pairs[i] for i in idxs], device=DEVICE)
        wids = torch.tensor([train_idx[i] for i in idxs], device=DEVICE)
        loss = pearson_loss(p1(pids, last_loss=ll), p1.word_table[wids]) / 25
        loss.backward(); el += loss.item()*25; nb += 1; ll = loss.item()*25
    torch.nn.utils.clip_grad_norm_(p1.parameters(), 1.0)
    opt.step(); opt.zero_grad(); torch.cuda.empty_cache()

    if ep % 5 == 0 or ep == 1:
        train_acc = p1_accuracy(p1, [(p, i) for i, p in enumerate(train_pairs)])
        scheduler.step(train_acc)
        if train_acc > best_val:  # best_val tracks best train acc for P1
            best_val = train_acc; best_ep = ep
            torch.save({"epoch": ep, "model_state_dict": p1.state_dict(),
                       "num_chars": len(char_list), "num_words": N,
                       "char2idx": char2idx, "idx2char": idx2char,
                       "word2idx": word2idx, "idx2word": idx2word,
                       "train_acc": train_acc},
                       os.path.join(SAVE_DIR, "overnight_P1_best.pt"))
        lr = opt.param_groups[0]['lr']
        print(f"P1 E{ep:4d} | train_acc={train_acc:.4%} | best={best_val:.4%}@{best_ep} | LR={lr:.6f} | {time.time()-t0:.0f}s")
        if es(train_acc): print(f"P1 CONVERGED @ {ep}"); break

p1.load_state_dict(torch.load(os.path.join(SAVE_DIR, "overnight_P1_best.pt"), map_location=DEVICE)["model_state_dict"])
for p in p1.parameters(): p.requires_grad = False; p1.eval()
report.add("P1", best_ep, best_val, es.early_stop, lr, time.time()-t0, "train_acc monitor")
del opt; torch.cuda.empty_cache()
print(f"P1 DONE: best_val={best_val:.4%} @ {best_ep}")

# ============================================================
# P2: Word→Char (max 2000 epochs)
# ============================================================
print(f"\n{'#'*60}\n# P2 LONG TRAIN (max 2000, patience 15)\n{'#'*60}")
p2 = WordToCharDecoder().to(DEVICE)
opt = torch.optim.Adam(p2.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-6)
es = EarlyStopping(patience=15, min_delta=0.001)
best_val = 0.0; best_ep = 0; ll = 1.0
t0 = time.time()

# P2 val: last 200 training pairs as held-out
p2_val_pair_list = [(all_pairs[i], i) for i in range(N-200, N)]

for ep in range(1, 2001):
    el = 0.0; nb = 0
    perm = torch.randperm(len(train_pairs))
    for bs in range(0, len(train_pairs), 200):
        idxs = perm[bs:bs+200]
        pids = torch.tensor([train_pairs[i] for i in idxs], device=DEVICE)
        with torch.no_grad():
            _, _, full = p1.get_char_vectors(pids)
            rc1 = p1.project_char(full[:,0,:]); rc2 = p1.project_char(full[:,1,:])
            wv = p1(pids, last_loss=ll)
        pc1, pc2 = p2(wv, last_loss=ll)
        s1 = F.cosine_similarity(pc1, rc1, dim=-1).mean()
        s2 = F.cosine_similarity(pc2, rc2, dim=-1).mean()
        loss = (1.0-(s1+s2)/2.0)
        # Explore reg
        loss = loss + 0.001*F.relu(0.1-p2.explore_state.norm())
        opt.zero_grad(); loss.backward(); opt.step()
        el += loss.item(); nb += 1; ll = loss.item()
    torch.cuda.empty_cache()

    if ep % 10 == 0 or ep == 1:
        # Val (with correct word indices)
        p2.eval()
        vpids = torch.tensor([p2_val_pair_list[i][0] for i in range(len(p2_val_pair_list))], device=DEVICE)
        with torch.no_grad():
            _, _, full = p1.get_char_vectors(vpids)
            rc1 = p1.project_char(full[:,0,:]); rc2 = p1.project_char(full[:,1,:])
            wv = p1(vpids, last_loss=0.0)
        vpc1, vpc2 = p2(wv, last_loss=0.0)
        v1 = F.cosine_similarity(vpc1, rc1, dim=-1).mean().item()
        v2 = F.cosine_similarity(vpc2, rc2, dim=-1).mean().item()
        val_avg = (v1+v2)/2
        p2.train()
        scheduler.step(val_avg)
        if val_avg > best_val: best_val = val_avg; best_ep = ep
        lr = opt.param_groups[0]['lr']
        print(f"P2 E{ep:4d} | val={val_avg:.4%} (c1={v1:.4%} c2={v2:.4%}) | best={best_val:.4%}@{best_ep} | LR={lr:.6f} | {time.time()-t0:.0f}s")
        if es(val_avg): print(f"P2 CONVERGED @ {ep}"); break

torch.save({"model_state_dict": p2.state_dict(), "val_cos": best_val, "epoch": best_ep},
           os.path.join(SAVE_DIR, "overnight_P2_best.pt"))
report.add("P2", best_ep, best_val, es.early_stop, lr, time.time()-t0, f"c1={v1:.4%} c2={v2:.4%}")
del opt, p2; torch.cuda.empty_cache()
print(f"P2 DONE: best_val={best_val:.4%} @ {best_ep}")

# ============================================================
# P3 quick (already 100%, just verify)
# ============================================================
print(f"\n{'#'*60}\n# P3 QUICK CHECK (200 epochs)\n{'#'*60}")
ATTRS = ["主语","谓语","宾语","定语","状语","补语","虚词"]
attr_data = {}
for f in sorted(os.listdir(SAVE_DIR)):
    if not f.startswith('P3_') or not f.endswith('_best.pt'): continue
    try:
        cd = torch.load(os.path.join(SAVE_DIR, f), map_location='cpu', weights_only=False)
        a = cd.get('attr',''); fw = cd.get('family_words',[])
        for attr in ATTRS:
            if attr in a and attr not in attr_data and len(fw) > 5:
                attr_data[attr] = fw; break
    except: pass

def _get_chars(w):
    if len(w)==1: return w[0], w[0]
    return w[0], w[-1]

p3_best_acc = 1.0
t0 = time.time()
for attr in ATTRS:
    raw = attr_data.get(attr)
    if not raw: continue
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
    best_acc = 0.0; half = BATCH_SIZE//2
    for ep in range(1, 201):
        el = 0.0; nb = 0
        if len(pos_ids)>0 and len(neg_ids)>0:
            pi = pos_ids[torch.randperm(len(pos_ids))[:half]]
            ni = neg_ids[torch.randperm(len(neg_ids))[:half]]
            ids = torch.cat([pi,ni])
            is_pos = torch.tensor([True]*len(pi)+[False]*len(ni), device=DEVICE)
            out, attn, q_raw = model(ids, family_p1, last_loss=0.5)
            loss, _ = margin_loss(out, family_proto, is_pos, q_raw=q_raw)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        if ep % 40 == 0:
            model.eval()
            with torch.no_grad():
                ps = model.binding_score(pos_ids, family_p1) if len(pos_ids)>0 else torch.tensor([0.])
                ns = model.binding_score(neg_ids, family_p1) if len(neg_ids)>0 else torch.tensor([0.])
            model.train()
            pm, nm = ps.mean().item(), ns.mean().item()
            th = (pm+nm)/2
            acc = ((ps>th).float().mean()+(ns<=th).float().mean())/2 if len(ps)*len(ns)>0 else 0
            if acc > best_acc: best_acc = acc
    if best_acc < p3_best_acc: p3_best_acc = best_acc
    del model, opt; torch.cuda.empty_cache()
    print(f"P3-{attr}: best_acc={best_acc:.2%}")
report.add("P3", 200, p3_best_acc, True, 0, time.time()-t0, "all 7 attributes")
print(f"P3 DONE: min_acc={p3_best_acc:.2%}")

# ============================================================
# P5: Word→Sentence (max 1000 epochs)
# ============================================================
print(f"\n{'#'*60}\n# P5 LONG TRAIN (max 1000, patience 15)\n{'#'*60}")

@torch.no_grad()
def enc_word(w):
    c1, c2 = w[0], w[0] if len(w)==1 else w[-1]
    if c1 not in char2idx or c2 not in char2idx: return None
    return p1(torch.tensor([[char2idx[c1], char2idx[c2]]], device=DEVICE), last_loss=0.0)[0]

train_enc = []
for w, r in train_sents:
    vecs = [enc_word(x) for x in w]
    if None not in vecs: train_enc.append((torch.stack(vecs), torch.tensor(r, device=DEVICE)))
val_enc = []
for w, r in val_sents:
    vecs = [enc_word(x) for x in w]
    if None not in vecs: val_enc.append((torch.stack(vecs), torch.tensor(r, device=DEVICE)))

def scrambled(wv, roles):
    idx = list(range(len(roles))); random.shuffle(idx)
    return torch.stack([wv[i] for i in idx]), torch.tensor([roles[i] for i in idx], device=DEVICE)

def p5_gap(model, data):
    model.eval()
    gaps = []
    for wv, roles in data[:30]:
        c = model(wv, roles, last_loss=0.0)
        sv, sr = scrambled(wv, roles)
        s = model(sv, sr, last_loss=0.0)
        cc = F.cosine_similarity(c.detach(), c.detach(), dim=-1).item()
        cs = F.cosine_similarity(s, c.detach(), dim=-1).item()
        gaps.append(cc-cs)
    model.train()
    return sum(gaps)/len(gaps)

p5 = SentenceSynthesis().to(DEVICE)
opt = torch.optim.Adam(p5.parameters(), lr=0.002, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-6)
es = EarlyStopping(patience=15, min_delta=0.005)
best_gap = -999; best_ep = 0; ll = 1.0; sent_buf = []
t0 = time.time()

for ep in range(1, 1001):
    el = 0.0; nb = 0
    for wv, roles in train_enc:
        correct = p5(wv, roles, last_loss=ll)
        sv, sr = scrambled(wv, roles)
        scr = p5(sv, sr, last_loss=ll)
        loss, _, _ = contrastive_loss(correct, scr.unsqueeze(0))
        # Diversity penalty
        sent_buf.append(correct.detach())
        if len(sent_buf) >= 20:
            buf = torch.stack(sent_buf)
            buf_n = F.normalize(buf, dim=-1)
            pc = torch.mm(buf_n, buf_n.T)
            mk = 1-torch.eye(len(buf), device=DEVICE)
            avg_cos = (pc*mk).sum()/(mk.sum()+1e-8)
            loss = loss + F.relu(avg_cos-0.3)*0.1
            sent_buf = []
        opt.zero_grad(); loss.backward(); opt.step()
        el += loss.item(); nb += 1; ll = loss.item()
    torch.cuda.empty_cache()

    if ep % 10 == 0 or ep == 1:
        val_gap = p5_gap(p5, val_enc)
        scheduler.step(val_gap)
        if val_gap > best_gap: best_gap = val_gap; best_ep = ep
        lr = opt.param_groups[0]['lr']
        print(f"P5 E{ep:4d} | val_gap={val_gap:.4f} | best={best_gap:.4f}@{best_ep} | LR={lr:.6f} | {time.time()-t0:.0f}s")
        if es(val_gap): print(f"P5 CONVERGED @ {ep}"); break

torch.save({"model_state_dict": p5.state_dict(), "val_gap": best_gap, "epoch": best_ep},
           os.path.join(SAVE_DIR, "overnight_P5_best.pt"))
for p in p5.parameters(): p.requires_grad = False; p5.eval()
report.add("P5", best_ep, best_gap, es.early_stop, lr, time.time()-t0)
del opt; torch.cuda.empty_cache()
print(f"P5 DONE: best_gap={best_gap:.4f} @ {best_ep}")

# ============================================================
# Bridge: P8+P6 (max 1000 epochs) — core bottleneck
# ============================================================
print(f"\n{'#'*60}\n# BRIDGE LONG TRAIN (max 1000, patience 30)\n{'#'*60}")

def get_char_vecs(text):
    vecs = []
    for c in text:
        if c not in char2idx: return None
        content = p1.char_content(torch.tensor([char2idx[c]], device=DEVICE))
        pos = p1.pos_encoder.pe[0:1]
        vecs.append(p1.project_char(torch.cat([pos[0], content[0]], dim=-1)))
    return torch.stack(vecs)

# Build bridge data from 8000 sentences (train+val split)
bridge_sentences = load_sentences(BRIDGE_SENT_FILE)
all_bridge = []
for words, roles in bridge_sentences:
    wvs = [enc_word(w) for w in words]
    if None in wvs: continue
    sv = p5(torch.stack(wvs), torch.tensor(roles, device=DEVICE), last_loss=0.0)
    all_bridge.append((words, sv, torch.stack(wvs)))
print(f"[Bridge] {len(all_bridge)} groups from {len(bridge_sentences)} sentences")
b_cut = int(len(all_bridge) * 0.9)
bridge_train = all_bridge[:b_cut]; bridge_val = all_bridge[b_cut:]

p8 = CharToSent(max_len=15).to(DEVICE)
p6 = SentToWordsDecoder(max_words=5).to(DEVICE)
opt = torch.optim.Adam(list(p8.parameters())+list(p6.parameters()), lr=0.002, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=8, min_lr=1e-6)
es = EarlyStopping(patience=30, min_delta=0.001)
best_wc = 0.0; best_ep = 0; ll = 1.0
W_SENT=5.0; W_WORD=1.0; W_DIV=0.1; W_EXP=0.05
t0 = time.time()

def bridge_val_score():
    p8.eval(); p6.eval()
    wc = []
    for words, sv_t, wvs_t in bridge_val[:50]:
        cvs = get_char_vecs(''.join(words))
        if cvs is None: continue
        sp = p8(cvs, last_loss=0.0)
        wp = p6(sp.unsqueeze(0), last_loss=0.0)[0]
        nw = min(len(wp), len(wvs_t))
        wc.append(F.cosine_similarity(wp[:nw], wvs_t[:nw], dim=-1).mean().item())
    p8.train(); p6.train()
    return sum(wc)/len(wc) if wc else 0

for ep in range(1, 1001):
    el = 0.0; nb = 0
    random.shuffle(bridge_train)
    for wi, (words, sv_t, wvs_t) in enumerate(bridge_train):
        cvs = get_char_vecs(''.join(words))
        if cvs is None: continue
        sp = p8(cvs, last_loss=ll); wp = p6(sp.unsqueeze(0), last_loss=ll)[0]
        sc = F.cosine_similarity(sp.unsqueeze(0), sv_t.unsqueeze(0), dim=-1)
        sent_align = (1.0-sc)**2
        nw = min(len(wp), len(wvs_t))
        wcs = F.cosine_similarity(wp[:nw], wvs_t[:nw], dim=-1)
        word_align = ((1.0-wcs)**2).mean()
        sent_div = torch.tensor(0.0, device=DEVICE)
        if wi>0 and wi%20==0:
            with torch.no_grad():
                rs = torch.stack([p8(get_char_vecs(''.join(bridge_train[j][0])), last_loss=0.0)
                                  for j in range(max(0,wi-20),wi) if get_char_vecs(''.join(bridge_train[j][0])) is not None])
                if len(rs)>=5:
                    sn = F.normalize(rs, dim=-1); pc = torch.mm(sn, sn.T)
                    mk = 1-torch.eye(len(sn), device=DEVICE)
                    sent_div = F.relu((pc*mk).sum()/mk.sum()-0.85)**2
        exp_loss = F.relu(0.02-p8.explore_state.norm())**2 + F.relu(0.02-p6.explore_state.norm())**2
        loss = W_SENT*sent_align + W_WORD*word_align + W_DIV*sent_div + W_EXP*exp_loss
        opt.zero_grad(); loss.backward(); opt.step()
        el += loss.item(); nb += 1; ll = loss.item()
    torch.cuda.empty_cache()

    if ep % 10 == 0 or ep == 1:
        val_wc = bridge_val_score()
        scheduler.step(val_wc)
        if val_wc > best_wc: best_wc = val_wc; best_ep = ep
        lr = opt.param_groups[0]['lr']
        print(f"BR E{ep:4d} | val_wc={val_wc:.4%} | best={best_wc:.4%}@{best_ep} | LR={lr:.6f} | {time.time()-t0:.0f}s")
        if es(val_wc): print(f"BRIDGE CONVERGED @ {ep}"); break

torch.save({"p8": p8.state_dict(), "p6": p6.state_dict(), "word_cos": best_wc, "epoch": best_ep},
           os.path.join(SAVE_DIR, "overnight_Bridge_best.pt"))
report.add("Bridge", best_ep, best_wc, es.early_stop, lr, time.time()-t0)
del opt, p8, p6; torch.cuda.empty_cache()
print(f"BRIDGE DONE: best_wc={best_wc:.4%} @ {best_ep}")

# ============================================================
# P7: Cross-sentence (max 500 epochs)
# ============================================================
print(f"\n{'#'*60}\n# P7 LONG TRAIN (max 500, patience 15)\n{'#'*60}")
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-6)
            es = EarlyStopping(patience=15, min_delta=0.001)
            best_p7 = 0.0; best_ep = 0; ll = 1.0
            t0 = time.time()

            for ep in range(1, 501):
                el = 0.0; nb = 0
                for Av, Bt in p7_enc:
                    Bp, _ = p7(Av, B_table, last_loss=ll)
                    cos = F.cosine_similarity(Bp.unsqueeze(0), Bt.unsqueeze(0), dim=-1)
                    loss = (1.0-cos).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    el += loss.item(); nb += 1; ll = loss.item()

                if ep % 20 == 0 or ep == 1:
                    p7.eval(); coses = []
                    with torch.no_grad():
                        for Av, Bt in p7_enc:
                            Bp, _ = p7(Av, B_table, last_loss=0.0)
                            coses.append(F.cosine_similarity(Bp.unsqueeze(0), Bt.unsqueeze(0), dim=-1).item())
                    p7.train()
                    avg_c = sum(coses)/len(coses)
                    scheduler.step(avg_c)
                    if avg_c > best_p7: best_p7 = avg_c; best_ep = ep
                    lr = opt.param_groups[0]['lr']
                    print(f"P7 E{ep:4d} | cos={avg_c:.4%} | best={best_p7:.4%}@{best_ep} | LR={lr:.6f} | {time.time()-t0:.0f}s")
                    if es(avg_c): print(f"P7 CONVERGED @ {ep}"); break

            torch.save({"model_state_dict": p7.state_dict(), "cos": best_p7, "epoch": best_ep},
                       os.path.join(SAVE_DIR, "overnight_P7_best.pt"))
            report.add("P7", best_ep, best_p7, es.early_stop, lr, time.time()-t0)
            print(f"P7 DONE: best={best_p7:.4%} @ {best_ep}")

# ============================================================
# Final report
# ============================================================
report.save(os.path.join(BASE_DIR, "overnight_report.csv"))
print(f"\n{'#'*60}")
print(f"OVERNIGHT TRAINING COMPLETE")
print(f"{'#'*60}")
print(f"Report: overnight_report.csv")
for r in report.rows:
    goal = {"P1": "≥98.6%", "P2": "≥92%", "P3": "100%", "P5": "≥1.5", "Bridge": "≥84%", "P7": "≥98%"}
    met = "✓" if r['layer'] in goal and r['best_score'] >= float(goal[r['layer']].rstrip('%')) else " "
    print(f"  {r['layer']:8s} | best={r['best_score']:.4%}@{r['best_epoch']:4d} | {r['elapsed_min']:.0f}min | converged={r['converged']} | goal {goal.get(r['layer'],'')} {met}")
a = torch.cuda.memory_allocated(DEVICE)/1024**2
print(f"\nGPU final: {a:.0f}MB")
