"""
V18 全链路一键训练 (v5.1 stable)
==================================
按序训练 P1→P2→P3→P5→P8+P6(bridge)→P7
用法: python train_all.py [--seed 789]
"""
import torch, torch.nn.functional as F, time, os, sys, re, random, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='Random seed')
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[Seed] {args.seed}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P2_word_char.model import WordToCharDecoder, cosine_loss as p2_loss
from P3_word_attr.model import SubjectBindingModel, margin_loss
from P5_sentence.model import SentenceSynthesis, contrastive_loss
from P6_sent_word.model import SentToWordsDecoder
from P8_char_sent.model import CharToSent
from P7_cross_sent.model import CrossSentenceRouter

# ============================================================
# 配置 (可修改)
# ============================================================
WORD_LIST = os.path.join(DATA_DIR, "word_list_v2.txt")
SENT_FILE = os.path.join(BASE_DIR, "P5_sentence", "sentences_v2.txt")

# P1 非对称: 内部2048D+512头, 需要控制batch
P1_MINI_BATCH = 64
P1_ACCUM = 25        # 64×25≈1600 per update
P1_EPOCHS = 300

P2_EPOCHS = 200
P3_EPOCHS = 200
P5_EPOCHS = 300
BRIDGE_EPOCHS = 500
P7_EPOCHS = 300

PATIENCE = 30

print(f"{'='*60}")
print(f"V18 全链路训练")
print(f"设备: {DEVICE} | 词表: {WORD_LIST}")
print(f"句子: {SENT_FILE}")
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
            if m:
                entries.append((m.group(1), m.group(2), m.group(3)))
            elif len(line) == 2:
                entries.append((line, line[0], line[1]))
    return entries

entries = load_word_list(WORD_LIST)
chars_set = set()
for _, c1, c2 in entries:
    chars_set.add(c1); chars_set.add(c2)
char_list = sorted(chars_set)
char2idx = {c: i for i, c in enumerate(char_list)}
idx2char = {i: c for c, i in char2idx.items()}
word2idx = {w: i for i, (w, _, _) in enumerate(entries)}
idx2word = {i: w for w, i in word2idx.items()}
all_pairs = [[char2idx[c1], char2idx[c2]] for _, c1, c2 in entries]
NUM_CHARS = len(char_list)
NUM_WORDS = len(entries)

print(f"\n[数据] {NUM_WORDS}词 | {NUM_CHARS}字")

def load_sentences(path):
    role_map = {"subj": 0, "verb": 1, "obj": 2}
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
                    words.append(w)
                    roles.append(role_map.get(r, -1))
                except: pass
            if words: data.append((words, roles))
    return data

sentences = load_sentences(SENT_FILE)
print(f"[句子] {len(sentences)}句")

# ============================================================
# P1: 字符 → 词语 (非对称: 2048D内部+512头 → 128D输出)
# ============================================================
print(f"\n{'#'*60}")
print(f"# P1: 字符→词语 ({P1_EPOCHS}epochs, batch={P1_MINI_BATCH}×{P1_ACCUM})")
print(f"{'#'*60}")

p1 = CharToWordModel(NUM_CHARS, NUM_WORDS).to(DEVICE)
n_p1 = sum(p.numel() for p in p1.parameters())
print(f"[P1] {n_p1:,} params ({n_p1*4/1024/1024:.0f}MB), 内部2048D+{P1_HEADS}头→输出128D")

opt = torch.optim.Adam(p1.parameters(), lr=0.005, weight_decay=WEIGHT_DECAY)
last_loss = 1.0; best_top1 = 0.0; no_improve = 0

def pearson_loss(pred, target):
    pm = pred.mean(dim=0, keepdim=True); tm = target.mean(dim=0, keepdim=True)
    pc = pred - pm; tc = target - tm
    num = (pc * tc).sum()
    den = torch.sqrt((pc**2).sum() * (tc**2).sum() + PEARSON_EPSILON)
    return 1.0 - num / den

@torch.no_grad()
def eval_p1():
    p1.eval()
    ref = p1.get_all_reference_vectors(DEVICE)
    ref_n = F.normalize(ref, dim=-1)
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
    opt.step()
    opt.zero_grad()
    torch.cuda.empty_cache()

    if epoch % 5 == 0 or epoch == 1:
        top1 = eval_p1()
        if top1 > best_top1:
            best_top1 = top1; no_improve = 0
            torch.save({"epoch": epoch, "model_state_dict": p1.state_dict(),
                       "num_chars": NUM_CHARS, "num_words": NUM_WORDS,
                       "char2idx": char2idx, "idx2char": idx2char,
                       "word2idx": word2idx, "idx2word": idx2word, "top1": top1},
                       os.path.join(SAVE_DIR, "P1_best.pt"))
        else: no_improve += 5
        elapsed = time.time() - t0
        a = torch.cuda.memory_allocated(DEVICE)/1024**2
        print(f"  E{epoch:4d} | Loss={epoch_loss/max(nb,1):.6f} | Top-1={top1:.4%} | best={best_top1:.4%} | GPU={a:.0f}MB | {elapsed:.0f}s")
        if no_improve >= PATIENCE:
            print(f"  Early stop @ {epoch}")
            break

print(f"[P1] DONE: best={best_top1:.4%} | {time.time()-t0:.0f}s")

# 加载最佳P1, 完全冻结 (投影层在Bridge阶段才解冻)
ckpt = torch.load(os.path.join(SAVE_DIR, "P1_best.pt"), map_location=DEVICE)
p1.load_state_dict(ckpt["model_state_dict"])
for p in p1.parameters(): p.requires_grad = False; p1.eval()
del opt; torch.cuda.empty_cache()

# ============================================================
# P2: 词语 → 字符
# ============================================================
print(f"\n{'#'*60}")
print(f"# P2: 词语→字符 ({P2_EPOCHS}epochs)")
print(f"{'#'*60}")

p2 = WordToCharDecoder().to(DEVICE)
print(f"[P2] 参数: {sum(p.numel() for p in p2.parameters()):,}")

opt = torch.optim.Adam(p2.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
last_loss = 1.0; best_p2_cos = 0.0

t0 = time.time()
for epoch in range(1, P2_EPOCHS + 1):
    epoch_loss = 0.0; epoch_sim1 = 0.0; epoch_sim2 = 0.0; nb = 0
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
        pred_c1, pred_c2 = p2(word_vec, last_loss=last_loss)
        loss, sim1, sim2 = p2_loss(pred_c1, pred_c2, real_c1, real_c2)
        # 探索区防衰减: 惩罚explore_state norm过小
        explore_norm = p2.explore_state.norm()
        loss = loss + 0.001 * F.relu(0.1 - explore_norm)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); epoch_sim1 += sim1; epoch_sim2 += sim2; nb += 1
        last_loss = loss.item()

    if epoch % 20 == 0 or epoch == 1:
        avg_sim = (epoch_sim1 + epoch_sim2) / (2 * nb)
        if avg_sim > best_p2_cos:
            best_p2_cos = avg_sim
            torch.save({"epoch": epoch, "model_state_dict": p2.state_dict(), "avg_cos": avg_sim},
                       os.path.join(SAVE_DIR, "P2_best.pt"))
        print(f"  Epoch {epoch:4d} | Loss={epoch_loss/nb:.6f} | cos={avg_sim:.4%} | best={best_p2_cos:.4%}")

print(f"[P2] DONE: best cos={best_p2_cos:.4%} | {time.time()-t0:.0f}s")

# ============================================================
# P3: 七属性绑定
# ============================================================
print(f"\n{'#'*60}")
print(f"# P3: 七属性绑定 ({P3_EPOCHS}epochs/attr)")
print(f"{'#'*60}")

ATTRS = ["主语", "谓语", "宾语", "定语", "状语", "补语", "虚词"]
# 加载属性词表 (按文件名关键词匹配, 兼容编码)
attr_data = {}
data_dir = os.path.join(BASE_DIR, "P3_data")
# 文件名 → 属性映射 (通过文件大小和内容推断)
attr_size_map = {
    "主语": None, "谓语": None, "宾语": None,
    "定语": None, "状语": None, "补语": None, "虚词": None
}
# 从P3 checkpoints恢复词表 (如果存在)
ckpt_dir = SAVE_DIR
for f in os.listdir(data_dir):
    path = os.path.join(data_dir, f)
    if not f.endswith('.txt') or os.path.getsize(path) < 10: continue
    # 读前几行推断属性
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        sample = fh.read(200)
    # 按文件名匹配
    f_lower = f.lower()
    if 'subj' in f_lower or 'adj' not in f_lower and 'adv' not in f_lower and 'comp' not in f_lower and 'func' not in f_lower and 'verb' not in f_lower and 'obj' not in f_lower and 'pred' not in f_lower and 'label' not in f_lower:
        pass  # can't identify

# 直接用P3 checkpoints中的词表
for f in sorted(os.listdir(ckpt_dir)):
    if not f.startswith('P3_') or not f.endswith('_best.pt'): continue
    path = os.path.join(ckpt_dir, f)
    try:
        ckpt_data = torch.load(path, map_location='cpu', weights_only=False)
        attr_name = ckpt_data.get('attr', '')
        fw = ckpt_data.get('family_words', [])
        # 匹配属性名
        for a in ATTRS:
            if a in attr_name or (a in f):
                if a not in attr_data and len(fw) > 5:
                    attr_data[a] = fw  # 直接用词表
                    break
    except: pass

# 补充: 用文件直接加载 (ASCII-safe names)
ascii_map = {
    "主语": "subj", "谓语": "verb", "宾语": "obj",
    "定语": "adj", "状语": "adv", "补语": "comp", "虚词": "func"
}
for attr, ascii_name in ascii_map.items():
    if attr not in attr_data:
        candidate = os.path.join(data_dir, f"words_{ascii_name}.txt")
        if os.path.exists(candidate) and os.path.getsize(candidate) > 10:
            with open(candidate, 'r', encoding='utf-8', errors='replace') as fh:
                words = [w.strip() for w in fh if w.strip()]
            if len(words) > 5:
                attr_data[attr] = words

print(f"[P3] 找到属性词表: {list(attr_data.keys())}")

def _get_chars(w):
    if len(w) == 1: return w[0], w[0]
    return w[0], w[1] if len(w) > 1 else w[0]

p3_results = {}
for attr in ATTRS:
    raw = attr_data.get(attr)
    if not raw:
        print(f"  [P3-{attr}] 无词表, 跳过")
        continue

    # raw可能是文件路径或词列表
    if isinstance(raw, str) and os.path.exists(raw):
        with open(raw, 'r', encoding='utf-8', errors='replace') as f:
            raw = [w.strip() for w in f if w.strip()]

    attr_words = [w for w in raw if _get_chars(w)[0] in char2idx and _get_chars(w)[-1] in char2idx]
    attr_set = set(attr_words)
    non_attr = [w for w in idx2word.values() if w not in attr_set and len(w) >= 1
                and w[0] in char2idx]

    @torch.no_grad()
    def encode(words):
        vecs = []
        for w in words:
            c1, c2 = _get_chars(w)
            cids = torch.tensor([[char2idx[c1], char2idx[c2]]], device=DEVICE)
            vecs.append(p1(cids, last_loss=0.0)[0])
        return torch.stack(vecs) if vecs else torch.zeros(0, WORD_DIM, device=DEVICE)

    family_p1 = encode(attr_words)
    family_proto = family_p1.mean(dim=0)
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
            for bs in range(0, len(ids), BATCH_SIZE):
                be = min(bs+BATCH_SIZE, len(ids))
                out, attn, q_raw = model(ids[bs:be], family_p1, last_loss=last_loss)
                loss, _ = margin_loss(out, family_proto, is_pos[bs:be], q_raw=q_raw)
                opt.zero_grad(); loss.backward(); opt.step()
                el += loss.item(); nb += 1; last_loss = loss.item()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                ps = model.binding_score(pos_ids, family_p1) if len(pos_ids)>0 else torch.tensor([0.])
                ns = model.binding_score(neg_ids, family_p1) if len(neg_ids)>0 else torch.tensor([0.])
            model.train()
            pm, nm = ps.mean().item(), ns.mean().item()
            t = (pm+nm)/2
            acc = ((ps>t).float().mean()+(ns<=t).float().mean())/2 if len(ps)*len(ns)>0 else 0
            if acc > best_acc:
                best_acc = acc
                torch.save({"model": model.state_dict(), "attr": attr, "acc": acc,
                           "p3_word2id": p3_w2id, "family_words": attr_words},
                           os.path.join(SAVE_DIR, f"P3_{attr}_best.pt"))
            print(f"  P3-{attr} E{epoch:4d} | Loss={el/max(nb,1):.6f} | pos={pm:.3f} neg={nm:.3f} | Acc={acc:.2%}")

    p3_results[attr] = (pm, nm, best_acc)
    print(f"  [P3-{attr}] DONE: pos={pm:.3f} neg={nm:.3f} Acc={best_acc:.2%}")

# ============================================================
# P5: 词序列 → 句子
# ============================================================
print(f"\n{'#'*60}")
print(f"# P5: 词序列→句子 ({P5_EPOCHS}epochs)")
print(f"{'#'*60}")

@torch.no_grad()
def enc_word(w):
    if len(w) == 1: c1 = c2 = w[0]
    else: c1, c2 = w[0], w[-1]
    if c1 not in char2idx or c2 not in char2idx: return None
    return p1(torch.tensor([[char2idx[c1], char2idx[c2]]], device=DEVICE), last_loss=0.0)[0]

encoded_sents = []
for words, roles in sentences:
    vecs = [enc_word(w) for w in words]
    if None not in vecs:
        encoded_sents.append((torch.stack(vecs), torch.tensor(roles, device=DEVICE)))
print(f"[P5] 有效句子: {len(encoded_sents)}")

def make_scrambled(word_vecs, roles):
    n = len(roles)
    idx = list(range(n)); random.shuffle(idx)
    return (torch.stack([word_vecs[i] for i in idx]),
            torch.tensor([roles[i] for i in idx], device=word_vecs.device))

p5 = SentenceSynthesis().to(DEVICE)
print(f"[P5] 参数: {sum(p.numel() for p in p5.parameters()):,}")

opt = torch.optim.Adam(p5.parameters(), lr=0.002, weight_decay=1e-5)
last_loss = 1.0; best_gap = -999; no_improve = 0

t0 = time.time()
sent_buf = []  # 句子向量缓冲区, 用于多样性正则
for epoch in range(1, P5_EPOCHS + 1):
    epoch_loss = 0.0; nb = 0
    for wv, roles in encoded_sents:
        correct = p5(wv, roles, last_loss=last_loss)
        sv, sr = make_scrambled(wv, roles)
        scr = p5(sv, sr, last_loss=last_loss)
        loss, cos_c, cos_s = contrastive_loss(correct, scr.unsqueeze(0))

        # 多样性正则: 每20个样本检查一次句子向量是否塌缩
        sent_buf.append(correct.detach())
        if len(sent_buf) >= 20:
            buf = torch.stack(sent_buf)
            # 成对余弦的平均值 — 如果所有句子相同, 这个值接近1.0
            buf_n = F.normalize(buf, dim=-1)
            pair_cos = torch.mm(buf_n, buf_n.T)
            mask = 1 - torch.eye(len(buf), device=DEVICE)
            avg_cos = (pair_cos * mask).sum() / (mask.sum() + 1e-8)
            # 惩罚高相似度 (塌缩检测)
            div_penalty = F.relu(avg_cos - 0.3) * 0.1
            loss = loss + div_penalty
            sent_buf = []

        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); nb += 1; last_loss = loss.item()

    if epoch % 10 == 0 or epoch == 1:
        gaps = []
        for wv, roles in encoded_sents[:20]:
            c = p5(wv, roles, last_loss=0.0)
            sv, sr = make_scrambled(wv, roles)
            s = p5(sv, sr, last_loss=0.0)
            ref = c.detach()
            cc = F.cosine_similarity(c, ref, dim=-1).item()
            cs = F.cosine_similarity(s, ref, dim=-1).item()
            gaps.append(cc - cs)
        avg_g = sum(gaps)/len(gaps)
        if avg_g > best_gap:
            best_gap = avg_g; no_improve = 0
            torch.save({"epoch": epoch, "model_state_dict": p5.state_dict(), "avg_gap": avg_g,
                       "w_subj": p5.w_subj.item(), "w_verb": p5.w_verb.item(), "w_obj": p5.w_obj.item()},
                       os.path.join(SAVE_DIR, "P5_best.pt"))
        else: no_improve += 10
        print(f"  Epoch {epoch:4d} | Loss={epoch_loss/nb:.6f} | gap={avg_g:.4f} | best={best_gap:.4f}")
        if no_improve >= PATIENCE:
            print(f"  Early stop @ epoch {epoch}")
            break

print(f"[P5] DONE: best gap={best_gap:.4f} | {time.time()-t0:.0f}s")

# 冻结P5
p5.load_state_dict(torch.load(os.path.join(SAVE_DIR, "P5_best.pt"), map_location=DEVICE)["model_state_dict"])
for p in p5.parameters(): p.requires_grad = False; p5.eval()

# ============================================================
# P8 + P6 桥接训练
# ============================================================
print(f"\n{'#'*60}")
print(f"# P8+P6 桥接训练 ({BRIDGE_EPOCHS}epochs)")
print(f"{'#'*60}")

@torch.no_grad()
def get_char_vecs(text):
    """获取128D字符向量(从P1内部2048D分阶段投影)"""
    vecs = []
    for c in text:
        if c not in char2idx: return None
        content = p1.char_content(torch.tensor([char2idx[c]], device=DEVICE))
        pos = p1.pos_encoder.pe[0:1]
        internal = torch.cat([pos[0], content[0]], dim=-1)  # 2048D
        vecs.append(p1.project_char(internal))                # →128D
    return torch.stack(vecs)

bridge_data = []
for words, roles in sentences:
    wvs = [enc_word(w) for w in words]
    if None in wvs: continue
    wvs_stack = torch.stack(wvs)
    roles_t = torch.tensor(roles, device=DEVICE)
    sv = p5(wvs_stack, roles_t, last_loss=0.0)
    text = ''.join(words)
    cvs = get_char_vecs(text)
    if cvs is None: continue
    bridge_data.append((cvs, sv, wvs_stack))

print(f"[桥接数据] {len(bridge_data)}组")

# 解冻P1投影层: Bridge阶段联合微调
for p in p1.output_proj.parameters():
    p.requires_grad = True

p8 = CharToSent(max_len=15).to(DEVICE)
p6 = SentToWordsDecoder(max_words=3).to(DEVICE)
n_b = sum(p.numel() for m in [p8,p6] for p in m.parameters())
n_proj = sum(p.numel() for p in p1.output_proj.parameters())
print(f"[P8+P6+Proj] {n_b:,} + {n_proj:,} = {n_b+n_proj:,} params")

# P8+P6正常LR, 投影层1/10 LR
opt = torch.optim.Adam([
    {'params': list(p8.parameters()) + list(p6.parameters()), 'lr': 0.002},
    {'params': p1.output_proj.parameters(), 'lr': 0.0002},
], weight_decay=1e-5)
last_loss = 1.0; best_word_cos = 0.0; no_improve = 0; bridge_patience = 50

t0 = time.time()
for epoch in range(1, BRIDGE_EPOCHS + 1):
    el = 0.0; nb = 0
    for cvs, sv_t, wvs_t in bridge_data:
        sent_pred = p8(cvs, last_loss=last_loss)
        word_preds = p6(sent_pred.unsqueeze(0), last_loss=last_loss)[0]
        sent_loss = (1.0 - F.cosine_similarity(sent_pred.unsqueeze(0), sv_t.unsqueeze(0), dim=-1)).mean()
        nw = min(len(word_preds), len(wvs_t))
        word_sims = F.cosine_similarity(word_preds[:nw], wvs_t[:nw], dim=-1)
        word_loss = (1.0 - word_sims).mean()
        loss = sent_loss + word_loss
        opt.zero_grad(); loss.backward(); opt.step()
        el += loss.item(); nb += 1; last_loss = loss.item()

    if epoch % 30 == 0 or epoch == 1:
        p8.eval(); p6.eval()
        sent_coses = []; word_coses = []
        with torch.no_grad():
            for cvs, sv_t, wvs_t in bridge_data[:20]:
                sp = p8(cvs, last_loss=0.0)
                sent_coses.append(F.cosine_similarity(sp.unsqueeze(0), sv_t.unsqueeze(0), dim=-1).item())
                wp = p6(sp.unsqueeze(0), last_loss=0.0)[0]
                nw = min(len(wp), len(wvs_t))
                word_coses.append(F.cosine_similarity(wp[:nw], wvs_t[:nw], dim=-1).mean().item())
        p8.train(); p6.train()
        avg_sent = sum(sent_coses)/len(sent_coses)
        avg_word = sum(word_coses)/len(word_coses)
        if avg_word > best_word_cos:
            best_word_cos = avg_word; no_improve = 0
            torch.save({"epoch": epoch, "p8": p8.state_dict(), "p6": p6.state_dict(),
                       "sent_cos": avg_sent, "word_cos": avg_word},
                       os.path.join(SAVE_DIR, "P6_bridge.pt"))
        else: no_improve += 30
        print(f"  Epoch {epoch:4d} | Loss={el/max(nb,1):.6f} | sent={avg_sent:.4f} word={avg_word:.4f} | best={best_word_cos:.4f}")
        if no_improve >= bridge_patience:
            print(f"  Bridge early stop @ epoch {epoch}")
            break

print(f"[P8+P6] DONE: best word_cos={best_word_cos:.4f} | {time.time()-t0:.0f}s")

# ============================================================
# P7: 跨句路由
# ============================================================
print(f"\n{'#'*60}")
print(f"# P7: 跨句路由 ({P7_EPOCHS}epochs)")
print(f"{'#'*60}")

p7_data_path = os.path.join(BASE_DIR, "P7_cross_sent", "data_p7.txt")
p7_pairs = []
if os.path.exists(p7_data_path):
    with open(p7_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("|")
            if len(parts) == 2:
                p7_pairs.append((parts[0].split(), parts[1].split()))
print(f"[P7] 句对: {len(p7_pairs)}")

if len(p7_pairs) > 0:
    B_vocab = sorted(set(w for _, B in p7_pairs for w in B))
    B_vecs = [enc_word(w) for w in B_vocab if enc_word(w) is not None]
    if len(B_vecs) >= 3:
        B_table = torch.stack(B_vecs)
        p7_encoded = []
        for A_words, B_words in p7_pairs:
            A_vecs = [enc_word(w) for w in A_words]
            B_wvs = [enc_word(w) for w in B_words]
            if None in A_vecs or None in B_wvs: continue
            A_wvs = torch.stack(A_vecs)
            B_sent = p5(torch.stack(B_wvs), torch.arange(len(B_wvs), device=DEVICE)%3, last_loss=0.0)
            p7_encoded.append((A_wvs, B_sent))

        if len(p7_encoded) >= 3:
            p7 = CrossSentenceRouter().to(DEVICE)
            print(f"[P7] 参数: {sum(p.numel() for p in p7.parameters()):,} | 有效对: {len(p7_encoded)}")
            opt = torch.optim.Adam(p7.parameters(), lr=0.003, weight_decay=1e-5)
            last_loss = 1.0; best_p7_cos = 0.0

            for epoch in range(1, P7_EPOCHS + 1):
                el = 0.0; nb = 0
                for A_wvs, B_sent_t in p7_encoded:
                    B_pred, _ = p7(A_wvs, B_table, last_loss=last_loss)
                    cos = F.cosine_similarity(B_pred.unsqueeze(0), B_sent_t.unsqueeze(0), dim=-1)
                    loss = (1.0 - cos).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    el += loss.item(); nb += 1; last_loss = loss.item()

                if epoch % 100 == 0 or epoch == 1:
                    p7.eval()
                    coses = []
                    with torch.no_grad():
                        for A_wvs, B_sent_t in p7_encoded:
                            B_pred, _ = p7(A_wvs, B_table, last_loss=0.0)
                            coses.append(F.cosine_similarity(B_pred.unsqueeze(0), B_sent_t.unsqueeze(0), dim=-1).item())
                    p7.train()
                    avg_c = sum(coses)/len(coses)
                    if avg_c > best_p7_cos:
                        best_p7_cos = avg_c
                        torch.save({"epoch": epoch, "model_state_dict": p7.state_dict(), "avg_cos": avg_c},
                                   os.path.join(SAVE_DIR, "P7_best.pt"))
                    print(f"  Epoch {epoch:4d} | Loss={el/max(nb,1):.6f} | cos={avg_c:.4f} | best={best_p7_cos:.4f}")

            print(f"[P7] DONE: best cos={best_p7_cos:.4f}")
        else:
            print(f"[P7] 有效句对不足, 跳过训练")
    else:
        print(f"[P7] B词表不足, 跳过训练")
else:
    print(f"[P7] 无训练数据, 跳过")

# ============================================================
# 总结
# ============================================================
print(f"\n{'#'*60}")
print(f"# 全链路训练完成")
print(f"{'#'*60}")
print(f"P1: Top-1={best_top1:.4%}")
print(f"P2: cos={best_p2_cos:.4%}")
for attr, (pm, nm, acc) in p3_results.items():
    print(f"P3-{attr}: Acc={acc:.2%}")
print(f"P5: gap={best_gap:.4f}")
print(f"P8+P6: word_cos={best_word_cos:.4f}")
if 'best_p7_cos' in dir():
    print(f"P7: cos={best_p7_cos:.4f}")
print(f"\n检查点目录: {SAVE_DIR}")
