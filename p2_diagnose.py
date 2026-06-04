"""
P2 诊断: 词→字反解层深度分析
================================
100轮, 每5轮输出完整诊断(分布/梯度/探索区/错误分析)
"""
import torch, torch.nn.functional as F, time, os, sys, re, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P2_word_char.model import WordToCharDecoder

WORD_LIST = os.path.join(DATA_DIR, "word_list_v2.txt")
SEED = 789
LOG_EVERY = 5
TOTAL_EPOCHS = 100

random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
print(f"P2 DIAGNOSE | Seed={SEED} | {TOTAL_EPOCHS}epochs | log every {LOG_EVERY}")

# === Data ===
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
all_pairs = [[char2idx[c1], char2idx[c2]] for _, c1, c2 in entries]
N = len(entries)
print(f"Words: {N} | Chars: {len(char_list)}")

# === Load P1 ===
p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
ckpt = torch.load(p1_path, map_location=DEVICE)
p1 = CharToWordModel(ckpt["num_chars"], ckpt["num_words"]).to(DEVICE)
p1.load_state_dict(ckpt["model_state_dict"])
for p in p1.parameters(): p.requires_grad = False; p1.eval()
print(f"P1 loaded: Top-1={ckpt.get('top1','?')}")

# === P2 ===
p2 = WordToCharDecoder().to(DEVICE)
print(f"P2 params: {sum(p.numel() for p in p2.parameters()):,}")
print(f"  explore_state: {list(p2.explore_state.shape)}")
print(f"  meta_fc: {[list(p.shape) for p in p2.meta_fc.parameters()]}")

opt = torch.optim.Adam(p2.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
last_loss = 1.0

# === Diagnostic helpers ===
@torch.no_grad()
def diagnose(epoch, loss_avg):
    d = {}
    p2.eval()

    # Sample 100 random word pairs
    idxs = torch.randperm(N)[:100]
    pids = torch.tensor([all_pairs[i] for i in idxs], device=DEVICE)
    with torch.no_grad():
        _, _, full = p1.get_char_vectors(pids)
        tgt_c1 = p1.project_char(full[:,0,:])  # [100, 128]
        tgt_c2 = p1.project_char(full[:,1,:])
        wv = p1(pids, last_loss=last_loss)

    pred_c1, pred_c2 = p2(wv, last_loss=last_loss)

    # A. Basic metrics
    sim1 = F.cosine_similarity(pred_c1, tgt_c1, dim=-1)
    sim2 = F.cosine_similarity(pred_c2, tgt_c2, dim=-1)
    d['cos_c1'] = sim1.mean().item()
    d['cos_c2'] = sim2.mean().item()
    d['top5_c1'] = (sim1 > 0.9).float().mean().item()  # rough top-5 proxy
    d['top5_c2'] = (sim2 > 0.9).float().mean().item()

    # B. Input/Output distribution
    d['wv_mean'] = wv.mean().item(); d['wv_std'] = wv.std().item()
    d['wv_norm'] = wv.norm(dim=-1).mean().item()
    d['pc1_norm'] = pred_c1.norm(dim=-1).mean().item()
    d['pc2_norm'] = pred_c2.norm(dim=-1).mean().item()
    d['has_nan'] = torch.isnan(pred_c1).any().item() or torch.isnan(pred_c2).any().item()
    d['has_inf'] = torch.isinf(pred_c1).any().item()

    # C. Explore/Meta state
    es = p2.explore_state
    d['explore_mean'] = es.mean().item(); d['explore_std'] = es.std().item()
    d['explore_norm'] = es.norm().item()
    mw = p2.meta_fc[0].weight  # Linear weight
    d['meta_w_norm'] = mw.norm().item()
    loss_f = min(last_loss * 20.0, 1.0)
    mod = p2.meta_fc(es * loss_f)
    d['mod_mean'] = mod.mean().item(); d['mod_std'] = mod.std().item()
    d['mod_range'] = (mod.max() - mod.min()).item()
    d['loss_factor'] = loss_f

    # D. Char confusion analysis
    # Pairwise: does c1 prediction have higher cosine with tgt_c1 than tgt_c2?
    c1_tgt1 = F.cosine_similarity(pred_c1, tgt_c1, dim=-1).mean()
    c1_tgt2 = F.cosine_similarity(pred_c1, tgt_c2, dim=-1).mean()
    c2_tgt2 = F.cosine_similarity(pred_c2, tgt_c2, dim=-1).mean()
    c2_tgt1 = F.cosine_similarity(pred_c2, tgt_c1, dim=-1).mean()
    d['c1_to_tgt1'] = c1_tgt1.item()
    d['c1_to_tgt2'] = c1_tgt2.item()  # cross-talk
    d['c2_to_tgt2'] = c2_tgt2.item()
    d['c2_to_tgt1'] = c2_tgt1.item()  # cross-talk

    # E. Prediction collapse check
    c1_pairwise = F.cosine_similarity(pred_c1.unsqueeze(1), pred_c1.unsqueeze(0), dim=-1)
    c1_mask = 1 - torch.eye(100, device=DEVICE)
    d['c1_self_sim'] = (c1_pairwise * c1_mask).sum().item() / c1_mask.sum().item()
    c2_pairwise = F.cosine_similarity(pred_c2.unsqueeze(1), pred_c2.unsqueeze(0), dim=-1)
    d['c2_self_sim'] = (c2_pairwise * c1_mask).sum().item() / c1_mask.sum().item()

    # F. Gradient norms
    total_grad = 0.0
    for name, p in p2.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            total_grad += gn
            if 'explore' in name: d['grad_explore'] = gn
            if 'meta' in name: d['grad_meta'] = gn
            if 'c1_head' in name: d['grad_c1head'] = gn
            if 'c2_head' in name: d['grad_c2head'] = gn
    d['grad_total'] = total_grad

    p2.train()

    # Print summary
    print(f"\n{'='*70}")
    print(f"DIAG Epoch {epoch:4d} | Loss={loss_avg:.6f}")
    print(f"{'='*70}")
    print(f"[A] cos c1={d['cos_c1']:.4f} c2={d['cos_c2']:.4f} | top5 c1={d['top5_c1']:.3f} c2={d['top5_c2']:.3f}")
    print(f"[B] wv: mean={d['wv_mean']:.4f} std={d['wv_std']:.4f} norm={d['wv_norm']:.2f} | pred norm c1={d['pc1_norm']:.2f} c2={d['pc2_norm']:.2f} | NaN={d['has_nan']} Inf={d['has_inf']}")
    print(f"[C] explore: mean={d['explore_mean']:.6f} std={d['explore_std']:.6f} norm={d['explore_norm']:.4f} | meta_w={d['meta_w_norm']:.4f}")
    print(f"    mod: mean={d['mod_mean']:.6f} std={d['mod_std']:.6f} range={d['mod_range']:.6f} | loss_f={d['loss_factor']:.3f}")
    print(f"[D] crosstalk: c1->tgt1={d['c1_to_tgt1']:.4f} c1->tgt2={d['c1_to_tgt2']:.4f} | c2->tgt2={d['c2_to_tgt2']:.4f} c2->tgt1={d['c2_to_tgt1']:.4f}")
    print(f"[E] collapse: c1_self={d['c1_self_sim']:.4f} c2_self={d['c2_self_sim']:.4f} (near 1 = collapse)")
    print(f"[F] grad: total={d['grad_total']:.4f} explore={d.get('grad_explore',0):.6f} meta={d.get('grad_meta',0):.6f} c1={d.get('grad_c1head',0):.4f} c2={d.get('grad_c2head',0):.4f}")

    return d

# === Training ===
t0 = time.time()
diag_history = []

for epoch in range(1, TOTAL_EPOCHS + 1):
    el = 0.0; nb = 0
    perm = torch.randperm(N)
    for bs in range(0, N, 200):
        be = min(bs + 200, N)
        idxs = perm[bs:be]
        pids = torch.tensor([all_pairs[i] for i in idxs], device=DEVICE)
        with torch.no_grad():
            _, _, full = p1.get_char_vectors(pids)
            tgt_c1 = p1.project_char(full[:,0,:])
            tgt_c2 = p1.project_char(full[:,1,:])
            wv = p1(pids, last_loss=last_loss)
        pred_c1, pred_c2 = p2(wv, last_loss=last_loss)
        sim1 = F.cosine_similarity(pred_c1, tgt_c1, dim=-1).mean()
        sim2 = F.cosine_similarity(pred_c2, tgt_c2, dim=-1).mean()
        loss = (1.0 - (sim1 + sim2) / 2.0)
        opt.zero_grad(); loss.backward(); opt.step()
        el += loss.item(); nb += 1
        last_loss = loss.item()
    torch.cuda.empty_cache()

    if epoch % LOG_EVERY == 0 or epoch == 1:
        d = diagnose(epoch, el/nb)
        d['epoch'] = epoch; d['loss'] = el/nb
        diag_history.append(d)

elapsed = time.time() - t0

# === Final report ===
print(f"\n{'#'*60}")
print(f"P2 DIAGNOSE COMPLETE ({elapsed:.0f}s)")
print(f"{'#'*60}")
print(f"\nKey findings from {len(diag_history)} diagnostic snapshots:")

# Trend analysis
cos1_trend = [d['cos_c1'] for d in diag_history]
cos2_trend = [d['cos_c2'] for d in diag_history]
mod_trend = [d['mod_range'] for d in diag_history]
grad_trend = [d.get('grad_explore', 0) for d in diag_history]
collapse_trend = [d['c1_self_sim'] for d in diag_history]

print(f"\nCosine trend: c1 {cos1_trend[0]:.4f}→{cos1_trend[-1]:.4f} | c2 {cos2_trend[0]:.4f}→{cos2_trend[-1]:.4f}")
print(f"Modulation range: {mod_trend[0]:.6f}→{mod_trend[-1]:.6f}")
print(f"Explore grad: {grad_trend[0]:.6f}→{grad_trend[-1]:.6f}")
print(f"Collapse risk: {collapse_trend[0]:.4f}→{collapse_trend[-1]:.4f}")

if cos1_trend[-1] > cos1_trend[0] * 1.05:
    print("\n[OK] P2 is learning (cosine improving >5%)")
else:
    print("\n[WARN] P2 learning stalled (<5% improvement)")

if mod_trend[-1] < 0.001:
    print("[WARN] Modulation is near-zero — explore/meta not active")
else:
    print("[OK] Modulation is active")

if grad_trend[-1] < 1e-7:
    print("[WARN] Explore gradient vanished — last_loss not flowing")
else:
    print("[OK] Explore gradient flowing")

if collapse_trend[-1] > 0.5:
    print("[WARN] P2 output collapsing (self-similarity >0.5)")
else:
    print("[OK] P2 output diverse (no collapse)")

torch.save({"diag_history": diag_history, "p2_state": p2.state_dict()},
           os.path.join(SAVE_DIR, "P2_diag.pt"))
print(f"\nSaved: {SAVE_DIR}/P2_diag.pt")
