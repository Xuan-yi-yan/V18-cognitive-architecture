"""Bridge长训: 多目标loss, 100轮, 每2轮完整诊断"""
import torch, torch.nn.functional as F, time, os, sys, re, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P5_sentence.model import SentenceSynthesis
from P6_sent_word.model import SentToWordsDecoder
from P8_char_sent.model import CharToSent

SEED = 789; TOTAL = 50; LOG_EVERY = 1  # 50轮每轮展示; 若仍涨则扩到100轮每2轮
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
SD = os.path.join(BASE_DIR, "P1_char_word", "checkpoints")

# === Data ===
entries = []
with open(os.path.join(DATA_DIR, "word_list_v2.txt"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        m = re.match(r'!([^@]+)@(.)@(.)', line)
        if m: entries.append((m.group(1), m.group(2), m.group(3)))
        elif len(line) == 2: entries.append((line, line[0], line[1]))
chars_set = set()
for _, c1, c2 in entries: chars_set.add(c1); chars_set.add(c2)
char2idx = {c: i for i, c in enumerate(sorted(chars_set))}

sentences = []
with open(os.path.join(BASE_DIR, "P5_sentence", "sentences_v8k.txt"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split("|")
        if len(parts) != 2: continue
        words, roles = [], []
        for item in parts[1].split():
            try:
                w, r = item.split(":")
                words.append(w); roles.append({"subj": 0, "verb": 1, "obj": 2}.get(r, 0))
            except: pass
        if words: sentences.append((words, roles))
print(f"[Data] {len(sentences)} sentences")

# === Frozen models ===
ckpt = torch.load(os.path.join(SD, "P1_best.pt"), map_location=DEVICE)
p1 = CharToWordModel(ckpt["num_chars"], ckpt["num_words"]).to(DEVICE)
p1.load_state_dict(ckpt["model_state_dict"])
for p in p1.parameters(): p.requires_grad = False; p1.eval()

p5 = SentenceSynthesis().to(DEVICE)
p5.load_state_dict(torch.load(os.path.join(SD, "P5_best.pt"), map_location=DEVICE)["model_state_dict"])
for p in p5.parameters(): p.requires_grad = False; p5.eval()

# === Helpers ===
@torch.no_grad()
def enc_word(w):
    c1, c2 = w[0], w[0] if len(w) == 1 else w[-1]
    if c1 not in char2idx or c2 not in char2idx: return None
    return p1(torch.tensor([[char2idx[c1], char2idx[c2]]], device=DEVICE), last_loss=0.0)[0]

def get_char_vecs(text):
    vecs = []
    for c in text:
        if c not in char2idx: return None
        content = p1.char_content(torch.tensor([char2idx[c]], device=DEVICE))
        pos = p1.pos_encoder.pe[0:1]
        vecs.append(p1.project_char(torch.cat([pos[0], content[0]], dim=-1)))
    return torch.stack(vecs)

# === Bridge data ===
bridge_data = []
for words, roles in sentences:
    wvs = [enc_word(w) for w in words]
    if None in wvs: continue
    sv = p5(torch.stack(wvs), torch.tensor(roles, device=DEVICE), last_loss=0.0)
    bridge_data.append((words, sv, torch.stack(wvs)))
print(f"[Bridge] {len(bridge_data)} groups")

# === Models ===
p8 = CharToSent(max_len=15).to(DEVICE)
p6 = SentToWordsDecoder(max_words=5).to(DEVICE)
n_p = sum(p.numel() for m in [p8, p6] for p in m.parameters())
print(f"[P8+P6] {n_p:,} params")

opt = torch.optim.Adam(list(p8.parameters()) + list(p6.parameters()), lr=0.002, weight_decay=1e-5)

# Multi-objective weights
W_SENT = 5.0; W_WORD = 1.0; W_DIV = 0.1; W_EXP = 0.05
DIV_LIMIT = 0.85; EXP_MIN = 0.02

last_loss = 1.0; best_wc = 0.0; t0 = time.time()
history = []

for epoch in range(1, TOTAL + 1):
    # === Training ===
    el_sent = 0.0; el_word = 0.0; el_div = 0.0; el_exp = 0.0; nb = 0
    random.shuffle(bridge_data)
    for wi, (words, sv_t, wvs_t) in enumerate(bridge_data):
        cvs = get_char_vecs("".join(words))
        if cvs is None: continue
        sp = p8(cvs, last_loss=last_loss)
        wp = p6(sp.unsqueeze(0), last_loss=last_loss)[0]

        # Sent align
        sc = F.cosine_similarity(sp.unsqueeze(0), sv_t.unsqueeze(0), dim=-1)
        sent_align = (1.0 - sc) ** 2

        # Word align
        nw = min(len(wp), len(wvs_t))
        wcs = F.cosine_similarity(wp[:nw], wvs_t[:nw], dim=-1)
        word_align = ((1.0 - wcs) ** 2).mean()

        # Diversity (per 20 samples)
        sent_div = torch.tensor(0.0, device=DEVICE)
        if wi > 0 and wi % 20 == 0:
            with torch.no_grad():
                rs = torch.stack([p8(get_char_vecs("".join(bridge_data[j][0])), last_loss=0.0)
                                  for j in range(max(0, wi-20), wi)
                                  if get_char_vecs("".join(bridge_data[j][0])) is not None])
                if len(rs) >= 5:
                    sn = F.normalize(rs, dim=-1)
                    pc = torch.mm(sn, sn.T)
                    mk = 1 - torch.eye(len(sn), device=DEVICE)
                    sent_div = F.relu((pc * mk).sum() / mk.sum() - DIV_LIMIT) ** 2

        # Explore preservation
        exp_loss = F.relu(EXP_MIN - p8.explore_state.norm()) ** 2 + \
                   F.relu(EXP_MIN - p6.explore_state.norm()) ** 2

        loss = W_SENT * sent_align + W_WORD * word_align + W_DIV * sent_div + W_EXP * exp_loss
        opt.zero_grad(); loss.backward(); opt.step()
        el_sent += sent_align.item(); el_word += word_align.item()
        el_div += sent_div.item(); el_exp += exp_loss.item(); nb += 1
        last_loss = loss.item()

    if epoch % LOG_EVERY != 0 and epoch != 1 and epoch != TOTAL:
        continue

    # === Full Diagnostics ===
    p8.eval(); p6.eval()
    sample = random.sample(bridge_data, min(80, len(bridge_data)))
    sc_vals = []; wc_vals = []; wp_norms = []; sp_norms = []
    for words, sv_t, wvs_t in sample:
        cvs = get_char_vecs("".join(words))
        if cvs is None: continue
        sp = p8(cvs, last_loss=0.0)
        sc_vals.append(F.cosine_similarity(sp.unsqueeze(0), sv_t.unsqueeze(0), dim=-1).item())
        sp_norms.append(sp.norm().item())
        wp = p6(sp.unsqueeze(0), last_loss=0.0)[0]
        nw = min(len(wp), len(wvs_t))
        wc_vals.append(F.cosine_similarity(wp[:nw], wvs_t[:nw], dim=-1).mean().item())
        wp_norms.append(wp[:nw].norm(dim=-1).mean().item())

    # Sentence collapse
    s8s = [p8(get_char_vecs("".join(w)), last_loss=0.0)
           for w, _, _ in sample[:30] if get_char_vecs("".join(w)) is not None]
    sp_all = torch.stack(s8s); sp_n = F.normalize(sp_all, dim=-1)
    pc = torch.mm(sp_n, sp_n.T); mk = 1 - torch.eye(len(sp_n), device=DEVICE)
    sent_self = (pc * mk).sum().item() / max(mk.sum().item(), 1)

    # Gradients
    gP8 = sum(p.grad.norm().item() for p in p8.parameters() if p.grad is not None)
    gP6 = sum(p.grad.norm().item() for p in p6.parameters() if p.grad is not None)
    p8.train(); p6.train()

    # Compute metrics
    aw = sum(wc_vals) / len(wc_vals)
    a_s = sum(sc_vals) / len(sc_vals)
    w8 = sum(1 for c in wc_vals if c > 0.8) / len(wc_vals)
    w9 = sum(1 for c in wc_vals if c > 0.9) / len(wc_vals)
    a_wpn = sum(wp_norms) / len(wp_norms)
    a_spn = sum(sp_norms) / len(sp_norms)
    p8_en = p8.explore_state.norm().item()
    p6_en = p6.explore_state.norm().item()
    p8_mo = p8.meta_fc(p8.explore_state * min(last_loss * 20, 1.0)).norm().item()
    p6_mw = p6.meta_fc[0].weight.norm().item()
    if aw > best_wc: best_wc = aw

    trend = "↑" if len(history) > 0 and aw > history[-1]["word_cos"] else \
            ("↓" if len(history) > 0 and aw < history[-1]["word_cos"] else " ")

    d = {"epoch": epoch, "loss": el_sent/nb*W_SENT + el_word/nb*W_WORD,
         "sent_align": el_sent/nb, "word_align": el_word/nb,
         "sent_div": el_div/nb, "explore": el_exp/nb,
         "sent_cos": a_s, "word_cos": aw, "wc_gt08": w8, "wc_gt09": w9,
         "sent_self": sent_self, "wp_norm": a_wpn, "sp_norm": a_spn,
         "grad_P8": gP8, "grad_P6": gP6,
         "p8_explore_norm": p8_en, "p8_meta_out": p8_mo,
         "p6_explore_norm": p6_en, "p6_meta_w": p6_mw,
         "best": best_wc, "time": time.time() - t0}
    history.append(d)

    print(f"E{epoch:3d} | word={aw:.4f} {trend} | s_cos={a_s:.4f} | w>.8={w8:.2f} w>.9={w9:.2f} | "
          f"s_self={sent_self:.4f} | wp_n={a_wpn:.1f} sp_n={a_spn:.1f} | "
          f"gP8={gP8:.2f} gP6={gP6:.2f} | P8e={p8_en:.3f} P6e={p6_en:.3f} | best={best_wc:.4f} | {d['time']:.0f}s")

elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"Bridge 100-epoch complete ({elapsed:.0f}s, {len(history)} snapshots)")
print(f"Best word_cos: {best_wc:.4f} (epoch {max(history, key=lambda x: x['word_cos'])['epoch']})")
print(f"{'='*70}")

# Trend segments
first5 = sum(h["word_cos"] for h in history[:5]) / min(5, len(history))
last5 = sum(h["word_cos"] for h in history[-5:]) / min(5, len(history))
print(f"Early avg (first 5): {first5:.4f} -> Late avg (last 5): {last5:.4f} "
      f"({'still rising' if last5 > first5 + 0.005 else 'plateaued' if abs(last5-first5) < 0.005 else 'declining'})")

# Save
torch.save({"history": history, "best_wc": best_wc, "p8": p8.state_dict(), "p6": p6.state_dict()},
           os.path.join(SD, "Bridge_long_best.pt"))
print(f"Saved: {SD}/Bridge_long_best.pt")
