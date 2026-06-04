"""P1种子筛选: 训练P1+P2, 找P2>=90%的种子"""
import torch, torch.nn.functional as F, time, os, sys, re, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P2_word_char.model import WordToCharDecoder

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
all_pairs = [[char2idx[c1], char2idx[c2]] for _, c1, c2 in entries]
N = len(entries)

@torch.no_grad()
def eval_p1(p1):
    p1.eval()
    ref = p1.get_all_reference_vectors(DEVICE); ref_n = F.normalize(ref, dim=-1)
    correct = 0
    for i in range(0, N, 100):
        end = min(i+100, N)
        batch_p = torch.tensor([all_pairs[j] for j in range(i, end)], device=DEVICE)
        sims = torch.mm(F.normalize(p1(batch_p, last_loss=0.0), dim=-1), ref_n.T)
        correct += (sims.argmax(dim=-1) == torch.arange(i, end, device=DEVICE)).sum().item()
    p1.train()
    return correct / N

seeds = [42, 123, 789]
results = []

for seed in seeds:
    torch.manual_seed(seed); random.seed(seed)
    print(f"\nSEED {seed}", flush=True)

    # P1
    p1 = CharToWordModel(len(char_list), N).to(DEVICE)
    opt = torch.optim.Adam(p1.parameters(), lr=0.005, weight_decay=WEIGHT_DECAY)
    ll = 1.0; best_t1 = 0.0; ni = 0
    for e in range(1, 200):
        el = 0.0; nb = 0
        opt.zero_grad()
        perm = torch.randperm(N)
        for acc in range(25):
            s = acc * 64; idxs = perm[s:s+64]
            if len(idxs) == 0: break
            pids = torch.tensor([all_pairs[i] for i in idxs], device=DEVICE)
            wids = torch.tensor([i for i in idxs], device=DEVICE)
            pred = p1(pids, last_loss=ll)
            target = p1.word_table[wids]
            # Pearson loss (same as train_all.py)
            pm = pred.mean(dim=0, keepdim=True); tm = target.mean(dim=0, keepdim=True)
            pc = pred - pm; tc = target - tm
            num = (pc * tc).sum()
            den = torch.sqrt((pc**2).sum() * (tc**2).sum() + 1e-8)
            loss = (1.0 - num / den) / 25
            loss.backward()
            el += loss.item() * 25; nb += 1
            ll = loss.item() * 25
        torch.nn.utils.clip_grad_norm_(p1.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if e % 10 == 0:
            t1 = eval_p1(p1)
            if t1 > best_t1: best_t1 = t1; ni = 0
            else: ni += 10
            if ni >= 30: break
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    del opt; torch.cuda.empty_cache()

    # P2
    p2 = WordToCharDecoder().to(DEVICE)
    opt = torch.optim.Adam(p2.parameters(), lr=0.001)
    ll = 1.0; best_p2 = 0.0
    for e in range(1, 200):
        es1 = 0.0; es2 = 0.0; enb = 0
        perm = torch.randperm(N)
        for bs in range(0, N, 200):
            idxs = perm[bs:bs+200]
            pids = torch.tensor([all_pairs[i] for i in idxs], device=DEVICE)
            with torch.no_grad():
                _, _, full = p1.get_char_vectors(pids)
                rc1 = p1.project_char(full[:,0,:]); rc2 = p1.project_char(full[:,1,:])
                wv = p1(pids, last_loss=ll)
            pc1, pc2 = p2(wv)
            s1 = F.cosine_similarity(pc1, rc1, dim=-1).mean()
            s2 = F.cosine_similarity(pc2, rc2, dim=-1).mean()
            loss = (1.0 - (s1 + s2) / 2.0)
            opt.zero_grad(); loss.backward(); opt.step()
            es1 += s1.item(); es2 += s2.item(); enb += 1
            ll = loss.item()
        torch.cuda.empty_cache()
        if e % 40 == 0:
            avg = (es1 + es2) / (2 * enb)
            if avg > best_p2: best_p2 = avg
    torch.cuda.empty_cache()

    results.append((seed, best_t1, best_p2))
    del p1, p2, opt; torch.cuda.empty_cache()
    print(f"Seed {seed}: P1={best_t1:.4%} P2={best_p2:.4%}", flush=True)

print("\n" + "="*50)
print("RESULTS (sorted by P2)")
for seed, t1, p2 in sorted(results, key=lambda x: -x[2]):
    flag = "*** BEST ***" if p2 >= 0.90 else ""
    print(f"Seed {seed:4d}: P1={t1:.4%}  P2={p2:.4%}  {flag}")
