"""下载公开数据集 → V18 A|B格式 (仅HF+MuCGEC, 无本地)"""
import os, random, requests

out_dir = "C:/ai/data/public"
os.makedirs(out_dir, exist_ok=True)
all_pairs = []

# ── 1. HuggingFace (53K对, 主力) ──
print("1. HuggingFace (53K)...")
from datasets import load_dataset
ds = load_dataset("shibing624/chinese_text_correction", split="train")
for row in ds:
    s, t = str(row['source']), str(row['target'])
    # 中文逐字拆分 (无空格分隔)
    sw = list(s.replace(' ','').replace('\u3000',''))  # 去空格,逐字拆分
    tw = list(t.replace(' ','').replace('\u3000',''))
    if len(sw)>=3 and len(tw)>=3 and s!=t:
        all_pairs.append((sw, tw))
print(f"   {len(all_pairs)}对")

# ── 2. MuCGEC (NAACL 2022, ~1K) ──
print("2. MuCGEC...")
try:
    r = requests.get(
        "https://raw.githubusercontent.com/HillZhang1999/MuCGEC/main/data/MuCGEC/MuCGEC_dev.txt",
        headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    if r.status_code == 200:
        for line in r.text.strip().split('\n'):
            p = line.strip().split('\t')
            if len(p)>=2 and p[0] and p[1] and p[0]!=p[1]:
                s = list(p[0].replace(' ','').replace('\u3000',''))
                t = list(p[1].replace(' ','').replace('\u3000',''))
                if len(s)>=3 and len(t)>=3:
                    all_pairs.append((s, t))
    print(f"   {len(all_pairs)}对(累计)")
except Exception as e: print(f"   失败: {e}")

# ── 保存 ──
out_path = os.path.join(out_dir, "public_combined.txt")
random.shuffle(all_pairs)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(f"# V18公开训练集: HF(53K) + MuCGEC(1K)\n# 总: {len(all_pairs)}\n")
    for a,b in all_pairs: f.write(f"{''.join(a)}\t{''.join(b)}\n")
print(f"\n保存: {out_path} ({len(all_pairs)}对)")
