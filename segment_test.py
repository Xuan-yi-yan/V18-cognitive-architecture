"""
V18 分词能力压力测试 (零训练, 纯推理)
======================================
输入无空格汉字序列 → P9管线 → 自动找词边界
"""
import torch, torch.nn.functional as F, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import DEVICE
from P1_char_word.model import CharToWordModel
from P9_pipeline.model import UnifiedPipeline

SAVE_DIR = os.path.join(os.path.dirname(__file__), "P1_char_word", "checkpoints")

def load_models():
    # P1
    ckpt = torch.load(os.path.join(SAVE_DIR, "P1_best.pt"), map_location=DEVICE)
    p1 = CharToWordModel(ckpt["num_chars"], ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    c2i = ckpt["char2idx"]; i2w = ckpt["idx2word"]; w2i = ckpt["word2idx"]

    # P9
    p9 = UnifiedPipeline(max_words=5).to(DEVICE)
    p9p = os.path.join(SAVE_DIR, "P9_final.pt")
    if not os.path.exists(p9p):
        p9p = os.path.join(SAVE_DIR, "P9_best.pt")
    if os.path.exists(p9p):
        s = torch.load(p9p, map_location=DEVICE).get("model_state_dict", {})
        if s: p9.load_state_dict(s)
    for p in p9.parameters(): p.requires_grad = False; p9.eval()

    return p1, p9, c2i, i2w, w2i


def segment(text, p1, p9, c2i, i2w, w2i):
    """
    输入: "老师教学生" (无空格)
    输出: [(词, 置信度), ...]
    """
    chars = list(text)
    n = len(chars)

    # 编码所有字符
    char_vecs = []
    for c in chars:
        if c not in c2i:
            print(f"  [跳过] 字符'{c}'不在P1表中"); return []
        content = p1.char_content(torch.tensor([c2i[c]], device=DEVICE))
        pos = p1.pos_encoder.pe[0:1]
        char_vecs.append(torch.cat([pos[0], content[0]]))
    char_batch = torch.stack(char_vecs)  # [n_chars, 128]

    # === 方法1: P9直接输出词向量, 匹配P1词表 ===
    print(f"\n  [方法1] P9直接解码:")
    with torch.no_grad():
        sent_vec, word_vecs = p9(char_batch.unsqueeze(0), last_loss=0.0)
    word_vecs = word_vecs[0]  # [5, 128]

    # 每个词槽位匹配最接近的P1词
    word_matches = []
    for i in range(5):
        wv = word_vecs[i]
        if wv.norm() < 0.01: continue
        wv_sem = wv[8:]
        best_w, best_s = None, -1
        for j in range(len(i2w)):
            w = i2w[j]
            if len(w) != 2 or w not in w2i: continue
            c1, c2 = w[0], w[1]
            if c1 not in c2i or c2 not in c2i: continue
            cids = torch.tensor([[c2i[c1], c2i[c2]]], device=DEVICE)
            ref = p1(cids, last_loss=0.0)[0]
            ref_sem = ref[8:]
            sim = F.cosine_similarity(wv_sem.unsqueeze(0), ref_sem.unsqueeze(0), dim=-1).item()
            if sim > best_s:
                best_s = sim; best_w = w
        if best_w and best_s > 0.3:
            word_matches.append((best_w, best_s, i))

    print(f"    词槽位匹配: {word_matches}")
    matched_words = [w for w, _, _ in word_matches]

    # === 方法2: 枚举所有2-gram, 检查在P9句子空间的一致性 ===
    print(f"\n  [方法2] 2-gram滑动窗口分析:")
    all_2grams = []
    for i in range(n - 1):
        bigram = chars[i] + chars[i+1]
        # 编码这个词
        if chars[i] in c2i and chars[i+1] in c2i:
            cids = torch.tensor([[c2i[chars[i]], c2i[chars[i+1]]]], device=DEVICE)
            with torch.no_grad():
                wv = p1(cids, last_loss=0.0)[0]
            # 与P9句子向量中对应位置的词向量比较
            if len(word_vecs) > 0:
                sims = [F.cosine_similarity(wv[8:].unsqueeze(0), word_vecs[j][8:].unsqueeze(0), dim=-1).item() for j in range(min(5, len(word_vecs)))]
                max_sim = max(sims)
                all_2grams.append((bigram, max_sim, sims))
                status = "██" if max_sim > 0.9 else ("▓▓" if max_sim > 0.7 else "··")
                print(f"    [{i}:{i+2}] '{bigram}' max_cos={max_sim:.3f} {status} slots={[f'{s:.2f}' for s in sims]}")
            else:
                all_2grams.append((bigram, 0, []))

    # === 方法3: 用P9句子向量反查 ===
    print(f"\n  [方法3] 句子向量一致性验证:")
    with torch.no_grad():
        sent_vec_norm = F.normalize(sent_vec[0], dim=-1)
    print(f"    句子向量 L2={sent_vec_norm.norm().item():.3f}")

    return matched_words, all_2grams


# ==================== 测试用例 ====================
if __name__ == "__main__":
    print("加载 V18 模型...")
    p1, p9, c2i, i2w, w2i = load_models()
    print(f"P1: {len(c2i)}字 {len(i2w)}词\n")

    tests = [
        ("老师教学生", ["老师", "教", "学生"]),
        ("同学写作业", ["同学", "写", "作业"]),
        ("朋友看电影", ["朋友", "看", "电影"]),
        ("医生帮助病人", ["医生", "帮助", "病人"]),
        ("学生学习知识", ["学生", "学习", "知识"]),
        ("工人建城市", ["工人", "建", "城市"]),
        ("春天来花开", ["春天", "来", "花", "开"]),
    ]

    for text, expected in tests:
        print(f"\n{'='*60}")
        print(f"输入: \"{text}\" (无空格, {len(text)}字)")
        print(f"期望: {expected}")
        print(f"{'='*60}")

        try:
            words, grams = segment(text, p1, p9, c2i, i2w, w2i)
        except Exception as e:
            print(f"  [错误] {e}")

    print(f"\n{'#'*60}")
    print("# 压力测试结论")
    print("# 评分标准: 分词准确率 / 2-gram边界检测 / 句子向量一致性")
    print(f"{'#'*60}")
