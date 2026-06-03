"""
V18 端到端管线测试
==================
输入一句话 → 自动分词 → 属性绑定 → 完整句法分析
调用链: P8(char→sent) → P6(sent→words) → P1(word_match) → P3(attr)
"""
import torch, torch.nn.functional as F, os, sys, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import DEVICE
from P1_char_word.model import CharToWordModel
from P5_sentence.model import SentenceSynthesis
from P6_sent_word.model import SentToWordsDecoder
from P8_char_sent.model import CharToSent
from P3_word_attr.model import SubjectBindingModel


def load_all():
    """加载所有冻结模型"""
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "P1_char_word", "checkpoints")
    models = {}

    # P1
    ckpt = torch.load(os.path.join(SAVE_DIR, "P1_best.pt"), map_location=DEVICE)
    p1 = CharToWordModel(ckpt["num_chars"], ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    models["P1"] = p1
    models["P1_char2idx"] = ckpt["char2idx"]
    models["P1_idx2word"] = ckpt["idx2word"]
    models["P1_word2idx"] = ckpt["word2idx"]
    models["P1_num_words"] = ckpt["num_words"]
    print(f"[P1] {ckpt['num_chars']}字 {ckpt['num_words']}词")

    # P3_subj
    try:
        p3 = SubjectBindingModel(len(torch.load(os.path.join(SAVE_DIR, "P3_主语_best.pt") if os.path.exists(os.path.join(SAVE_DIR, "P3_主语_best.pt")) else os.path.join(SAVE_DIR, "P3_best.pt"), map_location=DEVICE)["p3_word2id"])).to(DEVICE)
        # Load from generic P3 checkpoints
        for p3_name in ["P3_best.pt"]:
            p3p = os.path.join(SAVE_DIR, p3_name)
            if os.path.exists(p3p):
                p3_ckpt = torch.load(p3p, map_location=DEVICE)
                if "p3_word2id" not in p3_ckpt: continue
                p3 = SubjectBindingModel(len(p3_ckpt["p3_word2id"])).to(DEVICE)
                p3.load_state_dict(p3_ckpt["model"])
                for p in p3.parameters(): p.requires_grad = False; p3.eval()
                models["P3"] = p3
                models["P3_w2id"] = p3_ckpt["p3_word2id"]
                models["P3_family"] = p3_ckpt.get("family_words", [])
                break
        print(f"[P3] 加载属性模型")
    except Exception as e:
        print(f"[P3] 跳过: {e}")

    # P5
    p5 = SentenceSynthesis().to(DEVICE)
    p5p = os.path.join(SAVE_DIR, "P5_final.pt")
    if os.path.exists(p5p):
        s = torch.load(p5p, map_location=DEVICE).get("model_state_dict", {})
        if s: p5.load_state_dict(s)
    for p in p5.parameters(): p.requires_grad = False; p5.eval()
    models["P5"] = p5
    print("[P5] OK")

    # P6
    p6 = SentToWordsDecoder().to(DEVICE)
    for p6_name in ["P6_bridge.pt", "P6_final.pt"]:
        p6p = os.path.join(SAVE_DIR, p6_name)
        if os.path.exists(p6p):
            s = torch.load(p6p, map_location=DEVICE).get("model_state_dict", {})
            if s: p6.load_state_dict(s); break
    for p in p6.parameters(): p.requires_grad = False; p6.eval()
    models["P6"] = p6
    print("[P6] OK")

    # P8
    p8 = CharToSent(max_len=15).to(DEVICE)
    p8p = os.path.join(SAVE_DIR, "P8_final.pt")
    if os.path.exists(p8p):
        s = torch.load(p8p, map_location=DEVICE).get("model_state_dict", {})
        if s: p8.load_state_dict(s)
    for p in p8.parameters(): p.requires_grad = False; p8.eval()
    models["P8"] = p8
    print("[P8] OK")

    return models


def encode_chars(text, p1, c2i):
    """字符串 → 字符向量序列"""
    vecs = []
    chars = list(text)
    for c in chars:
        if c not in c2i:
            print(f"  [跳过] 字符'{c}'不在P1表中")
            return None, None
        content = p1.char_content(torch.tensor([c2i[c]], device=DEVICE))
        pos = p1.pos_encoder.pe[0:1]
        vecs.append(torch.cat([pos[0], content[0]]))
    return torch.stack(vecs), chars


def decode_words(sent_vec, p6, p1, p1_w2i, p1_i2w, c2i, n_heads=5):
    """句子向量 → 词向量序列 → P1词表匹配"""
    with torch.no_grad():
        word_vecs = p6(sent_vec.unsqueeze(0), last_loss=0.0)[0]  # [max_words, 32]

    # 空槽过滤
    results = []
    for i in range(len(word_vecs)):
        wv = word_vecs[i]  # [32]
        if wv.norm() < 0.1:
            continue  # 空槽

        # 与P1词表匹配(用语义区24D)
        best_word = None; best_sim = -1
        wv_sem = wv[8:]  # 取120D语义部分
        for j in range(len(p1_i2w)):
            w = p1_i2w[j]
            if w not in p1_w2i: continue
            # 用P1编码这个词
            c1, c2 = w[0], w[0] if len(w) == 1 else w[1]
            if c1 not in c2i or c2 not in c2i: continue
            cids = torch.tensor([[c2i[c1], c2i[c2]]], device=DEVICE)
            ref = p1(cids, last_loss=0.0)[0]
            ref_sem = ref[8:]
            sim = F.cosine_similarity(wv_sem.unsqueeze(0), ref_sem.unsqueeze(0), dim=-1).item()
            if sim > best_sim:
                best_sim = sim; best_word = w

        if best_word and best_sim > 0.3:
            results.append((best_word, best_sim))
    return results


def analyze(text, models):
    """完整管线分析"""
    p1 = models["P1"]
    c2i = models["P1_char2idx"]
    i2w = models["P1_idx2word"]
    w2i = models["P1_word2idx"]
    p5 = models["P5"]
    p6 = models["P6"]
    p8 = models["P8"]

    print(f"\n{'='*60}")
    print(f"V18 管线分析: \"{text}\"")
    print(f"{'='*60}")

    # Step 1: 字符编码
    char_vecs, chars = encode_chars(text, p1, c2i)
    if char_vecs is None: return
    print(f"[字符] {chars} → {char_vecs.shape[0]}个字符向量")

    # Step 2: P8 字符→句子
    with torch.no_grad():
        sent_vec = p8(char_vecs, last_loss=0.0)
    print(f"[P8] 字符→句子: 256D向量")

    # Step 3: 验证P5对齐
    # Try word-level if we can guess the words
    # For now just show P8 output

    # Step 4: P6 句子→词序列
    words = decode_words(sent_vec, p6, p1, w2i, i2w, c2i, n_heads=5)
    print(f"[P6] 句子→词序列: {len(words)}个候选词")
    for w, sim in words:
        print(f"  '{w}' (cos={sim:.3f})")

    # Step 5: P3 属性绑定 (如果有)
    if "P3" in models:
        p3 = models["P3"]
        p3_w2id = models["P3_w2id"]
        family_words = models["P3_family"]

        # 编码主语族P1向量
        @torch.no_grad()
        def enc_w(w):
            c1, c2 = w[0], w[0] if len(w)==1 else w[1]
            cids = torch.tensor([[c2i[c1], c2i[c2]]], device=DEVICE)
            return p1(cids, last_loss=0.0)[0]

        family_p1 = torch.stack([enc_w(w) for w in family_words if w[0] in c2i])

        print(f"\n[P3 属性分析]")
        for w, sim in words:
            if w in p3_w2id:
                wid = torch.tensor([p3_w2id[w]], device=DEVICE)
                score = p3.binding_score(wid, family_p1).item() if len(family_p1) > 0 else 0
                attr = "主语" if score > 0.5 else "?"
                print(f"  '{w}': {attr} (绑定={score:.3f})")
            else:
                print(f"  '{w}': P3词表外")

    # Step 6: 尝试用P5合成标准句子向量做对比
    # Try to build a word-level representation and compare
    print(f"\n[总结]")
    print(f"  输入: {text}")
    print(f"  字符: {''.join(chars)}")
    if words:
        print(f"  分词: {' '.join(w for w,_ in words)}")
    print(f"{'='*60}\n")


# ==================== 测试用例 ====================
if __name__ == "__main__":
    print("加载 V18 全模型...")
    models = load_all()

    tests = [
        "老师教学生",
        "同学写作业",
        "学生学习知识",
        "医生帮助病人",
        "工人建城市",
        "农民种蔬菜",
        "警察保城市",
        "护士备药品",
        "朋友看电影",
        "春天来花开",
    ]

    for t in tests:
        try:
            analyze(t, models)
        except Exception as e:
            print(f"  [错误] {e}")
            import traceback; traceback.print_exc()
