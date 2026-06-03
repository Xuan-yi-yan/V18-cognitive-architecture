"""
V18 全链路端到端评估
====================
输入句子 -> P8(char->sent) -> P6(sent->words) -> P1词表匹配 -> Top-1/3/5/10
"""
import torch, torch.nn.functional as F, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import DEVICE
from P1_char_word.model import CharToWordModel
from P5_sentence.model import SentenceSynthesis
from P6_sent_word.model import SentToWordsDecoder
from P8_char_sent.model import CharToSent

SAVE_DIR = os.path.join(os.path.dirname(__file__), "P1_char_word", "checkpoints")


def load_models():
    models = {}

    # P1
    ckpt = torch.load(os.path.join(SAVE_DIR, "P1_best.pt"), map_location=DEVICE)
    p1 = CharToWordModel(ckpt["num_chars"], ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    models["P1"] = p1
    models["c2i"] = ckpt["char2idx"]
    models["i2w"] = ckpt["idx2word"]
    models["w2i"] = ckpt["word2idx"]

    # P8 + P6 (from bridge checkpoint, trained together)
    p8 = CharToSent(max_len=15).to(DEVICE)
    p6 = SentToWordsDecoder(max_words=3).to(DEVICE)
    bridge_path = os.path.join(SAVE_DIR, "P6_bridge.pt")
    if os.path.exists(bridge_path):
        bridge_ckpt = torch.load(bridge_path, map_location=DEVICE)
        p8.load_state_dict(bridge_ckpt["p8"])
        p6.load_state_dict(bridge_ckpt["p6"])
        print(f"[Bridge] sent_cos={bridge_ckpt.get('sent_cos','?')} word_cos={bridge_ckpt.get('word_cos','?')}")
    else:
        # Fallback: independent checkpoints
        p8p = os.path.join(SAVE_DIR, "P8_best.pt")
        if os.path.exists(p8p):
            p8.load_state_dict(torch.load(p8p, map_location=DEVICE)["model_state_dict"])
        p6p = os.path.join(SAVE_DIR, "P6_best.pt")
        if os.path.exists(p6p):
            p6.load_state_dict(torch.load(p6p, map_location=DEVICE)["model_state_dict"])
    for p in p8.parameters(): p.requires_grad = False; p8.eval()
    for p in p6.parameters(): p.requires_grad = False; p6.eval()
    models["P8"] = p8
    models["P6"] = p6

    # P5
    p5 = SentenceSynthesis().to(DEVICE)
    p5p = os.path.join(SAVE_DIR, "P5_best.pt")
    if os.path.exists(p5p):
        p5.load_state_dict(torch.load(p5p, map_location=DEVICE)["model_state_dict"])
    for p in p5.parameters(): p.requires_grad = False; p5.eval()
    models["P5"] = p5

    return models


def evaluate_sentence(text, expected_words, models):
    """评估单句: 返回各级指标"""
    p1 = models["P1"]
    c2i = models["c2i"]
    i2w = models["i2w"]
    p8 = models["P8"]
    p6 = models["P6"]
    p5 = models["P5"]

    result = {"text": text, "expected": expected_words}

    # 1. 字符编码
    chars = list(text)
    char_vecs = []
    for c in chars:
        if c not in c2i:
            result["error"] = f"字符'{c}'不在词表"
            return result
        content = p1.char_content(torch.tensor([c2i[c]], device=DEVICE))
        pos = p1.pos_encoder.pe[0:1]
        char_vecs.append(torch.cat([pos[0], content[0]]))
    cvs = torch.stack(char_vecs)

    # 2. P8: 字->句
    with torch.no_grad():
        sent_vec = p8(cvs, last_loss=0.0)

    # 3. P6: 句->词
    with torch.no_grad():
        word_vecs = p6(sent_vec.unsqueeze(0), last_loss=0.0)[0]

    # 4. 词表匹配: Top-1/3/5/10
    p1_refs = p1.get_all_reference_vectors(DEVICE)
    results_per_slot = []
    for wv in word_vecs:
        if wv.norm() < 0.1: continue
        sims = F.cosine_similarity(wv.unsqueeze(0), p1_refs).squeeze(0)
        sorted_idx = torch.argsort(sims, descending=True)
        results_per_slot.append({
            "top1": (i2w[sorted_idx[0].item()], sims[sorted_idx[0]].item()),
            "top3": [(i2w[sorted_idx[i].item()], sims[sorted_idx[i]].item()) for i in range(min(3, len(sorted_idx)))],
            "top5": [(i2w[sorted_idx[i].item()], sims[sorted_idx[i]].item()) for i in range(min(5, len(sorted_idx)))],
            "top10": [(i2w[sorted_idx[i].item()], sims[sorted_idx[i]].item()) for i in range(min(10, len(sorted_idx)))],
        })
    result["slots"] = results_per_slot

    # 5. P5词级对比
    expected_vecs = []
    for w in expected_words:
        c1, c2 = w[0], w[0] if len(w) == 1 else w[-1]
        if c1 in c2i and c2 in c2i:
            cids = torch.tensor([[c2i[c1], c2i[c2]]], device=DEVICE)
            expected_vecs.append(p1(cids, last_loss=0.0)[0])

    if len(expected_vecs) >= 2:
        e_wvs = torch.stack(expected_vecs)
        roles_t = torch.arange(len(e_wvs), device=DEVICE) % 3
        with torch.no_grad():
            p5_sent = p5(e_wvs, roles_t, last_loss=0.0)
        result["p8_p5_cos"] = F.cosine_similarity(sent_vec.unsqueeze(0), p5_sent.unsqueeze(0), dim=-1).item()

    # 6. 精确匹配检查
    if results_per_slot:
        top1_words = [r["top1"][0] for r in results_per_slot]
        result["top1_match"] = [w for w in top1_words if w in expected_words]
        result["top1_precision"] = len(result["top1_match"]) / max(len(expected_words), 1)

    return result


def batch_evaluate(test_cases, models):
    """批量评估"""
    print(f"\n{'='*80}")
    print(f"V18 全链路端到端评估")
    print(f"{'='*80}")

    all_top1 = []
    all_p8p5 = []

    for text, expected in test_cases:
        r = evaluate_sentence(text, expected, models)
        print(f"\n--- \"{text}\" (期望: {expected}) ---")

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        if "p8_p5_cos" in r:
            print(f"  P8<->P5句子对齐: {r['p8_p5_cos']:.4f}")
            all_p8p5.append(r["p8_p5_cos"])

        for i, slot in enumerate(r.get("slots", [])):
            top1_w, top1_s = slot["top1"]
            match = "V" if top1_w in expected else "X"
            print(f"  槽位{i}: Top-1='{top1_w}' ({top1_s:.3f}) {match}")
            print(f"         Top-3: {[(w, f'{s:.3f}') for w, s in slot['top3']]}")
            print(f"         Top-5: {[(w, f'{s:.3f}') for w, s in slot['top5']]}")

        if "top1_precision" in r:
            all_top1.append(r["top1_precision"])
            print(f"  精确匹配: {r['top1_match']} ({r['top1_precision']:.1%})")

    # 汇总
    print(f"\n{'='*80}")
    print(f"汇总:")
    if all_top1:
        print(f"  Top-1 平均精确率: {sum(all_top1)/len(all_top1):.1%}")
    if all_p8p5:
        print(f"  P8<->P5 平均对齐: {sum(all_p8p5)/len(all_p8p5):.4f}")
    print(f"  测试句数: {len(test_cases)}")
    print(f"{'='*80}")


# ============================================================
# 测试用例
# ============================================================
TEST_CASES = [
    # 训练集中见过的
    ("老师教学生", ["老师", "教", "学生"]),
    ("同学写作业", ["同学", "写", "作业"]),
    ("学生学习知识", ["学生", "学习", "知识"]),
    ("医生帮助病人", ["医生", "帮助", "病人"]),
    ("工人建城市", ["工人", "建", "城市"]),
    # 训练集未见的组合 (泛化测试)
    ("警察建桥梁", ["警察", "建", "桥梁"]),
    ("厨师做蛋糕", ["厨师", "做", "蛋糕"]),
    ("画家画鲜花", ["画家", "画", "鲜花"]),
    ("歌手唱歌", ["歌手", "唱", "歌"]),
    ("司机开汽车", ["司机", "开", "汽车"]),
]

if __name__ == "__main__":
    print("加载模型...")
    models = load_models()
    print(f"P1: {models['P1'].num_chars}字 {models['P1'].num_words}词")
    batch_evaluate(TEST_CASES, models)
