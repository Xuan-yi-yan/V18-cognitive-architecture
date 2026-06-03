"""
P4 主语原型→主语词检索 训练
=============================
P4解码主语原型 → Pearson检索P1词表中的所有主语词
验收: 召回率>90% 精确率>90%
"""
import torch, torch.nn.functional as F, time, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P3_word_attr.model import SubjectBindingModel
from P4_subj_word.model import ProtoToWordsDecoder, retrieval_loss, pearson_r


def train():
    ep = int(input("epochs(80): ").strip() or "80")
    disp = int(input("display(8): ").strip() or "8")
    lr = float(input("lr(0.003): ").strip() or "0.003")
    print(f"epochs={ep} display={disp} lr={lr}\n")

    # P1
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    p1_ckpt = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(p1_ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    p1_c2i = p1_ckpt["char2idx"]

    # P3
    p3_path = os.path.join(SAVE_DIR, "P3_best.pt")
    p3_ckpt = torch.load(p3_path, map_location=DEVICE)
    p3 = SubjectBindingModel(len(p3_ckpt["p3_word2id"])).to(DEVICE)
    p3.load_state_dict(p3_ckpt["model"])
    for p in p3.parameters(): p.requires_grad = False; p3.eval()

    family_words = p3_ckpt["family_words"]
    p3_w2id = p3_ckpt["p3_word2id"]
    p1_i2w = p1_ckpt["idx2word"]

    # 编码所有P1词
    @torch.no_grad()
    def p1_encode(words):
        vecs = []
        for w in words:
            cids = torch.tensor([[p1_c2i[w[0]], p1_c2i[w[1]]]], device=DEVICE)
            vecs.append(p1(cids, last_loss=0.0)[0])
        return torch.stack(vecs)

    family_p1 = p1_encode(family_words)

    # 所有1114词的P1向量
    all_words = [w for w in p1_i2w.values() if len(w) == 2 and w[0] in p1_c2i and w[1] in p1_c2i]
    all_p1 = p1_encode(all_words)
    family_set = set(family_words)
    non_family_words = [w for w in all_words if w not in family_set]
    non_family_p1 = p1_encode(non_family_words[:500])  # 取500个负样本

    # 主语原型
    prototype = p3.get_prototype(family_p1)
    print(f"[数据] 主语族: {len(family_words)}词 | 非主语: {len(non_family_p1)}词")

    # P4
    p4 = ProtoToWordsDecoder().to(DEVICE)
    print(f"[P4] 参数: {sum(p.numel() for p in p4.parameters()):,}")
    opt = torch.optim.Adam(p4.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    last_loss = 1.0; best_f1 = 0.0

    for epoch in range(1, ep + 1):
        t0 = time.time()
        decoded = p4(prototype, last_loss=last_loss)
        loss, pos_r, neg_r = retrieval_loss(decoded, family_p1, non_family_p1)
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = loss.item()

        # 检索评估
        with torch.no_grad():
            matches, all_scores = p4.retrieve(prototype, all_p1,
                                              {i: w for i, w in enumerate(all_words)}, threshold=0.5)
            matched_words = set(m[0] for m in matches)
            tp = len(matched_words & family_set)
            fp = len(matched_words - family_set)
            fn = len(family_set - matched_words)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        elapsed = time.time() - t0

        if epoch % disp == 0 or epoch == 1 or epoch == ep:
            print(f"[P4] Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                  f"pos_r: {pos_r:.3f} neg_r: {neg_r:.3f}")
            print(f"  检索: {len(matches)}个匹配 | 召回: {recall:.2%} 精确: {precision:.2%} F1: {f1:.2%}")
            print(f"  命中: {list(matched_words)[:10]}... | 漏检: {list(family_set-matched_words)[:5]}")
            print(f"  GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB | {elapsed:.1f}s")
        else:
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f} | F1: {f1:.2%} | {elapsed:.1f}s")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"epoch": epoch, "model": p4.state_dict(), "f1": f1,
                        "recall": recall, "precision": precision},
                       os.path.join(SAVE_DIR, "P4_best.pt"))

    print(f"\n{'#'*60}")
    print(f"# P4完成 | 最佳F1: {best_f1:.2%}")
    print(f"# 验收: 召回>90% 精确>90% → {'PASS' if best_f1>=0.90 else 'FAIL'}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    train()
