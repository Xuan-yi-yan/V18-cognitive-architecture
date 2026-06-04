"""
V18 大规模数据扩展 v2
=====================
1. 从 jieba 词典提取高频 2 字中文词 (5000+)
2. 字符表扩展到 3500+
3. 生成 2000+ 多样化训练句
"""
import os, sys, re, random
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. 从 jieba 词典提取高频中文词
# ============================================================
def load_jieba_dict():
    """加载 jieba 词典，筛选 2 字中文高频词"""
    import jieba
    dict_paths = []
    for p in jieba.__path__:
        dp = os.path.join(p, 'dict.txt')
        if os.path.exists(dp):
            dict_paths.append(dp)

    if not dict_paths:
        print("[ERROR] jieba dict not found")
        return [], set()

    words = []
    all_chars = set()
    chinese_char = re.compile(r'^[\u4e00-\u9fff]$')

    with open(dict_paths[0], 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            word, freq = parts[0], int(parts[1])
            # 只取 2 字纯中文词，频率 >= 10
            if len(word) == 2 and chinese_char.match(word[0]) and chinese_char.match(word[1]):
                if freq >= 10:
                    words.append((word, freq))
                    all_chars.add(word[0])
                    all_chars.add(word[1])

    # 按频率排序，取前 6000 词
    words.sort(key=lambda x: -x[1])
    words = words[:6000]
    return words, all_chars


def load_existing_chars():
    """加载已有词表中的字符"""
    existing_path = os.path.join(BASE_DIR, "data", "word_list.txt")
    chars = set()
    if os.path.exists(existing_path):
        with open(existing_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                m = re.match(r'!([^@]+)@(.)@(.)', line)
                if m:
                    chars.add(m.group(2))
                    chars.add(m.group(3))
                elif len(line) == 2:
                    chars.add(line[0])
                    chars.add(line[1])
    return chars


# ============================================================
# 2. 生成扩展字符表（补充到 3500+）
# ============================================================
def expand_chars(existing_chars, jieba_chars):
    """合并现有字符 + jieba 字符 + CJK 常用区"""
    all_c = existing_chars | jieba_chars

    # 补充 CJK Unified Ideographs 前段（最常用区域）
    # U+4E00 to U+9FFF = 20,992 chars
    # 常用 3500 字主要在 U+4E00-U+6FFF 范围
    for cp in range(0x4E00, 0x6FFF):
        c = chr(cp)
        if len(all_c) >= 4000:
            break
        all_c.add(c)

    return sorted(all_c)


# ============================================================
# 3. 词性分类（基于简单规则）
# ============================================================
def classify_words(words):
    """将词按简单规则分为名词/动词/形容词/其他"""
    # 常见名词尾字
    noun_ends = set('人子头员家师生手工匠兵官兵众亲友敌客主长'
                    '者性体物器机具料品件处所部院院校厂店馆'
                    '山水火风雷雨雪云天地日月星'
                    '花草树木竹石金银铜铁'
                    '身心眼耳口鼻手足脑'
                    '国家城乡市县镇村里街道路桥'
                    '时节年月日秒分点刻'
                    '前后左右上下内外东西南北中')

    # 常见动词特征字
    verb_chars = set('打把抓拿推拉开关读写看听说唱跳跑走'
                     '吃喝吞咽尝闻听学习问思考试'
                     '买卖出入进退还送借给'
                     '建立修造制作加工改变'
                     '开始停止结束继续完成'
                     '欢喜爱恨怕想忘记'
                     '来去上下进出过回')

    nouns = []; verbs = []; adjs = []; others = []

    for w, freq in words:
        if w[-1] in noun_ends:
            nouns.append((w, freq))
        elif w[0] in verb_chars or w[-1] in verb_chars:
            verbs.append((w, freq))
        elif freq >= 500:  # 高频词大概率是功能词/形容词
            adjs.append((w, freq))
        else:
            others.append((w, freq))

    # 确保各类都有足够数量
    # 如果名词不够，从 others 补充
    if len(nouns) < 500:
        nouns.extend([(w, f) for w, f in others[:500-len(nouns)] if (w, f) not in nouns])

    return nouns, verbs, adjs, others


# ============================================================
# 4. 保存词表
# ============================================================
def save_word_list(words, char_list, outpath):
    char2idx = {c: i for i, c in enumerate(char_list)}
    with open(outpath, "w", encoding="utf-8") as f:
        for w, _ in words:
            c1, c2 = w[0], w[1]
            f.write(f"!{w}@{c1}@{c2}\n")
    print(f"[词表] {len(words)}词 -> {outpath}")


# ============================================================
# 5. 生成多样化训练句
# ============================================================
def generate_sentences(nouns, verbs, adjs, others, target=2000):
    """生成多种句型的训练句"""
    sentences = set()
    random.seed(42)

    noun_list = [w for w, _ in nouns]
    verb_list = [w for w, _ in verbs]
    adj_list = [w for w, _ in adjs]
    other_list = [w for w, _ in others]

    # 句型 1: SVO (主语+动词+宾语) — 50%
    svo_target = target // 2
    attempts = 0
    while len([s for s in sentences if s.count(':') == 3]) < svo_target and attempts < 50000:
        s = random.choice(noun_list)
        v = random.choice(verb_list)
        o = random.choice(noun_list)
        if s != o:
            sentences.add(f"{s}:subj {v}:verb {o}:obj")
        attempts += 1

    # 句型 2: SV (主语+动词) — 15%
    sv_target = target * 15 // 100
    while len([s for s in sentences if s.count(':') == 2 and 'verb' in s and 'obj' not in s]) < sv_target:
        s = random.choice(noun_list)
        v = random.choice(verb_list)
        sentences.add(f"{s}:subj {v}:verb")

    # 句型 3: Adj+N+V+N (定语+主语+动词+宾语) — 20%
    adj_n_v_n_target = target * 20 // 100
    while len([s for s in sentences if s.count(':') == 4]) < adj_n_v_n_target:
        adj = random.choice(adj_list)
        s = random.choice(noun_list)
        v = random.choice(verb_list)
        o = random.choice(noun_list)
        if s != o:
            sentences.add(f"{adj}:adj {s}:subj {v}:verb {o}:obj")

    # 句型 4: S+V+Adj+N (主语+动词+定语+宾语) — 15%
    while len(sentences) < target:
        s = random.choice(noun_list)
        v = random.choice(verb_list)
        adj = random.choice(adj_list)
        o = random.choice(noun_list)
        if s != o:
            sentences.add(f"{s}:subj {v}:verb {adj}:adj {o}:obj")

    sentences = sorted(sentences)
    return sentences[:target]


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("V18 大规模数据扩展 v2")
    print("=" * 60)

    # Step 1: 加载 jieba 词典
    print("\n[1/6] 加载 jieba 词典...")
    jieba_words, jieba_chars = load_jieba_dict()
    print(f"  jieba 高频词: {len(jieba_words)} (频率>=10)")
    print(f"  jieba 字符: {len(jieba_chars)}")

    # Step 2: 加载已有字符
    print("\n[2/6] 加载已有字符...")
    existing_chars = load_existing_chars()
    print(f"  已有: {len(existing_chars)} 字")

    # Step 3: 扩展字符
    print("\n[3/6] 扩展字符表...")
    char_list = expand_chars(existing_chars, jieba_chars)
    print(f"  扩展后: {len(char_list)} 字")

    # Step 4: 词性分类
    print("\n[4/6] 词性分类...")
    nouns, verbs, adjs, others = classify_words(jieba_words)
    print(f"  名词: {len(nouns)}, 动词: {len(verbs)}, 形容: {len(adjs)}, 其他: {len(others)}")
    print(f"  名词示例: {[w for w,_ in nouns[:10]]}")
    print(f"  动词示例: {[w for w,_ in verbs[:10]]}")

    # Step 5: 保存词表
    print("\n[5/6] 保存词表...")
    out_path = os.path.join(BASE_DIR, "data", "word_list_v2.txt")
    save_word_list(jieba_words, char_list, out_path)

    # Step 6: 生成句子
    print("\n[6/6] 生成训练句...")
    sentences = generate_sentences(nouns, verbs, adjs, others, target=2000)
    sent_path = os.path.join(BASE_DIR, "P5_sentence", "sentences_v2.txt")
    with open(sent_path, "w", encoding="utf-8") as f:
        f.write(f"# V18 v2 训练句 (自动生成, {len(sentences)}句)\n")
        f.write(f"# 句型分布: SVO+SV+AdjN+SAdjN\n\n")
        for i, sent in enumerate(sentences):
            f.write(f"{i+1}|{sent}\n")

    # 统计句型分布
    type_counts = Counter()
    for s in sentences:
        n_roles = s.count(':')
        type_counts[f"{n_roles}-role"] += 1
    print(f"  生成: {len(sentences)} 句")
    print(f"  句型分布: {dict(type_counts)}")
    print(f"  保存: {sent_path}")

    print(f"\n=== 扩展完成 ===")
    print(f"字符: {len(char_list)}")
    print(f"词汇: {len(jieba_words)}")
    print(f"句子: {len(sentences)}")
