"""
数据扩展: 补充常用字词 + 生成多样化训练句
===========================================
输出: data/word_list_expanded.txt, P5_sentence/sentences_expanded.txt
"""
import os, random, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. 加载现有词表
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

existing = load_word_list(os.path.join(BASE_DIR, "data", "word_list.txt"))
existing_words = {w for w, _, _ in existing}
existing_chars = set()
for _, c1, c2 in existing:
    existing_chars.add(c1); existing_chars.add(c2)
print(f"[现有] {len(existing)}词, {len(existing_chars)}字")

# ============================================================
# 2. 补充高频常用词 (覆盖日常对话、问答、指代)
# ============================================================
new_words_raw = [
    # 疑问/指代
    "什么", "是谁", "哪个", "这里", "那里", "这个", "那个", "如何", "怎么",
    # 食物
    "苹果", "香蕉", "西瓜", "葡萄", "米饭", "面条", "面包", "牛奶", "鸡蛋",
    # 娱乐
    "电影", "音乐", "游戏", "小说", "电视", "手机", "电脑", "网络",
    # 日常物品
    "桌子", "椅子", "窗户", "大门", "书本", "铅笔", "衣服", "鞋子",
    # 地点
    "学校", "医院", "公司", "商店", "餐厅", "公园", "超市", "银行", "图书馆",
    # 动作 (补充)
    "吃饭", "睡觉", "走路", "跑步", "游泳", "唱歌", "跳舞", "画画", "拍照",
    "打电话", "发短信", "上网", "下载", "打印",
    # 描述
    "美丽", "聪明", "勇敢", "温柔", "善良", "诚实", "勤劳", "耐心",
    "寒冷", "炎热", "干燥", "潮湿", "安静", "吵闹",
    # 抽象
    "问题", "答案", "方法", "结果", "原因", "机会", "梦想", "希望",
    "爱情", "友情", "家庭", "社会", "国家", "世界",
]

# 解析新词 → 提取新字
new_entries = []
new_chars = set()
for w in new_words_raw:
    if w in existing_words: continue
    if len(w) == 2:
        new_entries.append((w, w[0], w[1]))
        new_chars.add(w[0]); new_chars.add(w[1])
    elif len(w) == 3:
        # 3字词: 用前两字和后两字各做一个entry
        # 简化: 取首尾两字
        w2 = w[0] + w[-1]
        if w2 not in existing_words:
            new_entries.append((w, w[0], w[-1]))
        new_chars.add(w[0]); new_chars.add(w[1]); new_chars.add(w[2])

new_chars_only = new_chars - existing_chars
print(f"[新增] {len(new_entries)}词, {len(new_chars_only)}新字")

# ============================================================
# 3. 合并词表
# ============================================================
all_entries = existing + new_entries
all_words = [w for w, _, _ in all_entries]
all_chars = existing_chars | new_chars
print(f"[合并] {len(all_entries)}词, {len(all_chars)}字")

# 保存
out_path = os.path.join(BASE_DIR, "data", "word_list_expanded.txt")
with open(out_path, "w", encoding="utf-8") as f:
    for w, c1, c2 in all_entries:
        f.write(f"!{w}@{c1}@{c2}\n")
print(f"[保存] {out_path}")

# ============================================================
# 4. 生成多样化训练句 (500+ 句)
# ============================================================
# 主语候选 (agent nouns)
subjects = [
    "老师", "学生", "医生", "护士", "工人", "农民", "警察", "司机",
    "厨师", "画家", "歌手", "演员", "作家", "记者", "律师", "商人",
    "朋友", "同学", "同事", "邻居", "家人", "父母", "孩子", "老人",
    "男孩", "女孩", "男人", "女人", "青年", "少年",
]
# 谓语候选 (transitive verbs)
verbs = [
    "教", "写", "看", "吃", "喝", "买", "卖", "建", "修", "画",
    "唱", "跳", "学", "问", "帮", "找", "拿", "送", "借", "还",
    "读", "讲", "听", "洗", "扫", "开", "关", "推", "拉", "拍",
    "种", "养", "做", "玩", "用", "带", "给", "交", "记", "算",
]
# 宾语候选 (objects/nouns)
objects = [
    "作业", "知识", "病人", "城市", "蔬菜", "药品", "电影", "音乐",
    "书籍", "家具", "房屋", "桥梁", "道路", "汽车", "飞机", "轮船",
    "衣服", "食品", "饮料", "工具", "机器", "花园", "蛋糕", "咖啡",
    "茶水", "报纸", "杂志", "信件", "礼物", "鲜花", "玩具", "乐器",
    "体育", "数学", "语文", "英语", "科学", "历史", "地理", "物理",
    "化学", "生物", "艺术", "文学", "哲学", "经济", "法律", "医学",
    "程序", "数据", "网络", "系统", "软件", "硬件", "芯片", "电池",
]

# 筛选: 只保留P1词表中的词
subjects_ok = [s for s in subjects if s in all_words]
verbs_ok = [v for v in verbs if v in all_words]
objects_ok = [o for o in objects if o in all_words]
print(f"[候选] 主语{len(subjects_ok)} 谓语{len(verbs_ok)} 宾语{len(objects_ok)}")

# 如果候选太少, 放宽筛选 (用包含的字判断)
if len(subjects_ok) < 8:
    subjects_ok = [s for s in subjects if all(c in all_chars for c in s)]
if len(verbs_ok) < 8:
    verbs_ok = [v for v in verbs if v in all_chars]
if len(objects_ok) < 8:
    objects_ok = [o for o in objects if all(c in all_chars for c in o)]

print(f"[放宽后] 主语{len(subjects_ok)} 谓语{len(verbs_ok)} 宾语{len(objects_ok)}")

random.seed(42)
sentences = set()
attempts = 0
while len(sentences) < 500 and attempts < 10000:
    s = random.choice(subjects_ok)
    v = random.choice(verbs_ok)
    o = random.choice(objects_ok)
    # 避免主语=宾语
    if s == o: continue
    sentences.add(f"{s}:subj {v}:verb {o}:obj")
    attempts += 1

# 也生成一些2词短句
while len(sentences) < 600 and attempts < 20000:
    s = random.choice(subjects_ok)
    v = random.choice(verbs_ok)
    sentences.add(f"{s}:subj {v}:verb")
    attempts += 1

sentences = sorted(sentences)
print(f"[生成] {len(sentences)}句")

# 保存句子 (带编号)
sent_path = os.path.join(BASE_DIR, "P5_sentence", "sentences_expanded.txt")
with open(sent_path, "w", encoding="utf-8") as f:
    f.write(f"# V18 扩展训练句 (自动生成, {len(sentences)}句)\n")
    f.write(f"# 格式: 编号 | 词:角色 词:角色 ...\n\n")
    for i, sent in enumerate(sentences):
        f.write(f"{i+1}|{sent}\n")
print(f"[保存] {sent_path}")

print("\n=== 数据扩展完成 ===")
print(f"词表: {len(all_entries)}词 {len(all_chars)}字")
print(f"训练句: {len(sentences)}句")
