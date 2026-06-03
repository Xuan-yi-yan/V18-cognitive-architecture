# V18 认知架构 (Cognitive Architecture)

**全链路 128D 统一表征** — 从字符到对话的九层神经网络管线。

## 架构概览

```
字符 → [P1] → 词语 → [P5] → 句子 → [P7] → 跨句路由
  ↕ [P2]       ↕ [P3]       ↕ [P6]
字符 ← 词语   属性绑定    句子 → 词语
  ↕ [P9]       ↕ [P8]
字符 ← 句子   字符 → 句子
```

## 层级说明

| 层级 | 功能 | 输入→输出 | 关键指标 |
|------|------|-----------|----------|
| **P1** | 字符→词语 | 2字→128D词向量 | Top-1: 99.89%, 1800词 |
| **P2** | 词语→字符 | 128D词向量→2字 | 余弦: 99.28% |
| **P3** | 属性绑定 | 词语→7属性分类 | 100% (主谓宾定状补虚) |
| **P5** | 词序列→句子 | n×128D→256D句子 | gap: 1.459 |
| **P6** | 句子→词序列 | 256D→n×128D | 余弦: 86-99% |
| **P7** | 跨句路由 | A词→B句预测 | 余弦: ~99.98% |
| **P8** | 字符→句子 | 字符序列→256D句子 | 余弦: 99.98% |
| **P9** | 统一管线 | 字符→句子+词向量 | 端到端 |

## 技术要点

- **统一 128 维度**: 无投影层，信息无损穿透全链路
- **64 头交叉注意力**: 256D Q/K/V 投影，4D/头
- **探索区+元学习调制链**: loss 驱动的自适应调制
- **白箱设计**: 位置(8D sin/cos)与语义(120D)解耦
- **分形递归**: P1 的字→词逻辑同构复用到 P5 的词→句

## 环境要求

- Python 3.10+, PyTorch 2.12+, CUDA 12.8
- RTX 5070 (12GB) 或相似 GPU
- 显存占用: ~75MB (极轻量)

## 快速开始

```bash
# 训练全链路
cd P1_char_word && python train.py       # P1 字符→词语
cd ../P2_word_char && python train.py    # P2 词语→字符
cd ../P3_word_attr && python train_batch.py  # P3 七属性
cd ../P5_sentence && python train.py     # P5 句子合成
cd ../P6_sent_word && python train.py    # P6 句子→词序列
cd ../P8_char_sent && python train.py    # P8 字符→句子

# 端到端测试
python pipeline_test.py

# 分词压力测试
python segment_test.py
```

## 目录结构

```
C:/ai/
  utils/config.py          # 全局配置
  data/word_list.txt       # 1800词训练数据
  P1_char_word/            # 字符→词语
  P2_word_char/            # 词语→字符
  P3_word_attr/            # 属性绑定
  P3_data/                 # 七属性词表
  P5_sentence/             # 词序列→句子
  P6_sent_word/            # 句子→词序列
  P7_cross_sent/           # 跨句路由
  P8_char_sent/            # 字符→句子
  P9_sent_char/            # 句子→字符
  P9_pipeline/             # 统一管线
  pipeline_test.py         # 端到端测试
  segment_test.py          # 分词测试
```

## 日志

详见 [V18_2026-06-03_开发日志.txt](V18_2026-06-03_开发日志.txt)
