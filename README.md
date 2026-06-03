# V18 Cognitive Architecture · 认知架构

**A 9-layer white-box neural pipeline for Chinese language understanding — from characters to sentences via 128D unified representations and 64-head cross-attention.**

**九层白箱神经管线 — 128D 统一表征，64 头交叉注意力，字符到句子全闭环。**

---

## Architecture Diagram · 架构全景图

```
                        ┌──────────────────────────────────────┐
                        │         V18 COGNITIVE CORE           │
                        │    128D uniform representation       │
                        │    64-head cross-attention           │
                        │    Exploration + Meta-learning       │
                        └──────────────────────────────────────┘

    CHARACTER LEVEL              WORD LEVEL              SENTENCE LEVEL
    ─────────────               ──────────              ──────────────

    ┌──────────┐               ┌──────────┐            ┌──────────┐
    │   c₁ c₂  │──── P1 ──────▶│  w_vec   │─── P5 ───▶│ s_vec    │─── P7 ──▶ cross-sent
    │  (char)  │◀─── P2 ───────│ (128D)   │◀── P6 ────│ (256D)   │           routing
    └────┬─────┘               └────┬─────┘            └────┬─────┘
         │                          │                       │
         │         ┌────────────────┤                       │
         │         │    P3: 7-attr  │                       │
         │         │    binding     │                       │
         │         │  subj/verb/obj │                       │
         │         │  adj/adv/comp  │                       │
         │         │     func       │                       │
         │         └────────────────┘                       │
         │                                                  │
         └────────────── P8 ───────────────────────────────▶│
         ◀───────────────────── P9 ─────────────────────────┘

         ┌──────────────────────────────────────────────────┐
         │           FOUR CLOSED LOOPS · 四级闭环           │
         │  P1 ↔ P2 : char ↔ word     (余弦 99.28%)        │
         │  P5 ↔ P6 : word ↔ sentence (余弦 86-99%)        │
         │  P8 ↔ P9 : char ↔ sentence (余弦 99.98%)        │
         │  P7     : cross-sentence routing (余弦 ~99.98%)  │
         └──────────────────────────────────────────────────┘
```

### Data Flow · 数据流

```
  "学生学习知识"
       │
       ▼
  [学, 习, 生, 学, 习, 知, 识]          ← raw characters
       │
       ├── P1 ──▶ [学生, 学习, 知识]     ← word vectors (128D each)
       │              │
       │              ├── P3 ──▶ subj=学生, verb=学习, obj=知识
       │              │
       │              └── P5 ──▶ [0.91·subj + 0.14·verb - 1.77·obj]  ──▶ 256D sent
       │
       ├── P8 ──▶ 256D sentence vector (direct from chars)
       │
       └── P9 pipeline: chars → 256D sent + 3×128D words (one-shot)
```

---

## Layer Benchmarks · 层级指标

| Layer | Task · 任务 | Input → Output | Metric | Score | Status |
|:-----:|-------------|----------------|--------|------:|:------:|
| **P1** | Char → Word · 字→词 | 2 chars → 128D word | Top-1 Accuracy | **99.89%** | ✓ |
| **P2** | Word → Char · 词→字 | 128D word → 2 chars | Cosine Similarity | **99.28%** | ✓ |
| **P3** | Attribute Binding · 属性绑定 | word → 7 categories | Classification | **100%** | ✓ |
| **P5** | Words → Sentence · 词→句 | n×128D → 256D | Order Gap | **1.459** | ✓ |
| **P6** | Sentence → Words · 句→词 | 256D → n×128D | Cosine Similarity | 86.68%* | ~ |
| **P7** | Cross-Sentence · 跨句路由 | A-words → B-sent | Cosine Similarity | **~99.98%** | ✓ |
| **P8** | Chars → Sentence · 字→句 | char seq → 256D | Cosine Similarity | **99.98%** | ✓ |
| **P9** | Unified Pipeline · 统合管线 | chars → sent+words | End-to-end | — | ✓ |

*\*P6 reaches ~99% with P8→P6 bridge training (see `P6_sent_word/train_bridge.py`)*

**Vocabulary:** 1,800 Chinese words, 734 unique characters  
**Training data:** 100 template sentences (subject-verb-object)  
**Hardware:** NVIDIA RTX 5070 12GB, GPU memory ~75MB, CUDA 12.8, PyTorch 2.12  
**Total training time:** ~30 minutes for all layers

---

## Technical Highlights · 技术要点

### 1. Uniform 128D Representation · 统一维度

All vectors across all 9 layers share the same 128-dimensional space. No projection layers, no dimension bottlenecks. This eliminates the "information refraction" that occurs when data crosses dimension boundaries through linear projections.

```
Legacy (asymmetric):  120D internal → Linear(120→24) → 32D output  ✗ lossy
V18 v3.0 (uniform):   120D semantic + 8D positional = 128D end-to-end  ✓ lossless
```

### 2. 64-Head Cross-Attention · 多头交叉注意力

Every layer uses multi-head cross-attention where Q encodes the querying entity and K,V encode the reference space:

```
Q = cat(char₁, char₂) @ W_q  →  [batch, 64 heads, 4D]
K = word_table @ W_k          →  [vocab, 64 heads, 4D]
V = word_table @ W_v          →  [vocab, 64 heads, 4D]

attn = softmax(Q·Kᵀ / √4)     →  [batch, 64, 1, vocab]
output = attn·V · W_o          →  [batch, 128D]
```

### 3. White-Box Modulation Chain · 白箱调制链

Two learnable modulation zones operate on every forward pass:

- **Exploration Zone** (`探索区`): Loss-driven basis vectors (pos_basis / neg_basis) scaled by an MLP that reads the current loss value. High loss → strong exploration modulation; low loss → modulation decays.
- **Meta-Learning Zone** (`元学习区`): A 2-layer Tanh network that transforms exploration signals into structured weight-space perturbations, preventing representation collapse.

```
last_loss ──▶ [MLP] ──▶ pos_basis·α + neg_basis·β  ──▶ [Meta: Tanh→Tanh] ──▶ +inject into output
```

### 4. Position-Semantic Decomposition · 位置语义解耦

Every character/word vector decomposes into:

- **8D position** (sin/cos encoding, frozen): Encodes character position within a word (char₁ vs char₂)
- **120D semantic** (learned embedding): Encodes the meaning of the character

This white-box split allows the position component to be directly inspected and the semantic component to be compared via cosine similarity.

### 5. Fractal Recursion · 分形递归

P1's word-composition logic (position-weighted fusion + cross-attention) is reused identically in P5 for sentence composition. The same architecture that composes 2 characters into a word also composes N words into a sentence — just with different parameters.

### 6. Seven-Attribute Grammar · 七属性语法体系

P3 independently binds each word to 7 Chinese grammatical roles using separate cross-attention models:

| Attribute · 属性 | English | Family Size | Accuracy |
|:---:|---------|:-----------:|:--------:|
| 主语 | Subject | 269 | 100% |
| 谓语 | Predicate | 207 | 100% |
| 宾语 | Object | 305 | 100% |
| 定语 | Adjective | 13 | 100% |
| 状语 | Adverbial | 10 | 100% |
| 补语 | Complement | 37 | 100% |
| 虚词 | Function word | 7 | 100% |

---

## Quick Start · 快速开始

### Requirements

```bash
conda create -n v18 python=3.10
conda activate v18
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Train All Layers

```bash
cd P1_char_word && python train.py        # P1: Char → Word
cd ../P2_word_char && python train.py     # P2: Word → Char
cd ../P3_word_attr && python train_batch.py  # P3: 7-Attribute Binding
cd ../P5_sentence && python train.py      # P5: Word Seq → Sentence
cd ../P6_sent_word && python train.py     # P6: Sentence → Word Seq
cd ../P8_char_sent && python train.py     # P8: Char Seq → Sentence
```

### End-to-End Test

```bash
python pipeline_test.py     # Full pipeline: chars → sent → words → attributes
python segment_test.py      # Word segmentation stress test (zero-training inference)
```

---

## Project Structure · 目录结构

```
V18-cognitive-architecture/
├── utils/config.py              # Global config (128D, 64 heads, all hyperparams)
├── data/word_list.txt           # 1,800 Chinese word training data
├── cards/                       # Architecture specification cards
├── P1_char_word/                # P1: Char → Word (model + train)
├── P2_word_char/                # P2: Word → Char decoder
├── P3_word_attr/                # P3: 7-attribute binding
├── P3_data/                     # Attribute word lists (7 categories)
├── P5_sentence/                 # P5: Word sequence → Sentence
├── P6_sent_word/                # P6: Sentence → Word sequence decoder
├── P7_cross_sent/               # P7: Cross-sentence routing
├── P8_char_sent/                # P8: Character sequence → Sentence
├── P9_sent_char/                # P9: Sentence → Character sequence decoder
├── P9_pipeline/                 # P9: Unified pipeline (chars→sent+words)
├── pipeline_test.py             # End-to-end pipeline test
├── segment_test.py              # Word segmentation stress test
└── V18_2026-06-03_开发日志.txt   # Full development log (Chinese)
```

---

## Key Design Decisions · 关键设计取舍

1. **No projection layers** — The v2.0 asymmetric design (120D→24D projection) was scrapped after experiments showed it was the accuracy bottleneck. v3.0 uses uniform 128D throughout.

2. **Early stopping over long training** — P1 peaks at 99.89% (epoch 25) but collapses to 36% by epoch 500. All checkpoints are saved at peak performance. See the dev log for detailed training dynamics.

3. **Independent P3 embeddings** — Each grammatical attribute has its own P3 embedding table rather than sharing a single multi-label classifier. This avoids the word-category ambiguity problem (e.g., "学习" can be both subject and verb).

4. **Contrastive training for P5** — Training on correct-order sentences alone causes representation collapse (deterministic output). Contrastive loss with scrambled word order forces the model to learn order-sensitive representations.

---

## Development Log · 开发日志

See [V18_2026-06-03_开发日志.txt](V18_2026-06-03_开发日志.txt) for the complete 8-hour development timeline, including all failures, fixes, scaling experiments, and the dimension bottleneck discovery.
