# V18 Cognitive Architecture

**Asymmetric white-box neural pipeline for Chinese language understanding.**

2048D internal encoding (512-head cross-attention) → staged projection to 128D → 9-layer bidirectional pipeline from characters to sentences.

---

## Architecture Diagram

```
                        ┌──────────────────────────────────────────┐
                        │           V18 ASYMMETRIC CORE            │
                        │   P1: 2048D + 512 heads (high-D encode) │
                        │   ↓ StagedProjection: 2048→1024→128     │
                        │   P2-P9: 128D + 64 heads (low-D decode) │
                        └──────────────────────────────────────────┘

    CHAR LEVEL                   WORD LEVEL              SENTENCE LEVEL
    ─────────                    ──────────              ─────────────

    ┌──────────┐                ┌──────────┐            ┌──────────┐
    │  c1 c2   │──── P1 ───────▶│ w_vec    │─── P5 ────▶│ s_vec    │─── P7 ──▶ cross-sent
    │ (2048D)  │◀─── P2 ───────│ (128D)   │◀── P6 ─────│ (256D)   │           routing
    └────┬─────┘                └────┬─────┘            └────┬─────┘
         │                           │                       │
         │         ┌─────────────────┤                       │
         │         │   P3: 7-attr    │                       │
         │         │   binding       │                       │
         │         │ subj/verb/obj   │                       │
         │         │ adj/adv/comp    │                       │
         │         │    func         │                       │
         │         └─────────────────┘                       │
         │                                                  │
         └─────────────── P8 ──────────────────────────────▶│
         ◀───────────────────── P9 ─────────────────────────┘

    P1↔P2: char↔word roundtrip    P5↔P6: word↔sentence roundtrip
    P8↔P9: char↔sentence roundtrip  P7: cross-sentence routing
```

### Data Flow Example

```
  "学生学习知识"
       │
       ▼
  [学, 习, 生, 学, 习, 知, 识]              ← 7 raw characters
       │
       ├── P1(2048D) ──▶ [学生, 学习, 知识]  ← 3 word vectors (128D)
       │                      │
       │                      ├── P3 ──▶ subj=学生, verb=学习, obj=知识
       │                      │
       │                      └── P5 ──▶ 0.88·subj + 0.96·verb - 1.00·obj → 256D sent
       │
       ├── P8 ──▶ 256D sentence vector (direct from chars, cos=0.99 vs P5)
       │
       └── P9: chars → 256D sent + 3×128D words (one-shot)
```

---

## Layer Benchmarks

### Current Stable (v7.1-stable, tag: `v7.1-stable`)

| Layer | Task | Input → Output | Metric | Score |
|:-----:|------|----------------|--------|------:|
| **P1** | Char → Word | 2 chars → 128D word | Top-1 (6000 words) | **98.60%** |
| **P2** | Word → Char | 128D word → 2 chars | Cosine Similarity | **89.25%** |
| **P3** | Attribute Binding | word → 7 categories | Classification | **100%** |
| **P5** | Words → Sentence | n×128D → 256D | Order Gap | **1.39** |
| **P6** | Sentence → Words | 256D → n×128D | Cosine (bridge) | **78.19%** |
| **P7** | Cross-Sentence | A-words → B-sent | Cosine Similarity | **97.90%** |
| **P8** | Chars → Sentence | char seq → 256D | Cosine Similarity | ~99% |

### Bridge Experiments (branch: `bridge-fix`)

Multi-objective independent-target loss, 8000 sentences, 50 epochs:

| Epoch | word_cos | wc>.8 | wc>.9 | sent_self |
|:-----:|:--------:|:-----:|:-----:|:---------:|
| 1 | 79.62% | 46% | 0% | 0.995 |
| 10 | 82.47% | 80% | 0% | 0.994 |
| 20 | **83.76%** | **94%** | 0% | 0.991 |
| 30 | 82.55% | 84% | 3% | 0.992 |
| 40 | 83.08% | 85% | 1% | 0.991 |

**Best: 83.76% @ epoch 20** (up from 72.29% on main branch)

Data scaling results (simple sl+wl loss, 10 epochs):
- 2000 sentences: 74.7% → declining
- 4000 sentences: 78.6% → 80.2%
- 6000 sentences: 79.5% → 81.2% ← data sweet spot
- 8000 sentences: 78.7% → 79.9%

Multi-objective loss formula:
```python
sent_align = (1 - sent_cos)²      # P8→P5 alignment, weight 5.0
word_align = (1 - word_cos)²      # P6→target decoding, weight 1.0
sent_div   = relu(sent_self-0.85)²  # P8 diversity, weight 0.1
explore    = relu(0.02-explore_norm)² # explore preservation, weight 0.05
total = Σ(weight × objective)
```

### P3 Seven-Attribute Grammar

| Attribute | English | Family Size | v5.1 Accuracy |
|:---:|---------|:-----------:|:------------:|
| 主语 | Subject | 269 | 100% |
| 谓语 | Predicate | 207 | 100% |
| 宾语 | Object | 305 | 100% |
| 定语 | Adjective | 13 | 100% |
| 状语 | Adverbial | 10 | 100% |
| 补语 | Complement | 37 | 100% |
| 虚词 | Function word | 7 | 100% |

---

## Technical Highlights

### 1. Asymmetric Architecture (Core Innovation)

P1 internally encodes at 2048D with 512 attention heads for rich word discrimination, then projects to 128D via a staged projection for downstream compatibility. Encoder:decoder ratio = 16:1 (dims) and 8:1 (heads).

### 2. Staged Projection (v5.1, Best)

Two-stage dimension reduction that preserves decoder-navigable structure:
```
Stage 1 (*/gate): 2048D → sigmoid gate → 1024D (selective signal preservation)
Stage 2 (+-fuse): 1024D → path_a + path_b → 128D (additive feature fusion)
```

### 3. 512-Head Einsum Attention

Memory-efficient cross-attention using einsum instead of tensor expansion:
```
einsum('bhd,hnd->bhn', q, k)  — no [batch, heads, vocab, dim] materialization
GPU peak: 5GB for full 6000-word vocabulary with batch=64
```

### 4. White-Box Modulation Chain

Two learnable zones modulate every forward pass:
- **Exploration Zone**: Loss-driven basis vectors scaled by current training loss
- **Meta-Learning Zone**: 2-layer Tanh network for structured perturbation injection

### 5. Position-Semantic Decomposition

Every vector decomposes into 8D position (frozen sin/cos) + 120D semantic (learned), enabling direct inspection and cosine comparison.

### 6. Fractal Recursion

P1's word-composition logic (position-weighted fusion + cross-attention) is reused identically in P5 for sentence composition.

---

## Key Design Decisions

1. **Asymmetric over uniform**: Full 2048D destroys decoder performance. Asymmetric 2048D→128D preserves encoder quality.
2. **Staged over linear**: Single Linear(2048→128) loses 5% on P2 vs staged projection.
3. **Linear over nonlinear**: GELU/MLP projections degraded downstream. Sigmoid gating is the sweet spot.
4. **All layers have independent explore/meta zones**: P2 was missing — now fixed.
5. **P5 diversity regularization**: Prevents sentence vector collapse during contrastive training.
6. **F.normalize on P2 output**: Prevents norm collapse under cosine loss.

---

## Reproduce · 复现指南

### Requirements

```bash
conda create -n v18 python=3.10 && conda activate v18
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install jieba
git clone https://github.com/Xuan-yi-yan/V18-cognitive-architecture.git && cd V18-cognitive-architecture
```

### Stable Baseline (main branch, tag v7.1-stable)

```bash
git checkout main
python expand_data_v2.py                    # 6000 words + 2000 sentences
python train_all.py --seed 789              # P1→P2→P3→P5→P8+P6→P7 (~25 min)
python full_pipeline_eval.py                # End-to-end Top-1/3/5/10
```

### Bridge Experiments (bridge-fix branch)

```bash
git checkout bridge-fix
python seed_sweep.py                        # P1+P2 seed screening (3 seeds, ~20 min)
python p2_diagnose.py                       # P2 deep diagnosis (100 epochs, log every 5)
python bridge_diagnose.py                   # Bridge with joint fine-tuning
python bridge_long_train.py                 # Bridge 50-epoch long training, multi-obj loss
```

### Expected Results (seed 789, RTX 5070 12GB)

| Script | When | Key Metric |
|--------|:----:|------------|
| `train_all.py --seed 789` | after ~25 min | P1=98.6%, P2=89.3%, P3=100%, P5=1.39, Bridge=78.2%, P7=97.9% |
| `seed_sweep.py` | after ~20 min | 2/3 seeds P2>=90% |
| `bridge_long_train.py` | after ~60 min | Bridge word_cos peak ~83.8% @ epoch 20 |

**GPU memory**: ~1.1GB (train_all), peak ~5GB | **Model params**: P1=27.8M, P2=181K, total ~30M

---

## Project Structure

```
V18-cognitive-architecture/
├── utils/config.py               # Global config (asymmetric dims, heads, training)
├── data/word_list_v2.txt         # 6,000 Chinese words (jieba frequency dict)
├── P1_char_word/model.py         # P1: 2048D encoder + StagedProjection → 128D
├── P2_word_char/model.py         # P2: 128D word → char decoder
├── P3_word_attr/model.py         # P3: 7-attribute binding (einsum attention)
├── P3_data/                      # Attribute word lists (7 categories)
├── P5_sentence/model.py          # P5: Word sequence → Sentence synthesis
├── P6_sent_word/model.py         # P6: Sentence → Word sequence decoder
├── P7_cross_sent/model.py        # P7: Cross-sentence routing
├── P8_char_sent/model.py         # P8: Character sequence → Sentence
├── P9_sent_char/model.py         # P9: Sentence → Character sequence decoder
├── P9_pipeline/model.py          # P9: Unified pipeline
├── train_all.py                  # One-click full pipeline training
├── train_all_v4.py               # 2048D experiment variant
├── expand_data.py                # v1 data expansion
├── expand_data_v2.py             # v2: jieba-based data expansion
├── full_pipeline_eval.py         # End-to-end evaluation (Top-1/3/5/10)
├── pipeline_test.py              # Pipeline integration test
├── segment_test.py               # Word segmentation stress test
├── README.md
└── V18_2026-06-03_开发日志.txt    # Full development log (Chinese)
```

---

## Current Limitations

| Limitation | Current | Root Cause | Mitigation |
|-----------|:-------:|------------|------------|
| Bridge word retrieval | 72.29% | P6 decoding in 128D space, 2000 sentences | Scale to real corpus |
| P2 char reconstruction | 88.57% | Projection bottleneck for reverse mapping | Multi-task training |
| Sentence diversity | 2000 SVO templates | Auto-generated, no complex grammar | Real Chinese corpus |
| Modulation decay | α/β→0 late training | Shared learning rate | Independent LR per zone |
| End-to-end Top-1 | P6 limited | Bridge retrieval precision | P6 contrastive loss |

**Architecture validated. P1 at 2048D scales to 6000 words. The bottleneck is data diversity, not design.**

---

## Version History

| Version | Date | Key Change | P1 | P2 | Bridge | P7 |
|---------|------|------------|:---:|:---:|:---:|:---:|
| v3.0 | 06-04 | 128D uniform | 99.89% | 99.19% | 96.19% | 85.75% |
| v4.0 | 06-04 | 2048D uniform | 98.38% | 81.04% | 71.61% | 99.16% |
| v5.0 | 06-04 | Asymmetric Linear | 98.65% | 83.57% | 67.43% | 93.49% |
| **v5.1** | **06-04** | **Asymmetric Staged** | **98.60%** | **88.57%** | **72.29%** | **96.90%** |
| v5.2 | 06-04 | Asymmetric MLP | 98.73% | 80.54% | 71.71% | 93.31% |

---

## Development Log

See [V18_2026-06-03_开发日志.txt](V18_2026-06-03_开发日志.txt) for the complete development history including all experiments, failures, and fixes.
