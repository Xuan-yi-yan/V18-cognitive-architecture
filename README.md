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

### v8.0 Final (tag `v8.0`, branch `master`)

| Layer | Task | Metric | Solo Best | P11 Joint |
|:-----:|------|--------|:---------:|:---------:|
| **P1** | Char → Word | Top-1 | **98.60%** | 97.60% |
| **P2** | Word → Char | Cosine | 80.70% | **95.95%** |
| **P3** | Attribute Binding | Accuracy | **100%** | 100% |
| **P5** | Words → Sentence | Gap | **1.39** | — |
| **P6** | Sentence → Words (Bridge) | Cosine | 78.19% | **96.31%** |
| **P7** | Cross-Sentence | Cosine | **97.90%** | — |

**P11 is the key innovation**: Joint training of P1's 2048→128D projection with P2 feedback. This unlocks P2 from 80.7% → 95.95% and Bridge from 78.2% → 96.3% — the projection reversibility bottleneck is solved.

### Key Architecture Features

- **Asymmetric**: P1: 2048D/512 heads internal → 128D output (16:1 dim ratio, 8:1 head ratio)
- **Staged projection**: \*/\ (multiplicative gate 2048→1024) → +- (additive fusion 1024→128)
- **Einsum attention**: Zero-expand cross-attention, GPU peak 5GB at batch=64
- **All layers have independent explore/meta zones** (Tanh removed, no scaling)
- **Multi-objective independent-target loss** (each sub-goal has its own target)

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
- **Meta-Learning Zone**: learnable modulation network. In v8.0, the final Tanh clamp was removed to preserve modulation magnitude.

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
git clone https://github.com/Xuan-yi-yan/V18-cognitive-architecture.git
cd V18-cognitive-architecture && git checkout v8.0
conda create -n v18 python=3.10 && conda activate v18
pip install -r requirements.txt
```

GPU: RTX 5070 12GB recommended. CPU works but slow. Checkpoints NOT in repo — generated by training.

### Step 1: Baseline Pipeline (~25 min)

```bash
python expand_data_v2.py              # 6000 words + 2000 sentences
python train_all.py --seed 789        # Train P1→P2→P3→P5→Bridge→P7
```

Expected: P1=98.6%, P2≈89%, P3=100%, P5 gap≈1.4, Bridge≈78%, P7≈98%

### Step 2: P11 Joint Optimization (~15 min)

```bash
python p11_joint_train.py --epochs 200 --display 10
```

**Pushes P2 from ~89% → ~96%** by jointly training P1's projection with P2 feedback.

### Step 3: Full Pipeline with P11 (~30 min)

```bash
python fullpipe_p11.py
```

Bridge reaches **~96%** with the optimized P11 projection.

### Quick Diagnostics

```bash
python seed_sweep.py         # P1+P2 seed screening, 3 seeds ~20 min
python p2_diagnose.py        # P2 deep diagnosis, 100 epochs log every 5
```

### Expected Results

| Step | Time | Key Metric |
|------|:----:|------------|
| train_all.py | ~25 min | P1=98.6%, P2=89%, Bridge=78% |
| p11_joint_train.py | ~15 min | P2=95.95% |
| fullpipe_p11.py | ~30 min | Bridge=96.3% |

GPU: ~300MB normal, 5GB peak. P1=27.8M params, total ~30M.

---

## Project Structure

```
├── utils/config.py              # Global config (asymmetric dims, 512/64 heads)
├── data/word_list_v2.txt        # 6,000 words (jieba frequency dict)
├── P1_char_word/                # P1: 2048D encoder + StagedProjection → 128D
├── P2_word_char/                # P2: 128D word → char decoder
├── P3_word_attr/                # P3: 7-attribute binding
├── P5_sentence/                 # P5: Words → Sentence
├── P6_sent_word/                # P6: Sentence → Words decoder
├── P7_cross_sent/               # P7: Cross-sentence routing
├── P8_char_sent/                # P8: Chars → Sentence
├── P9_sent_char/                # P9: Sentence → Chars
├── p11_joint_train.py           # P11: P1_proj + P2 joint optimization
├── p12_full_joint.py            # P12: P1_proj + P2 + Bridge 3-way joint
├── fullpipe_p11.py              # Full pipeline with P11 projection
├── train_all.py                 # Sequential P1→P2→P3→P5→Bridge→P7
├── expand_data_v2.py            # jieba-based data generation
├── full_pipeline_eval.py        # End-to-end Top-1/3/5/10 eval
└── requirements.txt             # torch, jieba
```

---

## Current Limitations

| Limitation | Root Cause | Mitigation |
|-----------|------------|------------|
| Projection reversibility | 2048→128D information loss | P11 joint training (solved: 80.7→96%) |
| Sentence diversity | 8000 SVO templates | Real Chinese corpus |
| Modulation decay | Shared LR across zones | Independent LR per zone (planned) |

**Architecture validated. P11 proves the 2048→128D projection is optimizable for downstream friendliness.**

---

## Version History

| Version | Key Change | P1 | P2 | Bridge |
|---------|------------|:---:|:---:|:---:|
| v5.1 | Asymmetric Staged baseline | 98.60% | 88.57% | 72.29% |
| v7.1 | P2 explore/meta + P5 fix | 98.60% | 89.25% | 78.19% |
| **v8.0** | **P11 joint (projection reversibility)** | **97.60%** | **95.95%** | **96.31%** |

---

## Development Log

[V18_2026-06-03_开发日志.txt](V18_2026-06-03_开发日志.txt) — complete history, all experiments, failures, and fixes.
