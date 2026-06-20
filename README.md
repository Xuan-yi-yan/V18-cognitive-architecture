# V19 White-box Chinese Cognition Engine

**A fully transparent, from-scratch Chinese language understanding system.**

[![Model](https://img.shields.io/badge/HuggingFace-V19_Model-blue)](https://huggingface.co/MIHUJIOUY/V19-cognitive-engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
![Params](https://img.shields.io/badge/params-4.7M-green)
![GPU](https://img.shields.io/badge/GPU-141MB-orange)

> Built in 16 days. No black boxes. No pretrained embeddings. Every weight has a reason.

---

## What is this?

V19 is a white-box Chinese cognition engine. Unlike LLMs that hide reasoning behind billions of opaque parameters, V19 makes every linguistic decision **traceable and auditable**:

- You can inspect **which attributes** were assigned to each word
- You can see **which words routed to which sentence positions**
- The gate system tells you **which decoding dimensions are active**
- No GPU needed for inference — runs on CPU at 71 sentences/second

## Architecture

```
Input Sentence (A)
    │
    ▼
┌─────────────────────────────────┐
│  P1  Char→Word Encoder  [96K]   │  Frozen. Cross-attention over 6000-word table.
│      2-char → 128D vector       │  Batch encoding, ~141MB GPU.
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  P7  Cross-sentence Router      │  32 heads × 4D. P5-style ±superposition.
│      [226K, 128→256 sent_vec]   │  Learnable positional weights instead of mean pooling.
└─────────────────────────────────┘
    │
    ├── word_out[nA, 128]  (diagnostics)
    └── sent_vec[256]      (to context cache)
    │
    ▼
┌─────────────────────────────────┐
│  Explore + Meta  [101K]         │  12D loss → 256D signal → sigmoid gate
│      Gate control over P6       │  Learns when to open/close decode dimensions
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  P6  Sent→Word Decoder  [4.37M] │  128 independent heads × Linear(256,128)
│      Position Embedding V6      │  h + pos_embed[i] per head — naturally anti-collapse
└─────────────────────────────────┘
    │
    ▼
Output Words (B)
```

## Key Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Position embedding (V6) | Each head gets `h + pos_embed[i]` — unique starting point prevents mode collapse without rep_pen |
| 2 | Explore→Meta gate | Loss flows through explore network into meta gate; gate learns *when* to modulate, not just *how much* |
| 3 | P5-style ± superposition | `word_out * learnable_pos_weights → sum → sent_proj` preserves per-word information (vs. mean pooling) |
| 4 | 128-dim (not 512) | Qwen identified quadratic explosion trap; 128D is sufficient for Chinese characters |
| 5 | Strict 3-way split | 80/10/10 train/test/exam; exam set locked on disk until final eval |
| 6 | Public data only | HuggingFace (53K) + MuCGEC (1K); fully reproducible, no proprietary data |

## Benchmarks

### V18 (875K params, sentence reconstruction mode)
P7 uses B-sentence as K/V — this measures **sentence reconstruction quality**, not A→B translation. Cosine similarity evaluation.

| Metric | Score |
|--------|-------|
| Word Accuracy | **92.4%** |
| Exact Match | 76.3% |
| Rouge-L F1 | 93.2 |
| Inference Speed | 14ms/sent (71 sent/s) |

### V20 (5.22M params, translation mode)
P7 uses full vocabulary as K/V — true A→B translation. Cross-entropy direct character output. Strictly held-out exam set.

| Metric | Status |
|--------|--------|
| Architecture Verified | 49-pair test: 9/9 correct |
| One-shot Proof | 13-char sentence: 100% in 50 epochs |
| Full Training | 52K data, 1000 epochs in progress |
| V18 Score Context | 92.4% = reconstruction, not translation |

## Bugs We Slain (the hard way)

1. **Mode Collapse** — Every position output the same word. Fixed with gating + diversity loss (P7 v1→v8, 8 iterations)
2. **Gate Symmetry Lock** — All 256 gate dims had std=0.0001. Fixed by correcting zero-initialization in explore_net, act, and bias
3. **Gradient Chain Break** — `.item()` in loss severed the explore→meta gradient. 240 epochs wasted before discovery
4. **Repetition Collapse** — P6 decoded same high-frequency word 16 times. Fixed with Position Embedding V6 — the simplest solution after 5 failed approaches
5. **CUDA OOM** — P1 full cross-attention over 6000 words → 25.76GiB. Fixed with 50-word batching + `torch.no_grad()`
6. **Space-character Collapse** — HF data had spaces between chars; model learned to output spaces. Fixed with `ord(c) > 32` filter
7. **sent_vec Information Loss** — Mean pooling smoothed away per-word differences. Fixed with learnable ±weighted sum

## Training Data

| Source | Pairs | Domain |
|--------|-------|--------|
| HuggingFace `shibing624/chinese_text_correction` | 53K | News, legal, medical, automotive |
| MuCGEC (NAACL 2022) | 1K | Academic Chinese correction |

- **After sentence splitting**: 52,387 pairs
- **Train**: 41,909 | **Test**: 5,238 | **Exam**: 5,240 (locked)
- **Vocabulary**: 6,164 unique characters
- **Average sentence**: 20–80 characters

## Quick Start

```bash
# Download public data
python download_public_data.py

# Train V19 (1000 epochs, ~17 days on RTX 5070)
python train_v19_full.py --data public --epochs 1000 --display 10 --lr 0.003

# Evaluate
python eval_benchmark.py --data 5k --max_samples 1000

# Test context cache
python test_context_cache.py
```

## File Structure

```
C:/ai/
├── P1_char_word/          # Char→Word encoder (frozen)
├── P6_sent_word/          # Sent→Word decoder (position embedding V6)
├── P7_cross_sent/         # Cross-sentence router
├── train_v19_full.py      # Main training script
├── download_public_data.py
├── eval_benchmark.py
├── test_context_cache.py
├── tool_agent.py           # Dual-agent debate system
├── test_full_chain_v2.py
└── utils/config.py
```

## Author

**Wei Jinqi (卫锦旗)**

## Citation

```bibtex
@misc{wei2026v19,
  title={V19: A White-box Chinese Cognition Engine},
  author={Wei, Jinqi},
  year={2026},
  howpublished={\url{https://github.com/Xuan-yi-yan/V18-cognitive-architecture}},
}
```

## License

MIT — see [LICENSE](./LICENSE)
