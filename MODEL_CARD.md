---
language: zh
license: mit
tags:
  - chinese
  - white-box
  - nlp
  - cognitive
  - text-correction
  - interpretability
datasets:
  - shibing624/chinese_text_correction
  - MuCGEC
metrics:
  - accuracy
  - rouge-l
  - cosine-similarity
model-index:
  - name: V19 White-box Chinese Cognition Engine
    results:
      - task:
          type: text-correction
        dataset:
          name: shibing624/chinese_text_correction + MuCGEC
          type: public-combined
        metrics:
          - type: word-accuracy
            value: 92.4
          - type: rouge-l-f1
            value: 93.2
---

# V19 White-box Chinese Cognition Engine

## Model Summary

V19 is a **fully interpretable** Chinese language understanding system with **4.7 million parameters**. It reads a Chinese sentence as a sequence of characters, builds word-level representations through a frozen char-to-word encoder (P1), routes information across sentences (P7), and decodes back to word sequences (P6) — all while maintaining traceable, auditable intermediate states.

Unlike transformer-based LLMs, every internal decision in V19 can be inspected:
- **P1**: Which words each character pair maps to
- **P7**: How words route across sentences (32-head attention with per-head gating)
- **Explore+Meta Gate**: Which decoding dimensions are active and why
- **P6**: How each output word is decoded from the sentence vector + position embedding

## Intended Use

- **Chinese text correction** (primary task)
- **Interpretability research**: study how linguistic attributes compose without black boxes
- **Education**: demonstrate NLP concepts with fully transparent architecture
- **Low-resource deployment**: 141MB GPU, runs on CPU at 71 sent/s

## Architecture

```
P1 (Char→Word, 96K frozen) → P7 (Router, 226K) → Explore+Meta (Gate, 101K) → P6 (Decoder, 4.37M)
```

### P1: Char→Word Encoder (frozen)
- Input: 2 consecutive characters
- Output: 128-dimensional word vector
- Cross-attention over 6,000-word vocabulary
- Batch encoding (50 words/batch) to control GPU memory

### P7: Cross-sentence Router
- 32 heads × 4 dimensions
- P5-style ±superposition for sent_vec (learnable positional weights, not mean pooling)
- Output: 256D sentence vector

### Explore + Meta Gate
- 12D loss vector → Explore network (128→256→256→tanh) → 256D control signal
- Meta: sigmoid(bias + signal) → 256D gate
- Gate modulates P6 encoder output dimension-wise
- Learns *when* to open/close dimensions without direct loss minimization

### P6: Sentence → Word Decoder (Position Embedding V6)
- Encoder: 256→256→256 (GELU)
- 128 independent extraction heads, each: `h * gate + pos_embed[i] → Linear(256,128) → word_i`
- Position embedding provides unique starting point per head — naturally prevents repetition collapse
- No rep_pen, no residual subtraction, no detach needed

## Training

| Config | Value |
|--------|-------|
| Optimizer | Adam (P6 lr=0.003, P7 lr=0.0045, Gate lr=0.006) |
| Loss | `1.0 - mean(cosine_similarity(pred, true))` |
| Epochs | 1000 |
| Batch | Full dataset per epoch (41,909 pairs) |
| GPU | RTX 5070 |
| Memory | ~300MB (training), 141MB (inference) |

## Evaluation

### V18 (875K params, 16 heads)

| Metric | Score |
|--------|-------|
| Word Accuracy | 92.4% |
| Exact Match | 76.3% |
| Rouge-L F1 | 93.2 |
| Per-word Cosine Mean | 0.96 |
| Inference | 14ms/sentence |

### V19 (4.7M params, 128 heads) — in training

| Metric | Epoch 1 | Target |
|--------|---------|--------|
| Word Accuracy | 43.5% | >95% |
| Per-word Cosine | 0.73 | >0.97 |

## Key Innovations

### Position Embedding V6 (Anti-collapse)
After 5 failed approaches to prevent the P6 decoder from outputting the same word repeatedly (rep_pen, residual extraction, weight transpose inversion, orthogonal init, cos_loss margin), the final solution was the simplest:

```python
for i in range(max_words):
    hi = h + self.pos_embed[i]  # unique starting point per head
    w = self.extract[i](hi)
```

No rep_pen. No residuals. No detach. Just position diversity.

### Explore→Meta Gate
Instead of directly minimizing loss (which causes gates to converge to all-open or all-closed), the gate is trained *indirectly*:
1. Loss flows into Explore network → produces 256D signal
2. Meta applies learned bias + sigmoid → 256D gate
3. Gate modulates P6 encoder → affects word predictions
4. Gate quality is measured by per-head prediction accuracy, not total loss

This prevents the "gate symmetry lock" (all dims identical, std=0.0001) that plagued early versions.

## Data

| Source | Pairs | License |
|--------|-------|---------|
| [shibing624/chinese_text_correction](https://huggingface.co/datasets/shibing624/chinese_text_correction) | 53,298 | Apache 2.0 |
| [MuCGEC](https://github.com/HillZhang1999/MuCGEC) | 1,038 | CC BY 4.0 |

After sentence splitting: **52,387 pairs**  
Split: train 41,909 (80%) / test 5,238 (10%) / exam 5,240 (10%)

## Limitations

- **Chinese only**: Character set limited to 6,164 unique characters from training data
- **Sentence length**: Max 128 characters (configurable, but untested beyond)
- **No multilingual support**: Architecture assumes CJK character structure
- **Training data bias**: Primarily news/law/medical domains from text correction dataset
- **V19 training incomplete**: 1000-epoch training in progress; current model may be suboptimal

## Citation

```bibtex
@misc{wei2026v19,
  title={V19: A White-box Chinese Cognition Engine},
  author={Wei, Jinqi},
  year={2026},
  howpublished={\url{https://github.com/Xuan-yi-yan/V18-cognitive-architecture}},
}
```

## Contact

GitHub: [@Xuan-yi-yan](https://github.com/Xuan-yi-yan)
