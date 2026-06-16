# 16 Days, 4.7M Params, Zero Black Boxes: Building a White-box Chinese Cognition Engine from Scratch

**Author: Wei Jinqi | June 16, 2026**

---

Every time I use a large language model, the same thought nags at me: *I have no idea what's happening inside.*

95% accuracy? Great. But which weights fired? What linguistic features were extracted? Did it confuse "bank" (river) with "bank" (financial)? Nobody knows.

So I spent 16 days building a Chinese language engine where **every weight has a reason and every decision is traceable**.

## The Core Idea

Instead of training a transformer on terabytes of text and hoping it learns Chinese, I designed each module to handle a **specific linguistic function**:

| Module | Function | Params |
|--------|----------|--------|
| P1 | Char → Word encoding | 96K (frozen) |
| P3-L | Multi-dimensional attribute annotation | 0 (rule engine) |
| P7 | Cross-sentence word routing | 226K |
| Explore+Meta | Learned gating over decode dims | 101K |
| P6 | Sentence → Word sequence decoding | 4.37M |

The modules are chained: **P1 encodes → P7 routes → Gate modulates → P6 decodes**. Every intermediate state can be inspected.

## Day-by-Day: The Good, The Bad, and The Mode Collapse

### Days 1-2: Laying Foundations (and Fighting Collapse)

Day 1 was smooth. P1 (char→word encoder) and P3 (attribute stack — a rule engine that tags words with person/syntax/semantic/emotion/direction attributes) came together quickly.

Day 2 introduced P7, the cross-sentence router. And **everything broke**.

I used standard multi-head cross-attention. Every position — regardless of input — routed to the same output word. The dreaded **Mode Collapse**.

What followed was seven failed fixes:
- **v2**: Diversity loss → still collapsed
- **v3**: Grouped loss → partially better
- **v4**: Temperature scaling → not enough
- **v5**: Contrastive learning → oscillated wildly
- **v6**: Gating mechanism → unstable
- **v7**: Hierarchical modulation → almost converged

The breakthrough came when I noticed Q/K were eye-initialized, meaning each head saw only 1 dimension with zero discrimination power.

**v8 (final)**: Xavier init for Q/K, eye init for V. Added an Explore network (loss → GELU MLP → 64D control signal) and a Meta network (signal + state → per-word gate). Mode collapse solved.

### Day 3-4: Gating Innovation and the Repetition Monster

Day 3 built P3-L: 23 groups, 312 independent attention heads, each controlling one attribute dimension. Combined training with P7 via UnifiedExplore→UnifiedMeta gate.

Day 4 introduced **P6: the sentence→word decoder**. It was supposed to take a 256D sentence vector and output 16 distinct word embeddings.

It output the same word 16 times. The **Repetition Collapse** had begun.

Six versions over two days:
- **V1**: 16 parallel heads → all output same word
- **V2**: Serial residual extraction → gradient breakage
- **V3**: Remove detach, add damping → gradient entanglement
- **V4**: Weight transpose inverse projection → too aggressive
- **V5**: Orthogonal init, detach, 0.8 damping → too heavy, heads dead
- **V6**: Position embedding — `h + pos_embed[i]` per head → **solved**

The simplest fix won. Each head receives the same `h` but adds a unique learned position embedding. No rep_pen. No residuals. No detach. Just position diversity.

### Day 5: The Gate That Stopped Learning

Epoch after epoch, the gate stayed frozen — all 256 dimensions had **std=0.0001**. Three bugs conspired:
1. `explore_mod.weight` zero-initialized → identical signal per dim
2. `p3l_act` zero-initialized → sigmoid(0)=0.5 for all dims
3. `bias init scale=0.1` too small → output stuck at 0.5

Then I found an even worse bug: `gate.item()` was used in loss computation, converting a tensor to Python float — **severing the gradient chain**. The gate had been frozen for **240 epochs** without anyone noticing.

Fix: keep gate as tensor, let gradients flow back through explore and meta. Loss dropped from 0.56 to 0.28 in 3 epochs.

### Day 6: The AI That Debates Itself

I built a dual-agent debugging system: **DeepSeek (engineer)** proposes fixes, **Qwen (reviewer)** audits them. They debate until convergence.

The system diagnosed four major bugs, including the gradient chain break. It would have saved days if I'd built it earlier.

### Days 7-11: From 875K to 4.7M — Scaling Up

Key improvements:
- Replaced mean pooling with **P5-style ±superposition** for sentence vectors
- Expanded P6 from 16 heads to **128 independent heads**
- Built **Context Cache System**: 3-tier (GPU/RAM/Disk), adaptive retrieval window, drift detection
- First benchmark: **92.4% word accuracy** on 875K-param V18

### Days 12-16: CUDA Wars and Open Data

- Conquered CUDA OOM (P1 full attention → batch encoding)
- Fixed space-character collapse (HF data had spaces between Chinese chars → `ord(c) > 32` filter)
- Assembled 52K public training pairs from HuggingFace + MuCGEC
- Launched V19 1000-epoch training: **4.7M params, 141MB GPU, 100% public data**

## Seven Bugs That Almost Won

| Bug | Symptom | Root Cause | Fix |
|-----|---------|------------|-----|
| Mode Collapse | All outputs = same word | Q/K eye-init, zero discrimination | Xavier init + diversity architecture |
| Gate Symmetry Lock | All gate dims identical (std=0.0001) | Three zero-initializations | Proper random init for explore, act, bias |
| Gradient Chain Break | Gate not learning for 240 epochs | `.item()` severed gradient | Keep as tensor |
| Repetition Collapse | 16 heads → same word | Parallel heads share identical input | Position embedding V6 |
| CUDA OOM | 25.76 GiB allocated | P1 full cross-attention | Batch encoding (50 words) |
| Space Collapse | Model outputs spaces | HF data formatting | `ord(c) > 32` filter |
| sent_vec Info Loss | Different sentences → similar vectors | Mean pooling | Learnable ±weighted sum |

## Results

### V18 (875K params)

| Metric | Score |
|--------|-------|
| Word Accuracy | **92.4%** |
| Exact Match | 76.3% |
| Rouge-L F1 | 93.2 |
| Per-word Cosine | 0.96 |
| Speed | 14ms/sent (71 sent/s) |

### V19 (4.7M params, training in progress)

Epoch 1 (from scratch, no pretraining): **43.5%** word accuracy on held-out exam set. Target: >95% after 1000 epochs.

## Why This Matters

LLMs are powerful but opaque. When GPT makes a mistake, you can't trace which neurons fired wrong. With V19, you can:
- See exactly which word attributes were used
- Trace which input words influenced each output
- Inspect why the gate opened or closed each dimension
- Debug layer by layer, like stepping through code

This isn't about beating GPT. It's about building something **you can understand completely**.

## What's Next

- Context system "write path": feed retrieved context into P7 to enable multi-turn dialogue
- Expand to 512-dim if quality plateaus (currently 128D)
- Multi-language extension of the attribute stack (P3-L)
- Open-source community contributions

## Try It

```bash
git clone https://github.com/Xuan-yi-yan/V18-cognitive-architecture
cd V18-cognitive-architecture
python download_public_data.py
python train_v19_full.py --data public --epochs 1000 --display 10
```

Full model card and architecture docs on [Hugging Face](https://huggingface.co/).

---

*16 days. 7 dead bugs. 4.7 million parameters. Zero black boxes.*

*That's just how I like it.*
