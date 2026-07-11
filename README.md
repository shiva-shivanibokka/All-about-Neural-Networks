# All About Neural Networks — Deep Learning From Scratch

> A 32-notebook curriculum that rebuilds modern deep learning from first principles — autograd to GPT to diffusion to six reproduced landmark papers — every piece coded from scratch and verified against PyTorch.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243)
![Notebooks](https://img.shields.io/badge/notebooks-32-success)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Recruiter TL;DR

- **What it is:** a from-scratch deep-learning curriculum — 32 executable Jupyter notebooks across 7 tiers — that rebuilds every core component of modern ML (reverse-mode autograd, CNNs, RNN/LSTMs, transformers/GPT, LLM fine-tuning & RLHF, VAEs/GANs/diffusion) in pure NumPy or bare PyTorch, then **verifies each implementation against PyTorch's reference**.
- **Hardest problem solved:** implementing the full transformer/LLM stack from scratch — reverse-mode autograd, scaled-dot-product & multi-head attention, a trainable GPT, BPE tokenization, KV-cache inference, FlashAttention's tiled/online-softmax algorithm, LoRA, and RLHF (reward model + PPO + DPO) — with hand-written backward passes matching PyTorch to machine precision (~1e-14).
- **Grounding:** every notebook runs end-to-end with built-in `assert`-based correctness checks; the from-scratch GPT trains on tiny-Shakespeare to **1.56 nats/char** (vs. 4.17 random baseline), and six landmark papers are reproduced with matching results.

---

## Overview & Motivation

Most people can *use* PyTorch. Far fewer can *rebuild* it — and rebuilding is the fastest way to actually understand it. This repository is my systematic, ground-up reconstruction of modern deep learning, built as preparation for research residencies (OpenAI / Apple and similar), where the bar is **implementing ideas from scratch and reproducing papers**, not calling library functions.

The guiding rule for every notebook: **build the novel mechanism from scratch (NumPy or bare PyTorch — no `nn.Transformer`, no `nn.MultiheadAttention`, no black boxes for the core idea), then verify it against PyTorch's reference implementation.** Autograd is built from a scalar `Value` class up to a full tensor engine; attention is written matmul-by-matmul; optimizers are re-derived and checked step-for-step against `torch.optim`. Where a training loop needs GPU speed, PyTorch's autograd does the *plumbing* — but the idea under study is always hand-built and cross-checked.

Each notebook follows the same teaching arc: **intuition & analogy → step-by-step math derivation → from-scratch code → experiment → "what to notice" → exercises → interview-style Q&A → cited references.** They are visualization-rich (computational graphs, attention heatmaps, loss surfaces, latent manifolds, denoising trajectories) and end with the specific papers each one implements.

---

## Skills Demonstrated

- **ML/DL fundamentals from first principles** — reverse-mode automatic differentiation, backpropagation, the multivariate chain rule, and hand-derived gradients for every common layer (Linear, Conv2d, BatchNorm, LayerNorm, GELU, softmax-cross-entropy).
- **Transformer & LLM internals** — scaled dot-product / multi-head / causal attention, positional encodings (sinusoidal & RoPE), GPT training, BPE tokenization, sampling/decoding strategies, KV-cache inference, FlashAttention.
- **LLM training pipeline** — pretraining → fine-tuning, LoRA / parameter-efficient fine-tuning, RLHF (Bradley-Terry reward modeling, PPO, DPO), instruction tuning, and LLM evaluation methodology.
- **Generative modeling** — VAEs (reparameterization, ELBO), GANs (adversarial training, mode collapse), diffusion/DDPM, normalizing flows — the four major generative families.
- **Scaling & systems** — mixed-precision numerics, data/tensor/pipeline/FSDP parallelism, scaling laws, inference optimization.
- **Numerical methods & verification** — gradient checking, numerical stability, and reference-implementation cross-validation.
- **Research paper reproduction** — a repeatable read → minimal-repro → run → ablate → report workflow, applied to six landmark papers.
- **Scientific communication** — derivations, honest result reporting (calibrated to measured numbers, no overclaiming), and interview-ready explanations.

---

## What's Inside — The Curriculum

Seven tiers, each building on the last. Every notebook is self-contained, executable, and cites the papers it implements.

### Tier 0 · Foundations — *the machinery under everything*
| # | Notebook | Builds from scratch |
|---|----------|---------------------|
| 01 | Autograd from scratch | Scalar reverse-mode autodiff (a `Value` class), topological sort, the chain rule; trains an MLP on its own engine |
| 02 | Tensors & tensor autograd | NumPy tensor autograd: broadcasting (`unbroadcast`), matmul chain rule, softmax cross-entropy; a 2-moons classifier |
| 03 | Backprop through classic layers | Hand-derived backward passes for Linear, ReLU/GELU, LayerNorm, BatchNorm, Conv2d (im2col) — all gradient-checked |
| 04 | Optimizers from scratch | SGD → Momentum → Nesterov → RMSProp → Adam → AdamW, visualized on loss surfaces, matched to `torch.optim` |
| 05 | Weight init & normalization | Xavier/Kaiming derived from variance preservation; residual connections as gradient highways |

### Tier 1 · Core Neural Nets
| # | Notebook | Focus |
|---|----------|-------|
| 06 | MLP end-to-end (MNIST) | Full training pipeline; overfitting shown via the validation-loss U-curve; dropout + weight decay |
| 07 | CNN from scratch | im2col conv/pool verified vs PyTorch; LeNet beats an MLP with fewer parameters; learned-filter visualization |
| 08 | RNN & LSTM from scratch | Backprop-through-time by hand; the vanishing-gradient problem measured; a char-level language model |
| 09 | Regularization & training dynamics | LR schedules (warmup + cosine), label smoothing, gradient clipping, early stopping — each shown on a real curve |

### Tier 2 · Transformers
| # | Notebook | Focus |
|---|----------|-------|
| 10 | Attention from scratch | Scaled dot-product → multi-head; causal masking; sinusoidal & RoPE positions; attention heatmaps |
| 11 | **GPT from scratch** | A full decoder-only transformer trained on tiny-Shakespeare; hand-written attention matched to PyTorch's kernel |
| 12 | Tokenization / BPE | Byte-pair encoding from scratch; compared to `tiktoken`; why tokenization causes LLM failure modes |
| 13 | Sampling & decoding | Greedy, temperature, top-k, top-p, beam search, speculative decoding |
| 14 | KV cache & inference | The O(n²)→O(n) generation speedup, the memory cost, and prefill-vs-decode systems tradeoffs |

### Tier 3 · Scale & Efficiency
| # | Notebook | Focus |
|---|----------|-------|
| 15 | Mixed precision & numerics | fp32/fp16/bf16, gradient underflow, loss scaling, fp32 master weights — why bf16 won |
| 16 | FlashAttention from scratch | Online (streaming) softmax + tiling → exact attention in O(n) memory, verified identical to standard attention |
| 17 | Parallelism | Data / tensor / pipeline / FSDP — each simulated and shown to reassemble exactly; the pipeline bubble |
| 18 | Scaling laws | A trained power-law sweep reproducing Kaplan/Chinchilla behavior; the compute-optimal argument |

### Tier 4 · LLM Pipeline
| # | Notebook | Focus |
|---|----------|-------|
| 19 | Pretraining → fine-tuning | The transfer-learning payoff, quantified; full fine-tune vs linear probe |
| 20 | LoRA / PEFT | Low-rank adapters from scratch; the low-rank premise verified via SVD; merge equivalence; rank sweep |
| 21 | RLHF: reward model + PPO + DPO | Bradley-Terry reward modeling, PPO, and DPO — the full alignment stack on a controlled toy |
| 22 | Instruction tuning & evaluation | Base vs instruction-tuned models; an eval harness; why LLM evaluation is the genuinely hard part |

### Tier 5 · Generative Models
| # | Notebook | Focus |
|---|----------|-------|
| 23 | VAE from scratch | Reparameterization trick, ELBO, a 2-D latent manifold visualization |
| 24 | GAN from scratch | The adversarial game; mode collapse on an 8-Gaussian ring; sharper MNIST samples |
| 25 | Diffusion (DDPM) | Forward/reverse processes, the noise-prediction objective; denoising trajectory + MNIST generation |
| 26 | Normalizing flows | RealNVP coupling layers; exact likelihood via change-of-variables |

### Tier 6 · Paper Reproductions — *the residency skill*
A repeatable reproduction workflow applied to six landmark results:

| # | Notebook | Paper reproduced | Result |
|---|----------|------------------|--------|
| 27 | Reproduction workflow + **Grokking** | Power et al. 2022 | Sudden generalization ~29× *after* memorization; weight-decay ablation confirms the mechanism |
| 28 | **Deep Double Descent** | Nakkiran et al. 2019 | Test-error peak exactly at the interpolation threshold, second descent beyond it |
| 29 | **Lottery Ticket Hypothesis** | Frankle & Carbin 2018 | Winning ticket at ~3% of weights; random-reinit control collapses |
| 30 | **Knowledge Distillation** | Hinton et al. 2015 | Student distilled from unlabeled data reaches ~teacher accuracy, far above its own labels |
| 31 | **Vision Transformer (ViT)** | Dosovitskiy et al. 2020 | Pure-transformer image classifier; the data-efficiency caveat vs CNNs |
| 32 | **Rethinking Generalization** | Zhang et al. 2016 | Networks fit random labels to 100% train accuracy — capacity ≠ generalization |

---

## Verification Standard

The defining feature of this repo: **claims are checked, not asserted.** Representative verified results (all produced by executing the notebooks):

- **Autograd** matches PyTorch gradients to ~1e-9; **hand-derived layer backward passes** (LayerNorm, BatchNorm, Conv2d) match to ~1e-14.
- **All six optimizers** match `torch.optim` across their full trajectories to machine precision.
- **FlashAttention** produces output *identical* to standard attention (≈4e-16) while never materializing the n×n score matrix.
- **From-scratch GPT** trains on tiny-Shakespeare to **1.56 nats/char** (random baseline: 4.17) and generates coherent Shakespearean text.
- **Reproductions** land: grokking's delayed generalization, double descent's interpolation-threshold peak, the lottery-ticket gap (winning ticket 0.92 vs random-reinit 0.72 at ~3% weights), distillation's transfer to a small student (0.97 vs 0.88), and 100%-train-accuracy memorization of random labels.

Every code cell carries `assert`-based correctness checks; the notebooks are run top-to-bottom before shipping, so figures and outputs are real, not placeholders.

---

## Architecture

The curriculum is a **dependency graph**: a small from-scratch engine (autograd → layers → optimizers → init) is built once in Tier 0, then reused and extended upward. Later tiers build the transformer stack on that foundation, then scaling/LLM/generative topics, culminating in paper reproductions that synthesize everything.

```mermaid
flowchart TD
    T0["Tier 0 · Foundations<br/>autograd · tensors · layers · optimizers · init"]
    T1["Tier 1 · Core Nets<br/>MLP · CNN · RNN/LSTM · training dynamics"]
    T2["Tier 2 · Transformers<br/>attention · GPT · BPE · sampling · KV cache"]
    T3["Tier 3 · Scale & Efficiency<br/>precision · FlashAttention · parallelism · scaling laws"]
    T4["Tier 4 · LLM Pipeline<br/>fine-tuning · LoRA · RLHF · eval"]
    T5["Tier 5 · Generative<br/>VAE · GAN · diffusion · flows"]
    T6["Tier 6 · Paper Reproductions<br/>grokking · double descent · lottery ticket · distillation · ViT · generalization"]

    T0 --> T1 --> T2 --> T3 --> T4
    T0 --> T5
    T2 --> T5
    T1 --> T6
    T2 --> T6
    T3 --> T6
```

**Why this shape:** the hardest ideas (attention, RLHF, diffusion) are only tractable to build once the primitives (autograd, backprop, optimization) are truly owned — so Tier 0 is deliberately exhaustive and everything above it reuses that engine rather than re-deriving it. Reproductions come last because they require fluency across the whole stack.

---

## Tech Stack

| Tool | Version | Role |
|------|---------|------|
| Python | 3.12 | Language |
| PyTorch | 2.6 (CUDA) | Reference implementation to verify against; GPU-accelerated training in later tiers |
| NumPy | 1.26 | The from-scratch engine (Tier 0–1 autograd, layers, FlashAttention) is pure NumPy |
| Matplotlib | 3.10 | All visualizations (computational graphs, attention maps, loss surfaces, latent manifolds) |
| torchvision | 0.21 | MNIST loading only |
| tiktoken | 0.12 | Comparison baseline for the from-scratch BPE tokenizer |
| Jupyter / nbconvert | — | Notebook authoring and end-to-end execution |

A CUDA GPU is used for the training-heavy notebooks (Tiers 1–6); everything degrades gracefully to CPU (built and tested on an RTX 4060 Laptop GPU).

---

## Getting Started

```bash
# 1. Clone
git clone <your-repo-url> && cd All-about-Neural-Networks

# 2. Create an environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
#    For GPU acceleration, install the CUDA build of PyTorch for your system
#    from https://pytorch.org/get-started/locally/

# 4. Launch Jupyter and start at Tier 0
jupyter lab      # or: jupyter notebook
```

Then open `Tier 0 - Foundations/01_autograd_from_scratch.ipynb` and work upward — the tiers are ordered so each builds on the last. Datasets (MNIST, tiny-Shakespeare) download automatically to a local `data/` folder on first run (git-ignored).

---

## Usage

Each notebook is meant to be **read and run top-to-bottom** — the narrative, math, and code are interleaved. Notebooks are also self-verifying; for example, the autograd engine checks itself against PyTorch:

```python
# From Tier 0 · 01_autograd_from_scratch.ipynb
a = Value(0.7); b = Value(-1.3); c = Value(2.1)
L = (a*b + a.exp()).tanh() + (c * a**2).relu() + a / (b + 2.0)
L.backward()                                  # our from-scratch reverse-mode autodiff

# ...compared against PyTorch on the same graph:
assert all(abs(o - t) < 1e-9 for o, t in zip(ours, theirs))
print("✅ matches PyTorch to ~1e-9")
```

To run a notebook non-interactively (e.g. to reproduce all outputs):

```bash
jupyter nbconvert --to notebook --execute --inplace "Tier 2 - Transformers/11_gpt_from_scratch.ipynb"
```

---

## Project Structure

```
All-about-Neural-Networks/
├── Tier 0 - Foundations/          # autograd, tensors, layers, optimizers, init (5 notebooks)
├── Tier 1 - Core Neural Nets/     # MLP, CNN, RNN/LSTM, training dynamics (4)
├── Tier 2 - Transformers/         # attention, GPT, BPE, sampling, KV cache (5)
├── Tier 3 - Scale and Efficiency/ # precision, FlashAttention, parallelism, scaling laws (4)
├── Tier 4 - LLM Pipeline/         # fine-tuning, LoRA, RLHF, eval (4)
├── Tier 5 - Generative Models/    # VAE, GAN, diffusion, flows (4)
├── Tier 6 - Paper Reproductions/  # grokking, double descent, lottery ticket, KD, ViT, generalization (6)
├── micrograd_NN_from_scratch.ipynb  # early standalone micrograd exploration
├── requirements.txt
├── LICENSE
└── README.md
```

Each notebook is standalone and ends with a **📚 References & papers** section citing the work it implements.

---

## Testing

There is no separate unit-test suite — instead, **each notebook is its own executable test.** Every notebook:

- carries inline `assert`-based correctness checks (gradient checks against finite differences, output comparisons against `torch.optim` / `F.scaled_dot_product_attention` / `F.conv2d` / etc.), so a broken implementation fails loudly on execution;
- is run end-to-end with `jupyter nbconvert --execute` before being committed, so all embedded figures and printed outputs reflect real runs.

Reproducing the full repo is therefore: execute every notebook and confirm no cell errors and no failed asserts.

---

## Results / Impact

This is an educational/portfolio repository, so "impact" is measured in *correctness and completeness of the reconstructions*, all produced by executing the notebooks:

- **32 notebooks**, from a scalar autograd engine to a trained GPT to six reproduced papers, each verified against a PyTorch reference.
- **From-scratch GPT**: 1.56 nats/char on tiny-Shakespeare (random baseline 4.17), generating coherent text.
- **Machine-precision agreement** with PyTorch on all hand-written gradients and optimizers (~1e-14 to 1e-9).
- **Six landmark papers reproduced** with results matching the original findings.

No production deployment, external users, or business metrics are claimed — this is a learning and research-preparation artifact.

---

## Roadmap / Future Work

- **Classical ML from scratch** — a foundations tier covering linear/logistic regression, PCA, k-means, k-NN, decision trees, bias-variance, and cross-validation, to ground the deep-learning material.
- Scale the from-scratch GPT with BPE tokens (Tier 2's tokenizer) and a longer context, and reproduce a slice of the GPT-2 loss curve end-to-end.
- Additional reproductions and modern architectures (state-space models / Mamba, a DDIM sampler).
- Convert the shared from-scratch engine into an importable `mininn` package so later notebooks import it rather than restating it.
- Add CI to execute all notebooks on push as a regression guard.

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE).

---

*Built as ground-up preparation for ML research residencies. If you're learning deep learning, start at Tier 0 and build upward — the fastest way to understand a system is to rebuild it.*
