<div align="center">

### Attention-Aware Joint Mixed-Precision Quantization of Transformer Models

*Attention-aware, second-order sensitivity analysis for adaptive bit allocation in GPT-2 and Mistral-7B*

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](#prerequisites)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](#prerequisites)
[![Status](https://img.shields.io/badge/status-research--in--progress-orange)](#roadmap)

Nafiseh Hosseinpourfardi · Negar Alihadi

</div>

---

## Abstract

Post-training quantization (PTQ) methods overwhelmingly treat each linear layer as an isolated approximation problem, solving `min ‖XW − XŴ‖²_F` independently per layer. This ignores a key structural fact about Transformers: the Query, Key, and Value projections of an attention head only matter through their **joint, nonlinear** effect on the attention output

```
A(X) = softmax( X·W_Q · (X·W_K)ᵀ / √d_k ) · X·W_V
```

Because the softmax couples `W_Q`, `W_K`, and `W_V` nonlinearly, small per-matrix quantization errors can compound into a large, *correlated* distortion of the attention pattern. **JAB-Hessian** shifts the unit of quantization from the individual matrix to the functional QKV block: it derives a single sensitivity score — the trace of the Hessian of the joint attention-reconstruction loss with respect to the *concatenated* `[W_Q ; W_K ; W_V]` block — and uses it to drive both the quantization objective and the mixed-precision bit-allocation policy.

---

## Table of Contents

- [Abstract](#abstract)
- [Table of Contents](#table-of-contents)
- [Motivation](#motivation)
- [Core Idea](#core-idea)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
  - [1. Attention-Aware Joint Quantization](#1-attention-aware-joint-quantization)
  - [2. Importance / Sensitivity Criteria](#2-importance--sensitivity-criteria)
  - [3. Adaptive Mixed-Precision Allocation](#3-adaptive-mixed-precision-allocation)
- [Experimental Setup](#experimental-setup)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [2. Run the uniform-precision baseline](#2-run-the-uniform-precision-baseline)
  - [3. Explore interactively](#3-explore-interactively)
  - [Customization](#customization)
- [Evaluation Metrics](#evaluation-metrics)
- [Related Work](#related-work)
- [Roadmap](#roadmap)
- [References](#references)

---

## Motivation

Transformers dominate modern deep learning, but their memory footprint and compute cost make edge deployment difficult. Standard PTQ pipelines (GPTQ, AWQ, OmniQuant, …) are highly effective at compressing *individual* weight matrices, but none of them optimize — or measure sensitivity with respect to — the **joint** effect of `W_Q`, `W_K`, `W_V` on the attention output itself.

---

## Core Idea

| Stage | Standard PTQ | JAB-Hessian |
|---|---|---|
| **Objective** | Per-matrix reconstruction: `‖XW − XŴ‖²` | Joint attention-output reconstruction: `‖A(X) − Â(X)‖²_F + λ·KL(attention maps)` |
| **Sensitivity criterion** | Per-layer Hessian trace | **Joint Attention-Block Hessian Trace**: `Ω_block = Tr(H_joint)` over the concatenated `[W_Q ; W_K ; W_V]` |
| **Estimation** | Exact / diagonal Hessian | Hutchinson's stochastic trace estimator (Rademacher probes) |
| **Bit allocation** | Uniform or per-layer heuristic | Greedy sensitivity-per-cost allocator with Pareto-optimal / MCKP refinement |
| **Quantizer** | GPTQ (per-layer) | GPTQ extended to a block-structured joint Hessian |

The full derivation, positioning against prior work (GPTQ, AWQ, SmoothQuant, OmniQuant, APTQ, HAWQ / HAWQ-V2), and experimental protocol are laid out in the accompanying [research proposal](Attention-Aware_Quantization_Proposal_v2.pdf).

---

## Repository Structure

```
jab-hessian-quantization/
│
├── README.md
├── requirements.txt
├── apply_allocation_to_pipeline.py     # 🚀 Main entry point — full JAB-Hessian pipeline
├── uniform_baseline.py                 # Uniform-precision GPTQ baseline
│
├── notebooks/                          # Interactive validation & experiment notebooks
│   ├── week1.ipynb                                  # Main Week-1 validation notebook
│   ├── GPTQ_CORE.ipynb                               # From-scratch GPTQ core + calibration + perplexity
│   ├── version1_git_JAB_correct1.ipynb               # JAB-Hessian, GPT-2, base version
│   ├── version1_git_JAB_correct1_with_KL.ipynb       # + attention-map KL divergence term
│   ├── version1_git_JAB_with_C4.ipynb                # C4 calibration set
│   ├── version1_git_JAB_with_KL_Fisher.ipynb         # + Fisher-information sensitivity
│   ├── version1_git_JAB_with_KL_Fisher_Pareto_Frontier.ipynb          # + Pareto-frontier bit allocation
│   ├── version1_git_JAB_with_KL_Fisher_Pareto_Frontier_Last_Version.ipynb
│   ├── version1_git_with_per_matrix_trace.ipynb      # Per-matrix (ungrouped) trace ablation
│   ├── version1_git_JAB_gpt2_c4_metrics.ipynb        # C4 calibration, WikiText-2 eval, 4 metrics
│   ├── version1_git_JAB_gpt2_notMod.ipynb            # Adaptive allocation evaluation
│   ├── version1_git_JAB_correct1_with_KL_testLess_mistral7b_colabT4.ipynb  # Mistral-7B port (Colab T4)
│   ├── JAB_Mistral_calibration(c4)_evaluation(wiki).ipynb   # Mistral-7B, C4 calib → WikiText-2 eval
│   ├── JAB_Mistral_calibration(wiki)_evaluation(c4).ipynb   # Mistral-7B, WikiText-2 calib → C4 eval
│   └── final_version_git_Mistral_wikitext.ipynb      # Final Mistral-7B / WikiText-2 run
│
├── src/                                 # Core library
│   ├── attention_loss.py                # Attention-aware loss: MSE + optional KL divergence
│   ├── calibration.py                   # Captures X (activations) and target A(X) via forward hooks
│   ├── gptq_core.py                     # From-scratch GPTQ implementation (Hessian collection via hooks)
│   ├── jab_hessian.py                   # Computes per-block JAB-Hessian sensitivity scores
│   ├── hutchinson_trace_estimator.py    # Hutchinson's trace estimator (Rademacher vectors)
│   ├── fisher_coupling.py               # Fisher-information sensitivity criterion
│   ├── per_matrix_trace.py              # Per-matrix (non-joint) Hessian trace, for ablation
│   ├── greedy_allocator_simple.py       # Greedy sensitivity-per-cost bit allocator
│   ├── mckp_ilp_pareto.py               # Multiple-Choice Knapsack (ILP) + Pareto-frontier solver
│   ├── 09_uniform_gptq.py               # Uniform-precision GPTQ baseline routine
│   ├── 10_adaptive_allocation.py        # Adaptive allocation driver
│   ├── 11_joint_attention_finetune.py   # Joint-attention GPTQ fine-tuning step
│   └── utils.py                         # Calibration batching + sliding-window perplexity eval
│
└── tests/                               # Unit & integration tests
    ├── test_attention_loss.py
    ├── test_calibration.py
    ├── test_hutchinson_trace_estimator.py
    ├── test_fisher_coupling_simple.py
    ├── test_fisher_coupling_gpt2.py
    ├── test_per_matrix_trace_simple.py
    ├── test_per_matrix_trace_gpt2.py
    ├── test_per_matrix_pipeline.py
    ├── test_mckp_ilp_pareto.py
    ├── run_allocation.py
    ├── testing_new.py
    ├── testing on a small real batch.py
    └── Test JAB-Hessian on all blocks.py
```

> 📌 The `src/` and `tests/` directories reflect the modular files of the project; the `notebooks/` directory preserves the full experimental history, including the GPT-2 → Mistral-7B generalization runs.

---

## Methodology

### 1. Attention-Aware Joint Quantization

The joint loss is non-convex in `(Ŵ_Q, Ŵ_K, Ŵ_V)` because of the softmax nonlinearity, so a closed-form Hessian inversion is not directly available. The pipeline implements and compares:

- **Strategy — Learned reconstruction (OmniQuant-style)**: optimizes learnable rounding/clipping parameters against the joint attention loss via calibration gradient steps.

**Loss functions** compared: pure MSE on `A(X)`, attention-map KL divergence, and the combined loss `L = ‖A(X) − Â(X)‖²_F + λ·KL(attention maps)`.

**Calibration strategy**: standard WikiText-2 / C4 calibration.

### 2. Importance / Sensitivity Criteria

| Criterion | What it captures | Reference |
|---|---|---|
| **Joint Attention-Block Hessian Trace (proposed)** | Curvature of the joint QKV attention-output loss over the *concatenated* block | This work |
| Fisher information | Empirical outer-product-of-gradients proxy for the Hessian | Sensitivity-based quantization literature |
| Oracle (brute-force) sensitivity | Ground-truth importance from actually quantizing each block and measuring the resulting loss increase — the reference signal all cheap estimators are validated against | This work |
| Layer-wise relevance propagation | Attributes the joint loss back to each block via propagated relevance scores, rather than a single backward-pass gradient/curvature estimate | Voita et al. (propagation-based head importance) |

### 3. Adaptive Mixed-Precision Allocation

Given a sensitivity score `Ω_i` per block and a cost `C(b_i)` (parameter count × bit-width, or measured latency), bit-widths `b_i ∈ {2, 3, 4, 8, 16}` are chosen to solve

```
min  L(F(X), F̂(X))    s.t.   Σ_i C(b_i) ≤ B
```

via three complementary solvers:

1. **Multiple-Choice Knapsack (MCKP)** — each block's discrete precision choice is an "item"; solved for minimum total sensitivity under budget `B` (HAWQ-style).
2. **Pareto-frontier search** — traces the Pareto-optimal (cost, sensitivity) frontier per block and selects the knee-point configuration (HAWQ-V2-style); the primary allocator used in practice.
3. **Exact ILP** — small-scale exact solve, used to validate that the greedy/Pareto solution is near-optimal.

---

## Experimental Setup

**Primary setup** (development & core experiments)
- **Model**: GPT-2 small (124M params, decoder-only)
- **Dataset**: WikiText-2 (calibration + evaluation)
- **Metric**: perplexity, validation accuracy, attention reconstruction error, and quantization error

**Extended setup** (generalization)
- **Model**: Mistral-7B
- **Calibration / evaluation**: cross-tested on WikiText-2 and C4 in both directions (calibrate on one, evaluate on the other) to probe out-of-distribution robustness of the allocation
---

## Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU recommended (CPU is sufficient for small GPT-2-scale runs)

```

### Dependencies

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
numpy>=1.24.0
```

---

## Usage

### 1. Run the full JAB-Hessian adaptive-allocation pipeline

```bash
python apply_allocation_to_pipeline.py
```

This will:
- ✅ Load GPT-2 small
- ✅ Build calibration data from WikiText-2 (128 samples × 512 tokens)
- ✅ Compute JAB-Hessian scores for all 12 attention blocks
- ✅ Allocate bit-widths adaptively under a target average budget
- ✅ Quantize the model with the extended GPTQ core
- ✅ Evaluate perplexity on the WikiText-2 test set

Key configuration (edit at the top of the script):

```python
target_avg_bits = 6.0      # Target average bit-width
n_batches_to_use = 20      # Number of calibration batches
samples = 30                # Hutchinson samples per block
```

### 2. Run the uniform-precision baseline

```bash
python uniform_baseline.py
```

### 3. Explore interactively

```bash
jupyter notebook notebooks/week1.ipynb              # Main GPT-2 validation
jupyter notebook notebooks/GPTQ_CORE.ipynb          # GPTQ baseline core
jupyter notebook notebooks/final_version_git_Mistral_wikitext.ipynb   # Mistral-7B generalization
```

### Customization

Bit-width choices — edit in `src/greedy_allocator_simple.py` / `src/mckp_ilp_pareto.py`:
```python
BIT_WIDTHS = [2, 3, 4, 8, 16]
```

Budget — edit in `apply_allocation_to_pipeline.py`:
```python
target_avg_bits = 6.0
```

Calibration size — edit in `src/utils.py`:
```python
def build_calibration_batches(tokenizer, n_samples=128, seq_len=512):
    ...
```

---

## Evaluation Metrics

- **Attention-reconstruction error** `‖A(X) − Â(X)‖_F` — the direct, cheap Objective-1 signal
- **Validation perplexity** (language modeling) — the task-level signal on WikiText-2 / C4
- **Quantization error** — weight-level distortion introduced by quantization (e.g. `‖W − Ŵ‖`), the low-level signal used to sanity
---

## Related Work

This project builds directly on:

- **Attention** — Vaswani et al., *Attention Is All You Need* (NeurIPS 2017)
- **Layer-wise PTQ baselines** — GPTQ (Frantar et al., ICLR 2023), Optimal Brain Compression (NeurIPS 2022), AWQ (MLSys 2024), SmoothQuant (ICML 2023), OmniQuant (ICLR 2024)
- **Attention-aware / joint quantization** — APTQ (Guan et al., DAC 2024), Q-BERT (Shen et al., AAAI 2020)
- **Hessian-based mixed precision** — HAWQ (Dong et al., ICCV 2019), HAWQ-V2 (Dong et al., NeurIPS 2020)
- **Attention-specific importance** — Michel, Levy & Neubig (NeurIPS 2019), Voita et al. (ACL 2019)

Full annotated references are in the [research proposal](Attention-Aware_Quantization_Proposal_v2.pdf).

---

## Roadmap

- [x] From-scratch GPTQ core + WikiText-2 perplexity validation
- [x] Joint attention-output loss (MSE + KL) and JAB-Hessian sensitivity score
- [x] Fisher-information sensitivity ablation
- [x] Greedy allocator → Pareto-frontier / MCKP-ILP bit allocation
- [x] Generalization to Mistral-7B, cross-calibrated on WikiText-2 / C4
---

## References

1. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
2. Frantar, Ashkboos, Hoefler & Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023.
3. Frantar & Alistarh. *Optimal Brain Compression.* NeurIPS 2022.
4. Lin et al. *AWQ: Activation-aware Weight Quantization.* MLSys 2024.
5. Xiao et al. *SmoothQuant.* ICML 2023.
6. Shao et al. *OmniQuant.* ICLR 2024.
7. Guan et al. *APTQ: Attention-aware Post-Training Mixed-Precision Quantization.* DAC 2024.
8. Shen et al. *Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT.* AAAI 2020.
9. Dong et al. *HAWQ: Hessian AWare Quantization of Neural Networks.* ICCV 2019.
10. Dong et al. *HAWQ-V2.* NeurIPS 2020.
11. Michel, Levy & Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS 2019.
12. Voita et al. *Analyzing Multi-Head Self-Attention.* ACL 2019.

