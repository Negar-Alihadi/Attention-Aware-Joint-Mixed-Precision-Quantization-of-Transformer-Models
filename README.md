# Attention-Aware-Joint-Mixed-Precision-Quantization-of-Transformer-Models

**Adaptive bit allocation for attention layers in GPT-2 using second-order sensitivity analysis**

## Overview

This project implements **Attention-Aware-Joint-Mixed-Precision-Quantization-of-Transformer-Models**, a novel mixed-precision quantization method for Transformer models. Unlike traditional layer-wise quantization that treats each linear layer independently, JAB-Hessian:

- **Quantizes Q, K, V projections jointly** to preserve the attention mechanism's output
- **Allocates bit-widths adaptively** based on attention-output sensitivity
- **Uses Hutchinson's trace estimator** for efficient second-order curvature computation

The key insight is that small individual errors in W_Q, W_K, and W_V can compound into large correlated errors in the attention output A(X) due to the softmax nonlinearity. JAB-Hessian captures this coupling explicitly by computing the Hessian trace of the joint attention loss.

## 📁 Project Structure

```
Attention-Aware-Joint-Mixed-Precision-Quantization-of-Transformer-Models/
│
├── README.md # Project documentation
├── apply_allocation_to_pipeline.py # Main entry: Full JAB-Hessian pipeline
├── uniform_baseline.py # Uniform GPTQ baseline for comparison
│
├── src/ # Source code (core library)
│ ├── init.py # Package initialization
│ ├── attention_loss.py # Attention-aware loss functions (MSE + KL)
│ ├── calibration.py # Calibration data capture (X, A(X))
│ ├── gptq_core.py # GPTQ quantization core implementation
│ ├── greedy_allocator_simple.py # Greedy sensitivity-per-cost allocator
│ ├── hutchinson_trace_estimator.py # Hutchinson's trace estimator for Hessian
│ ├── jab_hessian.py # JAB-Hessian sensitivity computation
│ └── utils.py # Shared utilities (calibration, perplexity)
│
├── tests/ # Unit tests
│ ├── init.py # Test package initialization
│ ├── test_attention_loss.py # Tests for attention loss functions
│ └── test_hutchinson.py # Tests for Hutchinson estimator
│
├── notebooks/ # Jupyter notebooks
  ├── week1.ipynb # Main validation notebook (Week 1)
  └── GPTQ_CORE.ipynb # GPTQ quantization core validation
```

## File Descriptions

| File | Purpose |
|------|---------|
| **apply_allocation_to_pipeline.py** | Full pipeline: computes JAB scores, runs allocation, quantizes model, evaluates perplexity |
| **uniform_baseline.py** | Uniform bit-width GPTQ baseline for comparison against adaptive allocation |
| **attention_loss.py** | Defines MSE loss between real and quantized attention output, plus optional KL divergence |
| **calibration.py** | Captures X (input activations) and target_A (attention outputs) via forward hooks |
| **gptq_core.py** | From-scratch GPTQ implementation with Hessian collection via hooks |
| **greedy_allocator_simple.py** | Allocates bit-widths under budget using greedy heuristic with upgrade refinement |
| **hutchinson_trace_estimator.py** | Hutchinson's method for trace(H) estimation using Rademacher vectors |
| **jab_hessian.py** | Computes per-block JAB-Hessian sensitivity scores; converts to allocator format |
| **utils.py** | Calibration batch construction + sliding-window perplexity evaluation |
| **week1.ipynb** | Interactive notebook with step-by-step validation and experiments |
| **GPTQ_CORE.ipynb** | Implementation of GPTQ quantization in notebook |

## Features

| Feature | Description |
|---------|-------------|
| **Joint Attention Quantization** | Optimizes Q/K/V weights together against the attention output |
| **Adaptive Bit Allocation** | Greedy sensitivity-per-cost allocator with Pareto-optimal refinement |
| **Hessian Trace Estimation** | Hutchinson's stochastic estimator for cheap curvature computation |
| **Multi-metric Support** | MSE loss + optional KL divergence on attention maps |
| **GPT-2 Compatible** | Full implementation for GPT-2 small (124M parameters) |
| **Modular Design** | Plug-and-play components for easy experimentation |

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended; CPU works for small runs)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/jab-hessian-quantization.git  # TODO: update with actual repo URL
cd jab-hessian-quantization

# Install dependencies
pip install torch transformers datasets numpy
```

### Dependencies

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
numpy>=1.24.0
```

## Usage

### 1. Run JAB-Hessian Adaptive Allocation (Main Entry)

```bash
python apply_allocation_to_pipeline.py
```

This will:

- ✅ Load GPT-2 small
- ✅ Build calibration data from WikiText-2 (128 samples, 512 tokens each)
- ✅ Compute JAB-Hessian scores for all 12 attention blocks
- ✅ Allocate bit-widths adaptively under target average budget
- ✅ Quantize the model using GPTQ
- ✅ Evaluate perplexity on WikiText-2 test set

**Configuration** (modify in script):

```python
target_avg_bits = 6.0      # Target average bit-width
n_batches_to_use = 20      # Number of calibration batches
samples = 30                # Hutchinson samples per block
```

### 2. Run Uniform Baseline

```bash
python uniform_baseline.py
```

Compares uniform quantization (all blocks same bit-width) against adaptive allocation.

### 3. Run Notebooks

For interactive experimentation:

```bash
jupyter notebook week1.ipynb      # Main validation
jupyter notebook GPTQ_CORE.ipynb  # GPTQ baseline
```

## Methodology

### 1. Attention-Aware Loss Function

The loss measures how much quantization distorts the attention output:

```
L = ||A(X) - Â(X)||²_F + λ · KL(attention_maps)
```

- **MSE term**: Quantization error in attention output
- **KL term**: Distortion of attention patterns (optional, λ = 0.1)

### 2. JAB-Hessian Sensitivity

For each block's concatenated `[W_Q | W_K | W_V]`, compute:

```
Ω_block = Tr(H_joint)
```

Where `H_joint` is the Hessian of the attention loss w.r.t. the concatenated weights, estimated via Hutchinson's trace estimator.

### 3. Greedy Bit Allocation

1. Start with highest bit-width (16 bits) for all blocks
2. Iteratively downgrade blocks with smallest sensitivity-per-cost increase
3. Upgrade blocks if budget remains (Pareto-optimal refinement)

### 4. GPTQ Quantization

Uses GPTQ's inverse-Hessian error compensation, extended to handle joint Q/K/V blocks with:

1. **Group_size = 128**: Finer granularity for scales
2. **Act_order = True**: Quantizes columns in order of decreasing Hessian diagonal

## Customization Guide

### Changing Bit-Width Options

Modify in `greedy_allocator_simple.py`:

```python
BIT_WIDTHS = [2, 3, 4, 8, 16]  # Available choices
```

### Changing Budget

Modify target average bits in `apply_allocation_to_pipeline.py`:

```python
target_avg_bits = 6.0  # Change to desired average
```

### Changing Calibration Data

Modify in `utils.py` or `week1.ipynb`:

```python
def build_calibration_batches(tokenizer, n_samples=128, seq_len=512):
    # Adjust n_samples and seq_len as needed
```
