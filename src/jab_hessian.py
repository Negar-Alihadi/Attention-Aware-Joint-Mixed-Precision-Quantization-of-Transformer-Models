"""
JAB-Hessian: Measures how sensitive each attention block is to quantization.
What it does:
1. Takes a GPT-2 model and calibration data
2. For each attention block (Q, K, V together), computes a sensitivity score
3. Higher score = block is more important = needs more bits
- Computes trace of Hessian of attention loss 
- Uses Hutchinson estimator for efficiency
"""

import torch
from .attention_loss import attention_loss
from .hutchinson_trace_estimator import hutchinson_trace_estimator
from .calibration import get_calibration_data
from .gptq_core import gptq_quantize_layer, collect_hessian_via_hook   # <-- ADDED: needed by the HAWQ-V2 functions below


def compute_jab_trace_from_data(model, block_idx, X, target_A, device, n_embd, samples=30):
    block = model.transformer.h[block_idx]
    W = block.attn.c_attn.weight.data
    W_Q, W_K, W_V = W.split(n_embd, dim=1)
    w_flat = torch.cat([W_Q.flatten(), W_K.flatten(), W_V.flatten()])
    w_flat.requires_grad_(True)

    bias = block.attn.c_attn.bias.data
    b_Q, b_K, b_V = bias.split(n_embd)

    def loss_fn(params):
        return attention_loss(params, X, target_A, n_embd, b_Q=b_Q, b_K=b_K, b_V=b_V)

    trace = hutchinson_trace_estimator(loss_fn, w_flat, samples=samples)
    return trace.item()


def compute_all_jab_scores(model, batches, device, n_embd, samples=30, n_batches_to_use=4):
    n_blocks = len(model.transformer.h)
    use_batches = batches[:n_batches_to_use]
    all_traces = {f"block_{i}_QKV": [] for i in range(n_blocks)}

    print(f"\nComputing JAB scores for {n_blocks} blocks "
          f"(averaged over {len(use_batches)} batches, {samples} Hutchinson samples each)...")

    for b in use_batches:
        X_dict, target_A_dict, _ = get_calibration_data(model, b, device)
        for idx in range(n_blocks):
            trace = compute_jab_trace_from_data(
                model, idx, X_dict[idx], target_A_dict[idx], device, n_embd, samples
            )
            all_traces[f"block_{idx}_QKV"].append(trace)

    scores = {}
    for block_name, traces in all_traces.items():
        trace = sum(traces) / len(traces)
        scores[block_name] = trace
        level = "HIGH" if trace > 100 else "MEDIUM" if trace > 10 else "LOW"
        print(f"  {block_name}: {trace:.4f}  [{level} sensitivity]  "
              f"(min={min(traces):.4f}, max={max(traces):.4f})")

    print("Done!\n")
    return scores


# ---- OLD heuristic, kept for comparison, NOT used by default anymore ----
def scores_to_allocator_format(jab_scores):
    """
    Old guessed heuristic: trace / bits^1.5 -- kept for reference/comparison.
    """
    allocator_input = {}
    bit_widths = [2, 3, 4, 8, 16]

    for block_name, trace in jab_scores.items():
        sensitivity_by_bits = {}
        for bits in bit_widths:
            sensitivity_by_bits[bits] = trace / (bits ** 1.5)
        allocator_input[block_name] = sensitivity_by_bits

    return allocator_input


# ---- NEW: HAWQ-V2 style, measured perturbation instead of guessed ----
def measure_weight_perturbation(original_W, H, bits):
    """
    HAWQ-V2 style: ||Q(W) - W||_F^2, measured directly in weight space.
    """
    w_copy = original_W.clone()
    gptq_quantize_layer(w_copy, H, bits=bits)
    perturbation = torch.norm(w_copy - original_W, p='fro') ** 2
    return perturbation.item()


def scores_to_allocator_format_hawqv2(jab_scores, model, calibration_batches, device, bit_widths=None):
    """
    HAWQ-V2 style: Omega_i(bits) = trace_i * ||Q(W_i) - W_i||_F^2,
    instead of the guessed trace / bits^1.5.
    """
    if bit_widths is None:
        bit_widths = [2, 3, 4, 8, 16]

    allocator_input = {}
    for block_name, trace in jab_scores.items():
        block_idx = int(block_name.split("_")[1])          # extract the index from "block_0_QKV" -> 0
        block = model.transformer.h[block_idx]
        c_attn = block.attn.c_attn
        original_W = c_attn.weight.data

        H = collect_hessian_via_hook(model, c_attn, calibration_batches, device)

        sensitivity_by_bits = {}
        for bits in bit_widths:
            perturbation = measure_weight_perturbation(original_W, H, bits)
            sensitivity_by_bits[bits] = trace * perturbation
        allocator_input[block_name] = sensitivity_by_bits

        print(f"  {block_name}: " + ", ".join(f"{b}bit={v:.4e}" for b, v in sensitivity_by_bits.items()))

    return allocator_input


def run_jab_allocation(model, batches, device, n_embd, target_avg_bits=4.0, samples=30, n_batches_to_use=16): # n_batches_to_use=4
    """
    1. Compute JAB scores for all blocks
    2. Convert to allocator format (HAWQ-V2 style, measured perturbation)
    3. Run greedy allocation
    4. Return bit assignment in format of a dictionary    
    """
    from .greedy_allocator_simple import greedy_allocate
    print("JAB-Hessian Adaptive Allocation")
    print("\nComputing JAB scores...")
    jab_scores = compute_all_jab_scores(model, batches, device, n_embd, samples, n_batches_to_use)

    print("Preparing for allocator (HAWQ-V2 style, measured perturbation)...")
    allocator_input = scores_to_allocator_format_hawqv2(jab_scores, model, batches, device)  # <-- CHANGED

    n_blocks = len(allocator_input)
    budget = target_avg_bits * n_blocks
    print(f"Budget: {budget:.1f} bits ({target_avg_bits} avg for {n_blocks} blocks)")

    print("\nRunning greedy allocation...")
    assignment, cost_used, sensitivity = greedy_allocate(allocator_input, budget)

    print("Results")
    print(f"Total cost: {cost_used:.1f} bits")
    print(f"Average bits: {cost_used / n_blocks:.2f}")
    print(f"Total sensitivity: {sensitivity:.6f}")
    print("\nPer-block allocation:")
    for block_name, bits in assignment.items():
        print(f"    {block_name}: {bits} bits")

    return assignment


if __name__ == "__main__":
    print("Testing JAB-Hessian...\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .utils import build_calibration_batches

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    n_embd = model.config.n_embd
    print(f"Model dimension: {n_embd}")
    print("Creating calibration batches...")
    batches = build_calibration_batches(tokenizer, n_samples=128, seq_len=512)
    batches = [b.to(device) for b in batches]

    torch.manual_seed(42)

    print("\nTesting compute_all_jab_scores on all blocks...")
    scores = compute_all_jab_scores(model, batches, device, n_embd, samples=10, n_batches_to_use=2)
    print("\nTesting HAWQ-V2 style allocator format on 2 blocks (quick check)...")
    small_scores = {k: scores[k] for k in list(scores.keys())[:2]}  # just block_0, block_1
    allocator_input = scores_to_allocator_format_hawqv2(small_scores, model, batches, device, bit_widths=[2, 3, 4, 8, 16])
    print(allocator_input)