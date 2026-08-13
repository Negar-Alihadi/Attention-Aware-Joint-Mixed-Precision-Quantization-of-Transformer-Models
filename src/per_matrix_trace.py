"""
Per-matrix (independent) Hessian trace -- the "three separate per-matrix Hessians" baseline that JAB-Hessian is explicitly contrasted against in
the proposal's Section 3.
"""
import torch
import torch.nn.functional as F
from .attention_loss import compute_attention
from .hutchinson_trace_estimator import hutchinson_trace_estimator
from .calibration import get_calibration_data

def compute_per_matrix_traces(model, block_idx, X, target_A, device, n_embd, samples=30):
    """
    For ONE block, ONE batch: compute three separate Hutchinson traces, one per matrix (Q, K, V), holding the other two fixed.
    Returns: dict {"Q": trace, "K": trace, "V": trace}
    """
    block = model.transformer.h[block_idx]
    W = block.attn.c_attn.weight.data
    W_Q, W_K, W_V = [w.clone() for w in W.split(n_embd, dim=1)]

    bias = block.attn.c_attn.bias.data
    b_Q, b_K, b_V = bias.split(n_embd)

    traces = {}

    # --- W_Q, holding K and V fixed ---
    w_flat = W_Q.clone().flatten()
    w_flat.requires_grad_(True)
    def loss_fn_Q(params, WK=W_K, WV=W_V):
        WQ_ = params.view(n_embd, n_embd)
        A_hat, _ = compute_attention(WQ_, WK, WV, X, b_Q=b_Q, b_K=b_K, b_V=b_V)
        return F.mse_loss(A_hat, target_A)
    traces["Q"] = hutchinson_trace_estimator(loss_fn_Q, w_flat, samples=samples).item()

    # --- W_K, holding Q and V fixed ---
    w_flat = W_K.clone().flatten()
    w_flat.requires_grad_(True)
    def loss_fn_K(params, WQ=W_Q, WV=W_V):
        WK_ = params.view(n_embd, n_embd)
        A_hat, _ = compute_attention(WQ, WK_, WV, X, b_Q=b_Q, b_K=b_K, b_V=b_V)
        return F.mse_loss(A_hat, target_A)
    traces["K"] = hutchinson_trace_estimator(loss_fn_K, w_flat, samples=samples).item()

    # --- W_V, holding Q and K fixed ---
    w_flat = W_V.clone().flatten()
    w_flat.requires_grad_(True)
    def loss_fn_V(params, WQ=W_Q, WK=W_K):
        WV_ = params.view(n_embd, n_embd)
        A_hat, _ = compute_attention(WQ, WK, WV_, X, b_Q=b_Q, b_K=b_K, b_V=b_V)
        return F.mse_loss(A_hat, target_A)
    traces["V"] = hutchinson_trace_estimator(loss_fn_V, w_flat, samples=samples).item()

    return traces

def compute_all_per_matrix_scores(model, batches, device, n_embd, samples=30, n_batches_to_use=4):
    """
    Same shape/contract as compute_all_jab_scores in jab_hessian.py: returns {block_name: score}, one scalar sensitivity score per block,
    averaged over n_batches_to_use calibration batches.
    The per-block score is Q+K+V summed, so it's directly comparable to the joint JAB-Hessian trace.
    """
    n_blocks = len(model.transformer.h)
    use_batches = batches[:n_batches_to_use]
    all_sums = {f"block_{i}_QKV": [] for i in range(n_blocks)}
    all_breakdowns = {f"block_{i}_QKV": [] for i in range(n_blocks)}  # keep Q/K/V split too, for inspection

    print(f"\nComputing PER-MATRIX (independent) traces for {n_blocks} blocks "
          f"(averaged over {len(use_batches)} batches, {samples} Hutchinson samples each)...")

    for b in use_batches:
        X_dict, target_A_dict, _ = get_calibration_data(model, b, device)
        for idx in range(n_blocks):
            traces = compute_per_matrix_traces(
                model, idx, X_dict[idx], target_A_dict[idx], device, n_embd, samples
            )
            block_name = f"block_{idx}_QKV"
            all_sums[block_name].append(traces["Q"] + traces["K"] + traces["V"])
            all_breakdowns[block_name].append(traces)

    scores = {}
    for block_name, sums in all_sums.items():
        score = sum(sums) / len(sums)
        scores[block_name] = score

        # average Q/K/V breakdown
        avg_Q = sum(d["Q"] for d in all_breakdowns[block_name]) / len(all_breakdowns[block_name])
        avg_K = sum(d["K"] for d in all_breakdowns[block_name]) / len(all_breakdowns[block_name])
        avg_V = sum(d["V"] for d in all_breakdowns[block_name]) / len(all_breakdowns[block_name])

        level = "HIGH" if score > 100 else "MEDIUM" if score > 10 else "LOW"
        print(f"  {block_name}: sum={score:.4f}  [{level} sensitivity]  "
              f"(Q={avg_Q:.4f}, K={avg_K:.4f}, V={avg_V:.4f})  "
              f"(min={min(sums):.4f}, max={max(sums):.4f})")

    print("Done!\n")
    return scores


if __name__ == "__main__":
    print("Testing per-matrix (independent) trace vs. JAB-Hessian (joint) trace...\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils import build_calibration_batches
    from jab_hessian import compute_all_jab_scores

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    n_embd = model.config.n_embd

    print("Creating calibration batches...")
    batches = build_calibration_batches(tokenizer, n_samples=128, seq_len=512)
    batches = [b.to(device) for b in batches]

    torch.manual_seed(42)

    print("\n=== Joint trace (existing JAB-Hessian) ===")
    joint_scores = compute_all_jab_scores(model, batches, device, n_embd, samples=20, n_batches_to_use=16)

    torch.manual_seed(42)

    print("\n=== Per-matrix (independent) trace ===")
    independent_scores = compute_all_per_matrix_scores(model, batches, device, n_embd, samples=20, n_batches_to_use=16)

    print("\n=== Comparison: joint vs. sum-of-independent, per block ===")
    for block_name in joint_scores:
        j = joint_scores[block_name]
        i = independent_scores[block_name]
        diff = j - i
        pct = (diff / i * 100) if i != 0 else float("nan")
        print(f"  {block_name}: joint={j:.4f}  independent_sum={i:.4f}  "
              f"diff={diff:+.4f} ({pct:+.1f}%)")
