"""
Measures actual Q/K/V interaction via Fisher information (gradient dot products).
Unlike Hessian trace, this CAN see cross-matrix coupling.
"""
import torch
import torch.nn.functional as F
from .attention_loss import compute_attention
from .calibration import get_calibration_data


def compute_fisher_coupling(model, block_idx, X, target_A, n_embd, b_Q, b_K, b_V):
    """
    For one block, one batch: computes gradients g_Q, g_K, g_V of the joint
    attention-output loss w.r.t. the FULL-PRECISION W_Q, W_K, W_V, then
    returns all pairwise dot products (diagonal and cross-block Fisher
    traces). Single backward pass -- much cheaper than Hutchinson's
    double-backward trick used for the Hessian trace.
    """
    block = model.transformer.h[block_idx]
    W = block.attn.c_attn.weight.data
    W_Q, W_K, W_V = [w.clone().requires_grad_(True) for w in W.split(n_embd, dim=1)]

    A_hat, _ = compute_attention(W_Q, W_K, W_V, X, b_Q=b_Q, b_K=b_K, b_V=b_V)
    loss = F.mse_loss(A_hat, target_A)

    g_Q, g_K, g_V = torch.autograd.grad(loss, [W_Q, W_K, W_V])
    g_Q, g_K, g_V = g_Q.flatten(), g_K.flatten(), g_V.flatten()

    return {
        "QQ": torch.dot(g_Q, g_Q).item(),
        "KK": torch.dot(g_K, g_K).item(),
        "VV": torch.dot(g_V, g_V).item(),
        "QK": torch.dot(g_Q, g_K).item(),
        "QV": torch.dot(g_Q, g_V).item(),
        "KV": torch.dot(g_K, g_V).item(),
    }


def compute_all_fisher_coupling(model, batches, device, n_embd, n_batches_to_use=8):
    """
    Averages compute_fisher_coupling over multiple batches and blocks, then
    reports BOTH the raw diagonal/cross-term traces and the normalized
    coupling scores (cosine similarity between gradient directions,
    bounded in [-1, +1]) -- the actual "how coupled are Q/K/V" answer.
    """
    n_blocks = len(model.transformer.h)
    use_batches = batches[:n_batches_to_use]

    raw = {f"block_{i}_QKV": {k: [] for k in ["QQ", "KK", "VV", "QK", "QV", "KV"]} for i in range(n_blocks)}

    print(f"\nComputing Fisher coupling for {n_blocks} blocks over {len(use_batches)} batches...")

    for b in use_batches:
        X_dict, target_A_dict, _ = get_calibration_data(model, b, device)
        for idx in range(n_blocks):
            block = model.transformer.h[idx]
            bias = block.attn.c_attn.bias.data
            b_Q, b_K, b_V = bias.split(n_embd)

            result = compute_fisher_coupling(model, idx, X_dict[idx], target_A_dict[idx], n_embd, b_Q, b_K, b_V)
            for k, v in result.items():
                raw[f"block_{idx}_QKV"][k].append(v)

    coupling_scores = {}
    for block_name, terms in raw.items():
        avg = {k: sum(v) / len(v) for k, v in terms.items()}

        # normalized coupling: cosine similarity between gradient directions
        coupling_QK = avg["QK"] / ((avg["QQ"] * avg["KK"]) ** 0.5 + 1e-12)
        coupling_QV = avg["QV"] / ((avg["QQ"] * avg["VV"]) ** 0.5 + 1e-12)
        coupling_KV = avg["KV"] / ((avg["KK"] * avg["VV"]) ** 0.5 + 1e-12)

        coupling_scores[block_name] = {
            "QK": coupling_QK, "QV": coupling_QV, "KV": coupling_KV,
            "raw": avg,
        }

        print(f"  {block_name}: coupling(Q,K)={coupling_QK:+.4f}  "
              f"coupling(Q,V)={coupling_QV:+.4f}  coupling(K,V)={coupling_KV:+.4f}")

    print("Done!\n")
    return coupling_scores