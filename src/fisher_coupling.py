"""
Measures actual Q/K/V interaction via Fisher information (gradient dot products).
Unlike Hessian trace, this CAN see cross-matrix coupling.
"""
import torch
import torch.nn.functional as F
#from attention_loss import compute_attention
from attention_loss import attention_loss
from calibration import get_calibration_data


def compute_fisher_coupling(model, block_idx, X, target_A, target_attn, n_embd, b_Q, b_K, b_V, use_kl=True, lambda_kl=0.1, probe_bits=2):
    """
    probe_bits: the gradient MUST be evaluated at a PERTURBED (quantized)
    point, not the exact full-precision weights -- at full precision,
    A_hat == target_A exactly, the MSE residual is exactly zero, and the
    gradient (which is proportional to the residual) collapses to exactly
    zero regardless of the true Jacobian. Quantizing first gives a real,
    nonzero residual to differentiate against.
    """
    block = model.transformer.h[block_idx]
    W = block.attn.c_attn.weight.data  # (768, 2304) = [W_Q | W_K | W_V]

    # Simple round-to-nearest fake-quantization as the probe point (cheap,
    # no Hessian needed here -- just needs A_hat != target_A)
    qmax = 2 ** (probe_bits - 1) - 1
    scale = (W.abs().amax(dim=0, keepdim=True) / qmax).clamp(min=1e-8)
    W_quantized = torch.clamp(torch.round(W / scale), -qmax, qmax) * scale

    W_Q, W_K, W_V = [w.clone().requires_grad_(True) for w in W_quantized.split(n_embd, dim=1)]
    w_flat = torch.cat([W_Q.flatten(), W_K.flatten(), W_V.flatten()])

    loss = attention_loss(
        w_flat, X, target_A, n_embd,
        target_attn=target_attn if use_kl else None,
        lambda_kl=lambda_kl if use_kl else 0.0,
        b_Q=b_Q, b_K=b_K, b_V=b_V
    )
    g_Q, g_K, g_V = torch.autograd.grad(loss, [W_Q, W_K, W_V])
    g_Q, g_K, g_V = g_Q.flatten(), g_K.flatten(), g_V.flatten()

    def cos(a, b):
        return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()

    return {
        "QQ": torch.dot(g_Q, g_Q).item(), "KK": torch.dot(g_K, g_K).item(), "VV": torch.dot(g_V, g_V).item(),
        "QK_cos": cos(g_Q, g_K), "QV_cos": cos(g_Q, g_V), "KV_cos": cos(g_K, g_V),
    }


def compute_all_fisher_coupling(model, batches, device, n_embd, n_batches_to_use=8, use_kl=True, lambda_kl=0.1, probe_bits=2):
    n_blocks = len(model.transformer.h)
    use_batches = batches[:n_batches_to_use]

    raw = {f"block_{i}_QKV": [] for i in range(n_blocks)}  # list of per-batch dicts, per block

    print(f"\nComputing Fisher coupling for {n_blocks} blocks over {len(use_batches)} batches...")

    for b in use_batches:
        X_dict, target_A_dict, target_attn_dict = get_calibration_data(model, b, device, with_grad=True)
        for idx in range(n_blocks):
            block = model.transformer.h[idx]
            bias = block.attn.c_attn.bias.data
            b_Q, b_K, b_V = bias.split(n_embd)

            result = compute_fisher_coupling(
                model, idx, X_dict[idx], target_A_dict[idx], target_attn_dict[idx],
                n_embd, b_Q, b_K, b_V, use_kl=use_kl, lambda_kl=lambda_kl, probe_bits=2
            )
            raw[f"block_{idx}_QKV"].append(result)

    coupling_scores = {}
    for block_name, per_batch in raw.items():
        avg_QQ = sum(d["QQ"] for d in per_batch) / len(per_batch)
        avg_KK = sum(d["KK"] for d in per_batch) / len(per_batch)
        avg_VV = sum(d["VV"] for d in per_batch) / len(per_batch)

        signed_QK = sum(d["QK_cos"] for d in per_batch) / len(per_batch)
        signed_QV = sum(d["QV_cos"] for d in per_batch) / len(per_batch)
        signed_KV = sum(d["KV_cos"] for d in per_batch) / len(per_batch)

        magnitude_QK = sum(abs(d["QK_cos"]) for d in per_batch) / len(per_batch)
        magnitude_QV = sum(abs(d["QV_cos"]) for d in per_batch) / len(per_batch)
        magnitude_KV = sum(abs(d["KV_cos"]) for d in per_batch) / len(per_batch)

        coupling_scores[block_name] = {
            "signed": {"QK": signed_QK, "QV": signed_QV, "KV": signed_KV},
            "magnitude": {"QK": magnitude_QK, "QV": magnitude_QV, "KV": magnitude_KV},
            "raw": {"QQ": avg_QQ, "KK": avg_KK, "VV": avg_VV},
        }

        print(f"  {block_name}: diag(QQ={avg_QQ:.4f}, KK={avg_KK:.4f}, VV={avg_VV:.4f})  "
              f"signed(QK={signed_QK:+.4f}, QV={signed_QV:+.4f}, KV={signed_KV:+.4f})  "
              f"|magnitude|(QK={magnitude_QK:.4f}, QV={magnitude_QV:.4f}, KV={magnitude_KV:.4f})")

    print("Done!\n")
    return coupling_scores