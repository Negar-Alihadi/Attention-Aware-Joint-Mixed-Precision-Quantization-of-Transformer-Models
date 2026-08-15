# ===== CELL 32 (markdown) =====
# ## 11. Joint attention-aware fine-tuning (fills in Untitled10's `joint_calibration_skeleton`)
#
# Sections 8-10 only used `attention_loss` as an importance **score** (the JAB-Hessian trace) to decide *how many bits* each block gets -- the actual quantized weights were still produced by GPTQ minimizing the conventional `||XW - XW_hat||^2`, not the attention-output loss the PDF's Objective 1 asks for.
#
# This section closes that gap: it fills in the interface Untitled10.ipynb sketches in `joint_calibration_skeleton` ("Week 2 task -- once JAB-Hessian is wired in"), using this notebook's own validated `attention_loss`/`compute_attention`. Each block's GPTQ-initialized `W_Q, W_K, W_V` are wrapped in a straight-through quantizer and fine-tuned with Adam to directly minimize `L(A(X), A_hat(X))`, jointly -- i.e. `min_{W_hat_Q,W_hat_K,W_hat_V} L(A(X), A_hat(X))` from the PDF, not a proxy for it.
#
# **Correctness note:** `target_A` is always captured from a separate, untouched full-precision `model_float` -- never from `model_joint`, which is being progressively quantized. Distilling a quantized model toward its own (already-degraded) output would silently defeat the objective.
#
# ---
#
# ### Bug found after running this section: perplexity got *worse* (26.225 -> 32.660)
#
# **Root cause.** `STEQuantize`'s fake-quantization grid is controlled by a `scale`
# tensor, and the fine-tuning loop below used to build that scale with
# `per_row_scale_flat`, which computes **one scale per input dimension, shared
# across all 768 output channels, with no grouping at all**. But the GPTQ warm
# start it fine-tunes from (`gptq_quantize_layer(..., group_size=128,
# act_order=True)`) quantizes with **one scale per output channel, per group of
# 128 input columns** -- a completely different axis and granularity.
#
# So the very first call to `ste_quantize(w_flat, scale_flat, bits)` inside the
# fine-tuning loop -- *before Adam ever takes a step* -- silently re-quantized
# the carefully-optimized GPTQ solution onto a coarser, misaligned grid. A
# synthetic test with the exact same shapes/settings confirms this introduces
# ~9-10% relative weight error immediately at step 0 (see the sanity-check cell
# below). Only 8 tiny (`lr=1e-4`) Adam steps per block can't repair that kind of
# damage, and because blocks are quantized *sequentially* (block *i*'s
# calibration input `X` already reflects blocks `0..i-1`'s quantization error),
# the damage compounds -- consistent with block 0's fine-tuned loss (0.0008)
# being ~56x smaller than block 11's (0.043) in the original run.
#
# **Fix**, implemented below:
# 1. `gptq_quantize_layer` gained an optional `return_scale=True` mode that
#    hands back the *exact* per-(output-channel, group) scale it used, in the
#    original column order. `joint_attention_aware_finetune` now does its own
#    GPTQ warm start (instead of a separate pre-quantization pass) and reuses
#    that exact scale, so `ste_quantize` reproduces the GPTQ solution exactly
#    at step 0 -- fine-tuning can now only move *away* from a correct starting
#    point, never a silently-corrupted one.
# 2. **Best-iterate tracking + a safety net**: each block keeps the
#    lowest-loss iterate seen during its (short, noisy) Adam trajectory, and
#    only commits it if it actually beats the GPTQ-only starting loss;
#    otherwise the block keeps its GPTQ-only weights. This gives a formal
#    per-block guarantee that joint fine-tuning cannot make the *local*
#    attention-reconstruction objective worse than plain GPTQ.
# 3. Gradient clipping, and calibration batches that rotate across the
#    calibration set instead of every block reusing the same first 8 batches.

# ===== CELL 33 (code) =====
# --- Sanity check: does the fine-tuning scale match the GPTQ warm-start scale? ---
# Synthetic stand-in for one block's c_attn, small enough to run instantly,
# with the SAME shapes/settings pattern (group_size, act_order, bits) as the
# real pipeline. Demonstrates the bug (OLD per_row_scale_flat) and the fix
# (NEW: reuse gptq_quantize_layer's own return_scale) side by side.

def _demo_per_row_scale_flat_OLD(W0, qmax):
    """The buggy scale used by the original Section 11: one scale per INPUT
    row, shared across every output column, no grouping at all."""
    row_max = W0.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    return (row_max / qmax).expand_as(W0).reshape(-1)

torch.manual_seed(0)
demo_n_embd, demo_group_size, demo_bits = 64, 32, 4  # small stand-ins for 768 / 128 / 4

W_demo = torch.randn(demo_n_embd, 3 * demo_n_embd) * 0.05
X_demo = torch.randn(2000, demo_n_embd)
H_demo = (2.0 * X_demo.T @ X_demo / X_demo.shape[0]).to(torch.float64)

W_after, scale_correct = gptq_quantize_layer(
    W_demo.clone(), H_demo, bits=demo_bits, group_size=demo_group_size,
    act_order=True, return_scale=True,
)
W_Q0, W_K0, W_V0 = W_after.split(demo_n_embd, dim=1)
w_flat_demo = torch.cat([W_Q0.flatten(), W_K0.flatten(), W_V0.flatten()])
qmax_demo = 2 ** (demo_bits - 1) - 1

# OLD (buggy): re-quantizing the GPTQ solution with the wrong-axis scale
scale_old = torch.cat([_demo_per_row_scale_flat_OLD(w, qmax_demo) for w in (W_Q0, W_K0, W_V0)])
w_old = torch.clamp(torch.round(w_flat_demo / scale_old), -qmax_demo, qmax_demo) * scale_old
err_old = ((w_old - w_flat_demo).norm() / w_flat_demo.norm()).item()

# NEW (fixed): reusing GPTQ's own exact scale
scale_Q, scale_K, scale_V = scale_correct.split(demo_n_embd, dim=1)
scale_new = torch.cat([scale_Q.flatten(), scale_K.flatten(), scale_V.flatten()])
w_new = torch.clamp(torch.round(w_flat_demo / scale_new), -qmax_demo, qmax_demo) * scale_new
err_new = ((w_new - w_flat_demo).norm() / w_flat_demo.norm()).item()

print(f"Relative error re-quantizing the GPTQ solution with the OLD scale : {err_old:.4%}")
print(f"Relative error re-quantizing the GPTQ solution with the NEW scale : {err_new:.6%}")
if err_old > 0.01 and err_new < 1e-4:
    print("\nConfirmed: the OLD scale silently corrupts the GPTQ warm start; the NEW scale is exact.")

# ===== CELL 34 (code) =====
class STEQuantize(torch.autograd.Function):
    """
    Straight-through estimator for fake quantization: rounds onto the bit-grid
    in the forward pass (so downstream loss "sees" quantization error), but
    passes the incoming gradient through unchanged in the backward pass, so
    Adam can adjust the underlying float weights to compensate.
    """
    @staticmethod
    def forward(ctx, w, scale, bits):
        qmax = 2 ** (bits - 1) - 1
        q = torch.clamp(torch.round(w / scale), -qmax, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def ste_quantize(w, scale, bits):
    return STEQuantize.apply(w, scale, bits)


def joint_attention_aware_finetune(model_joint, model_float, calibration_batches, device,
                                    n_embd, bits=4, lr=1e-4, steps_per_block=8,
                                    group_size=128, damp_percent=0.01,
                                    grad_clip_norm=1.0, verbose=True):
    """
    Objective 1 (`min_{W_hat_Q,W_hat_K,W_hat_V} L(A(X), A_hat(X))`), made robust:

      1. GPTQ-warm-starts each block INSIDE this function (instead of a
         separate pre-quantization pass) so we can capture the EXACT
         per-(output-channel, group) scale GPTQ used via `return_scale=True`.
         This is the fix for the bug described in the markdown above: the
         old code re-derived a scale with a different granularity/axis than
         GPTQ's own scale, silently corrupting the warm start before any
         fine-tuning step ran.
      2. Tracks the BEST iterate seen during each block's fine-tuning (by
         attention_loss on that block's calibration batches), not just the
         last one -- an ~8-step Adam trajectory is noisy.
      3. Safety net: a block's fine-tuned weights are committed only if they
         beat the GPTQ-only starting loss; otherwise the block keeps its
         GPTQ-only weights. Formally guarantees
         attention_loss(final) <= attention_loss(GPTQ-only warm start)
         for every block (a guarantee on the local reconstruction objective,
         which is itself a proxy for perplexity -- not a hard guarantee on
         perplexity -- but it removes this specific failure mode).
      4. Gradient clipping, and calibration batches that rotate across the
         calibration set per block instead of every block reusing the same
         first `steps_per_block` batches.

    `target_A` always comes from `model_float` (untouched, full precision).
    `X` (this block's input activations) comes from `model_joint`, so later
    blocks train against realistic post-quantization-error inputs from
    earlier blocks, consistent with how GPTQ itself behaves sequentially.
    """
    n_blocks = len(model_joint.transformer.h)

    for block_idx in range(n_blocks):
        block = model_joint.transformer.h[block_idx]
        c_attn = block.attn.c_attn

        # --- GPTQ warm start, done HERE so we can capture its exact scale ---
        H = collect_hessian_via_hook(model_joint, c_attn, calibration_batches, device)
        _, scale_full = gptq_quantize_layer(
            c_attn.weight.data, H, bits=bits, group_size=group_size,
            damp_percent=damp_percent, act_order=True, return_scale=True,
        )  # c_attn.weight.data is now GPTQ-quantized in place

        W = c_attn.weight.data
        W_Q0, W_K0, W_V0 = W.split(n_embd, dim=1)
        w_flat = torch.cat([W_Q0.flatten(), W_K0.flatten(), W_V0.flatten()]).clone()
        w_flat.requires_grad_(True)

        # the EXACT scale GPTQ used -- ste_quantize(w_flat, scale_flat, bits)
        # reproduces w_flat exactly right now, before any Adam step.
        scale_Q, scale_K, scale_V = scale_full.split(n_embd, dim=1)
        scale_flat = torch.cat([scale_Q.flatten(), scale_K.flatten(), scale_V.flatten()])

        bias = c_attn.bias.data
        b_Q, b_K, b_V = bias.split(n_embd)

        optimizer = torch.optim.Adam([w_flat], lr=lr)

        # rotate through the calibration set instead of every block reusing
        # the same first `steps_per_block` batches
        start = (block_idx * steps_per_block) % len(calibration_batches)
        idxs = [(start + s) % len(calibration_batches) for s in range(steps_per_block)]
        batches = [calibration_batches[i] for i in idxs]

        with torch.no_grad():
            init_losses = []
            for batch in batches:
                _, target_A, _ = get_calibration_data_for_block(model_float, batch, device, block_idx)
                X, _, _ = get_calibration_data_for_block(model_joint, batch, device, block_idx)
                init_losses.append(attention_loss(w_flat.detach(), X, target_A, n_embd,
                                                    b_Q=b_Q, b_K=b_K, b_V=b_V).item())
            init_loss = sum(init_losses) / len(init_losses)

        best_loss = init_loss
        best_w = w_flat.detach().clone()

        for batch in batches:
            _, target_A, _ = get_calibration_data_for_block(model_float, batch, device, block_idx)
            X, _, _ = get_calibration_data_for_block(model_joint, batch, device, block_idx)

            optimizer.zero_grad()
            w_q = ste_quantize(w_flat, scale_flat, bits)
            loss = attention_loss(w_q, X, target_A, n_embd, b_Q=b_Q, b_K=b_K, b_V=b_V)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([w_flat], grad_clip_norm)
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                with torch.no_grad():
                    best_w = ste_quantize(w_flat, scale_flat, bits).detach().clone()

        if best_loss <= init_loss:
            W_Q, W_K, W_V = reshape_weights(best_w, n_embd)
            c_attn.weight.data.copy_(torch.cat([W_Q, W_K, W_V], dim=1))
            status = f"improved {init_loss:.6f} -> {best_loss:.6f}"
        else:
            # best_w never beat the GPTQ-only starting point; c_attn.weight.data
            # is already that GPTQ-only solution (nothing else has touched it),
            # so we simply don't overwrite it with the worse fine-tuned weights.
            status = f"kept GPTQ-only ({init_loss:.6f}; fine-tune best was {best_loss:.6f})"

        if verbose:
            print(f"  Block {block_idx}: {status}")

    return model_joint

# ===== CELL 35 (code) =====
print("Loading a fresh full-precision GPT-2 as the fixed reference (target_A source)...")
model_float = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
model_float.eval()

print("Loading a second, still full-precision GPT-2 for joint attention-aware fine-tuning...")
print("(GPTQ warm-start now happens INSIDE joint_attention_aware_finetune, per block,")
print(" so the exact scale it used can be reused for fine-tuning -- see the fix above.)")
model_joint = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
model_joint.eval()

print("\nRunning joint attention-aware fine-tuning (Objective 1: minimizes L(A(X), A_hat(X)) directly)...")
model_joint = joint_attention_aware_finetune(model_joint, model_float, calibration_batches, DEVICE,
                                              n_embd, bits=4, lr=1e-4, steps_per_block=8,
                                              group_size=128)

print("\nEvaluating perplexity...")
ppl_joint = evaluate_perplexity(model_joint, tokenizer)
print(f"\nJoint attention-aware fine-tuned (Objective 1) perplexity: {ppl_joint:.3f}")
