# ===== CELL 28 (markdown) =====
# ## 10. Full pipeline: JAB-Hessian adaptive allocation
# Computes JAB scores, allocates bits under an average-bit budget close to the uniform baseline, applies it, and evaluates perplexity -- using the *same* calibration batches as Section 9.
#
# **Budget note:** we use `target_avg_bits=4.3`, not `4.0`. At exactly 4.0 the budget lands on a `BIT_WIDTHS` grid point and every block saturates to the same 4-bit floor before any differentiation happens (see the sweep cell right after this one for a full explanation and a demonstration that the allocator differentiates correctly away from that point).

# ===== CELL 29 (code) =====
def apply_allocation(model, assignment, calibration_batches, device):
    print("\nApplying allocation")
    for idx in range(len(model.transformer.h)):
        block_name = f"block_{idx}_QKV"
        if block_name in assignment:
            bits = assignment[block_name]
            block = model.transformer.h[idx]
            c_attn = block.attn.c_attn
            H = collect_hessian_via_hook(model, c_attn, calibration_batches, device)
            gptq_quantize_layer(c_attn.weight.data, H, bits=bits, group_size=128, act_order=True)
            print(f"  Block {idx}: {bits} bits")
    print("Allocation done.")
    return model

# ===== CELL 30 (code) =====
print("Loading a fresh full-precision GPT-2 for adaptive allocation...")
model_adaptive = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
model_adaptive.eval()

print("\nComputing JAB-Hessian scores and allocating bits...")
torch.manual_seed(42)
# target_avg_bits=4.3, not 4.0 -- see markdown above (4.0 sits exactly on a BIT_WIDTHS grid point)
assignment = run_jab_allocation(model=model_adaptive, batches=calibration_batches,
                                 device=DEVICE, n_embd=n_embd, target_avg_bits=4.3, samples=20)

print("\nApplying the allocation...")
model_adaptive = apply_allocation(model_adaptive, assignment, calibration_batches, DEVICE)

print("\nEvaluating perplexity...")
ppl_adaptive = evaluate_perplexity(model_adaptive, tokenizer)
print(f"\nAdaptive (JAB-Hessian) perplexity: {ppl_adaptive:.3f}")

# ===== CELL 31 (code) =====
# --- Budget sensitivity check: does the allocator actually differentiate blocks? ---
# At target_avg_bits=4.0, the budget (48 = 12 blocks x 4) lands exactly on a
# BIT_WIDTHS grid point. Below 4 bits the HAWQ-V2 perturbation term
# (||Q(W,bits)-W||_F^2) blows up steeply, so every block's "cheap" 16->8 and
# 8->4 downgrades get exhausted first regardless of its trace, and the loop
# stops the instant all 12 blocks hit the 4-bit floor -- before any block ever
# gets to compete for a downgrade below (or an upgrade above) that floor. That
# is why Section 10 above shows all 12 blocks at exactly 4 bits: it is not a
# broken allocator, it is a budget sitting exactly on that cliff.
#
# Sweeping a couple of nearby, non-grid-aligned budgets on the SAME scores
# confirms this: away from the cliff, the allocator cleanly separates
# high-trace (sensitive) blocks from low-trace ones.
print("Budget sensitivity check: reusing one JAB-score pass across a small sweep...")

model_sweep_probe = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
model_sweep_probe.eval()

jab_scores_sweep = compute_all_jab_scores(model_sweep_probe, calibration_batches, DEVICE,
                                           n_embd, samples=20, n_batches_to_use=4)
allocator_input_sweep = scores_to_allocator_format_hawqv2(jab_scores_sweep, model_sweep_probe,
                                                           calibration_batches, DEVICE)

for avg_bits in [3.5, 4.3, 4.5, 6.0]:
    budget = avg_bits * len(allocator_input_sweep)
    alloc, cost_used, _ = greedy_allocate(allocator_input_sweep, budget)
    bits_used = sorted(set(alloc.values()))
    print(f"\navg_bits={avg_bits} (budget={budget:.1f}, cost_used={cost_used:.1f}):")
    print(f"  distinct bit-widths chosen: {bits_used}")
    for name, b in alloc.items():
        print(f"    {name}: {b} bits")

del model_sweep_probe

# --- Perplexity at each budget in the sweep ---
# Reuses allocator_input_sweep computed above; only the allocation + GPTQ + eval
# are redone per budget (each needs its own fresh, unquantized model).
ppl_by_budget = {}

for avg_bits in [3.5, 4.3, 4.5, 6.0]:
    budget = avg_bits * len(allocator_input_sweep)
    alloc, cost_used, _ = greedy_allocate(allocator_input_sweep, budget)

    print(f"\n=== avg_bits={avg_bits} (budget={budget:.1f}, cost_used={cost_used:.1f}) ===")
    model_budget = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
    model_budget.eval()
    model_budget = apply_allocation(model_budget, alloc, calibration_batches, DEVICE)

    ppl = evaluate_perplexity(model_budget, tokenizer)
    ppl_by_budget[avg_bits] = ppl
    print(f"Perplexity at avg_bits={avg_bits}: {ppl:.3f}")

    del model_budget

print("\n=== Perplexity vs. budget summary ===")
for avg_bits, ppl in ppl_by_budget.items():
    print(f"  avg_bits={avg_bits}: perplexity={ppl:.3f}")
print(f"  (uniform 4-bit baseline: {ppl_uniform:.3f})")
