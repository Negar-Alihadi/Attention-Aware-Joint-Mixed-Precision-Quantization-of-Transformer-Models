"""
Random-split baseline: same total bit budget as JAB, same discrete
bit-width menu, but WHICH block gets WHICH bit-width is decided
randomly instead of by sensitivity. This isolates whether JAB's
specific (sensitivity-informed) choice of blocks beats an uninformed
choice at the same total cost.
"""
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.gptq_core import gptq_quantize_layer, collect_hessian_via_hook
from src.utils import build_calibration_batches, evaluate_perplexity
from src.greedy_allocator_simple import greedy_allocate, BIT_WIDTHS


def random_allocation(block_names, budget, seed):
    """
    Build a random per-block bit-width assignment that still respects
    the exact same budget and bit-width menu as JAB, by feeding random
    'fake sensitivity' scores into the SAME greedy allocator JAB uses.
    This guarantees the assignment is achievable and directly
    comparable -- only the ranking signal changes, not the mechanism.
    """
    rng = random.Random(seed)
    fake_scores = {}
    for name in block_names:
        # random sensitivity value per bit-width, no relation to
        # real JAB traces or the bits^1.5 heuristic
        fake_scores[name] = {b: rng.uniform(0.0, 100.0) for b in BIT_WIDTHS}

    assignment, cost_used, _ = greedy_allocate(fake_scores, budget)
    return assignment, cost_used


def apply_allocation(model, assignment, calibration_batches, device):
    print("\nApplying random allocation")
    for idx in range(len(model.transformer.h)):
        block_name = f"block_{idx}_QKV"
        if block_name in assignment:
            bits = assignment[block_name]
            block = model.transformer.h[idx]
            c_attn = block.attn.c_attn

            H = collect_hessian_via_hook(model, c_attn, calibration_batches, device)
            gptq_quantize_layer(c_attn.weight.data, H, bits=bits)
            print(f"Block {idx}: {bits} bits")
    print("Allocation is Done!")
    return model


if __name__ == "__main__":
    print("Random-Split Baseline for GPT-2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading GPT-2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    n_blocks = len(model.transformer.h)

    print("\nCreating calibration data")
    calibration_batches = build_calibration_batches(tokenizer, n_samples=128, seq_len=512)
    print(f"{len(calibration_batches)} batches")

    target_avg_bits = 5.0
    budget = target_avg_bits * n_blocks
    block_names = [f"block_{i}_QKV" for i in range(n_blocks)]

    # Run a FEW different random seeds and average -- one random draw
    # could get lucky or unlucky; several draws gives a fair comparison
    # point instead of a single noisy sample.
    seeds = [0, 1, 2]
    ppls = []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Random seed {seed}")
        print(f"{'='*50}")

        # fresh model each time -- quantization is destructive/in-place
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        model.eval()

        assignment, cost_used = random_allocation(block_names, budget, seed)
        print(f"Budget: {budget:.1f} bits, achieved: {cost_used:.1f} bits")
        for name, bits in assignment.items():
            print(f"    {name}: {bits} bits")

        model = apply_allocation(model, assignment, calibration_batches, device)

        ppl = evaluate_perplexity(model, tokenizer)
        print(f"\n[seed {seed}] Perplexity: {ppl:.2f}")
        ppls.append(ppl)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n\n=== Summary ===")
    for seed, ppl in zip(seeds, ppls):
        print(f"seed={seed}: PPL={ppl:.2f}")
    print(f"Average random-split PPL: {sum(ppls)/len(ppls):.2f}")
    print(f"(compare against: uniform-5-bit=25.20, JAB avg-5.0=27.09)")