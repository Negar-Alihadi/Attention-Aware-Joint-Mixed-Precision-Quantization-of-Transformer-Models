"""
Full pipeline using per-matrix traces for adaptive allocation.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.per_matrix_trace import compute_all_per_matrix_scores
from src.utils import build_calibration_batches, evaluate_perplexity
from src.jab_hessian import scores_to_allocator_format_hawqv2
from src.greedy_allocator_simple import greedy_allocate
from src.apply_allocation_to_pipeline import apply_allocation

def run_per_matrix_allocation(target_avg_bits=4.0, n_batches_to_use=16, samples=20):
    """
    Full pipeline using per-matrix traces for allocation.
    """
    print("PER-MATRIX ADAPTIVE ALLOCATION PIPELINE")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    # Load model
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    n_embd = model.config.n_embd
    print(f"  n_embd: {n_embd}")
    # Build calibration data
    print("\nBuilding calibration data...")
    calibration_batches = build_calibration_batches(tokenizer, n_samples=128, seq_len=512)
    calibration_batches = [b.to(device) for b in calibration_batches]
    print(f"  {len(calibration_batches)} batches")
    # Compute per-matrix scores
    print(f"\nComputing per-matrix scores (samples={samples}, batches={n_batches_to_use})...")
    torch.manual_seed(42)
    scores = compute_all_per_matrix_scores(model=model, batches=calibration_batches, device=device, n_embd=n_embd, samples=samples, n_batches_to_use=n_batches_to_use)
    # Convert to allocator format
    print("\nConverting to allocator format...")
    allocator_input = scores_to_allocator_format_hawqv2(scores, model, calibration_batches, device)
    
    # Set budget
    n_blocks = len(allocator_input)
    budget = target_avg_bits * n_blocks
    print(f"\nBudget: {budget:.1f} bits ({target_avg_bits} avg for {n_blocks} blocks)")
    
    # Run allocation
    print("\nRunning greedy allocation...")
    assignment, cost_used, sensitivity = greedy_allocate(allocator_input, budget)
    print("\nAllocation Results:")
    print(f"  Total cost: {cost_used:.1f} bits")
    print(f"  Average bits: {cost_used / n_blocks:.2f}")
    print(f"  Total sensitivity: {sensitivity:.6f}")
    print("\nPer-block allocation:")
    for block_name, bits in assignment.items():
        print(f"  {block_name}: {bits} bits")
    
    # Apply allocation
    print("\nApplying allocation to model...")
    model = apply_allocation(model, assignment, calibration_batches, device)
    # Evaluate
    print("\nEvaluating perplexity...")
    ppl = evaluate_perplexity(model, tokenizer)
    print(f"\nPer-matrix adaptive {target_avg_bits}-bit perplexity: {ppl:.3f}")
    return assignment, ppl

def compare_allocation_methods():
    """Compare uniform, JAB, and per-matrix allocation."""
    print("COMPARING ALLOCATION METHODS")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model for uniform baseline
    print("\n=== Uniform 4-bit Baseline ===")
    model_uniform = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model_uniform.eval()
    calibration_batches = build_calibration_batches(tokenizer, n_samples=128, seq_len=512)
    calibration_batches = [b.to(device) for b in calibration_batches]
    # Quantize uniformly
    print("Quantizing all blocks to 4 bits...")
    from gptq_core import gptq_quantize_layer, collect_hessian_via_hook
    
    for idx in range(len(model_uniform.transformer.h)):
        block = model_uniform.transformer.h[idx]
        c_attn = block.attn.c_attn
        H = collect_hessian_via_hook(model_uniform, c_attn, calibration_batches, device)
        gptq_quantize_layer(c_attn.weight.data, H, bits=4, group_size=128, act_order=True)
    
    ppl_uniform = evaluate_perplexity(model_uniform, tokenizer)
    print(f"Uniform 4-bit perplexity: {ppl_uniform:.3f}")
    # Run per-matrix allocation
    print("\n=== Per-Matrix Adaptive Allocation ===")
    _, ppl_matrix = run_per_matrix_allocation(target_avg_bits=4.0, n_batches_to_use=8, samples=15)
    # Compare
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Uniform 4-bit:      {ppl_uniform:.3f}")
    print(f"Per-matrix adaptive: {ppl_matrix:.3f}")
    print(f"Improvement:        {ppl_uniform - ppl_matrix:+.3f}")

def main():
    """Run all tests."""
    # Test multiple bit configurations
    for bits in [3.0, 4.0, 6.0, 8.0]:
        print("\n" + "=" * 70)
        print(f"TESTING {bits} BITS")
        print("=" * 70)
        
        run_per_matrix_allocation(
            target_avg_bits=bits,
            n_batches_to_use=4,
            samples=10
        )
    
    # Uncomment for full comparison
    # compare_allocation_methods()

if __name__ == "__main__":
    main()