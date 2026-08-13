"""
Test per_matrix_trace.py with actual GPT-2 model.
Uses a small amount of data for quick testing.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.per_matrix_trace import compute_per_matrix_traces, compute_all_per_matrix_scores
from src.utils import build_calibration_batches, evaluate_perplexity
from src.calibration import get_calibration_data
from src.jab_hessian import compute_all_jab_scores

def test_single_block_gpt2():
    """Test per-matrix traces on one block of GPT-2."""
    print("\n=== Test: Per-Matrix Traces on GPT-2 Block ===") 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    # Load model (small for testing)
    print("  Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    n_embd = model.config.n_embd
    print(f"  n_embd: {n_embd}")
    # Create a small calibration batch
    print("  Creating calibration data...")
    text = "The quick brown fox jumps over the lazy dog. " * 10
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
    # Get calibration data for block 0
    from calibration import get_calibration_data_for_block
    X, target_A, _ = get_calibration_data_for_block(model, batch, device, block_idx=0)
    # Compute per-matrix traces for block 0
    print("  Computing per-matrix traces for block 0...")
    torch.manual_seed(42)
    traces = compute_per_matrix_traces(model=model, block_idx=0, X=X, target_A=target_A, device=device, n_embd=n_embd, samples=30)  # Fewer samples for speed)
    print(f"  Trace Q: {traces['Q']:.6f}")
    print(f"  Trace K: {traces['K']:.6f}")
    print(f"  Trace V: {traces['V']:.6f}")
    print(f"  Sum: {traces['Q'] + traces['K'] + traces['V']:.6f}")
    # Also compute JAB trace for comparison
    print("\n  Computing JAB trace for comparison...")
    from jab_hessian import compute_jab_trace_from_data
    torch.manual_seed(42)
    jab_trace = compute_jab_trace_from_data(model=model, block_idx=0, X=X, target_A=target_A, device=device, n_embd=n_embd, samples=30)
    print(f"  JAB trace (joint): {jab_trace:.6f}")
    print(f"  Sum of independent: {traces['Q'] + traces['K'] + traces['V']:.6f}")
    print(f"  Difference: {jab_trace - (traces['Q'] + traces['K'] + traces['V']):.6f}")
    return traces, jab_trace

def test_all_blocks_gpt2():
    """Test per-matrix traces on all blocks of GPT-2."""
    print("\n=== Test: Per-Matrix Traces on All GPT-2 Blocks ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    # Load model
    print("  Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    n_embd = model.config.n_embd
    print(f"  n_embd: {n_embd}")
    # Build calibration batches (small for testing)
    print("  Building calibration batches...")
    calibration_batches = build_calibration_batches(tokenizer, 
        n_samples=32,  # Fewer samples for speed
        seq_len=128    # Shorter sequence for speed
    )
    calibration_batches = [b.to(device) for b in calibration_batches]
    print(f"  {len(calibration_batches)} batches ready")
    
    # Compute per-matrix scores on a subset of blocks
    print("  Computing per-matrix scores (first 4 blocks only)...")
    torch.manual_seed(42)
    scores = compute_all_per_matrix_scores(
        model=model,
        batches=calibration_batches,
        device=device,
        n_embd=n_embd,
        samples=20,  # Fewer samples for speed
        n_batches_to_use=4  # Only use 4 batches for speed
    )
    # Show results
    print("\n  Results:")
    for block_name, score in scores.items():
        if int(block_name.split("_")[1]) < 4:  # Only show first 4
            print(f"    {block_name}: {score:.4f}")
    
    return scores

def test_vs_jab():
    """Compare per-matrix scores vs JAB scores on GPT-2."""
    print("\n=== Test: Compare Per-Matrix vs JAB Scores ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    # Load model
    print("  Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    n_embd = model.config.n_embd
    # Build small calibration set
    print("  Building calibration batches...")
    calibration_batches = build_calibration_batches(tokenizer, 
        n_samples=16,   # Small for speed
        seq_len=64      # Short sequence
    )
    calibration_batches = [b.to(device) for b in calibration_batches]
    # Compute both scores
    print("  Computing JAB scores...")
    torch.manual_seed(42)
    jab_scores = compute_all_jab_scores(model=model, batches=calibration_batches, device=device, n_embd=n_embd, samples=20, n_batches_to_use=4) 
    print("  Computing per-matrix scores...")
    torch.manual_seed(42)
    matrix_scores = compute_all_per_matrix_scores(model=model, batches=calibration_batches, device=device, n_embd=n_embd, samples=20, n_batches_to_use=4)
    # Compare
    print("\n  Comparison (first 4 blocks):")
    print(f"  {'Block':<15} {'JAB':<12} {'Matrix Sum':<12} {'Diff':<12}")
    print("  " + "-" * 55)
    
    for i in range(4):
        block_name = f"block_{i}_QKV"
        jab = jab_scores[block_name]
        matrix = matrix_scores[block_name]
        diff = jab - matrix
        pct = (diff / matrix * 100) if matrix != 0 else float("nan")
        print(f"  {block_name:<15} {jab:<12.4f} {matrix:<12.4f} {diff:<+12.4f} ({pct:+.1f}%)")
    
    return jab_scores, matrix_scores

def main():
    print("TESTING PER_MATRIX_TRACE WITH GPT-2")   
    # Test 1: Single block
    test_single_block_gpt2()
    # Test 2: All blocks (subset)
    test_all_blocks_gpt2()
    # Test 3: Compare with JAB
    test_vs_jab()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")

if __name__ == "__main__":
    main()