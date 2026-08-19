"""
Test Fisher coupling with actual GPT-2 model.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fisher_coupling import compute_fisher_coupling, compute_all_fisher_coupling
from utils import build_calibration_batches
from calibration import get_calibration_data_for_block


def test_single_block_gpt2():
    """Test Fisher coupling on one block of GPT-2."""
    print("\n=== Test: Fisher Coupling on GPT-2 Block ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print("  Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    n_embd = model.config.n_embd
    print(f"  n_embd: {n_embd}")
    print("  Creating calibration data...")
    text = "The quick brown fox jumps over the lazy dog. " * 10
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)

    # target_attn is now needed too, not discarded
    X, target_A, target_attn = get_calibration_data_for_block(model, batch, device, block_idx=0, with_grad=True)

    block = model.transformer.h[0]
    bias = block.attn.c_attn.bias.data
    b_Q, b_K, b_V = bias.split(n_embd)

    print("  Computing Fisher coupling for block 0 (MSE only)...")
    torch.manual_seed(42)
    result = compute_fisher_coupling(model, 0, X, target_A, target_attn, n_embd, b_Q, b_K, b_V,
                                      use_kl=True, lambda_kl=0.01)

    print(f"  QQ: {result['QQ']:.6f}")
    print(f"  KK: {result['KK']:.6f}")
    print(f"  VV: {result['VV']:.6f}")
    print(f"  QK (single-batch cosine): {result['QK_cos']:+.6f}")
    print(f"  QV (single-batch cosine): {result['QV_cos']:+.6f}")
    print(f"  KV (single-batch cosine): {result['KV_cos']:+.6f}")

    return result


def test_all_blocks_gpt2():
    """Test Fisher coupling on all blocks of GPT-2."""
    print("\n=== Test: Fisher Coupling on All GPT-2 Blocks ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print("  Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    n_embd = model.config.n_embd

    print("  Building calibration batches...")
    calibration_batches = build_calibration_batches(tokenizer, n_samples=16, seq_len=64)
    calibration_batches = [b.to(device) for b in calibration_batches]
    print(f"  {len(calibration_batches)} batches ready")

    print("  Computing Fisher coupling for all blocks...")
    torch.manual_seed(42)
    scores = compute_all_fisher_coupling(model=model, batches=calibration_batches, device=device,
                                          n_embd=n_embd, n_batches_to_use=4,
                                          use_kl=True, lambda_kl=0.01)

    print("\n  Results (first 4 blocks) -- signed (direction) vs. magnitude (|coupling|):")
    for block_name in list(scores.keys())[:4]:
        s = scores[block_name]
        print(f"    {block_name}:")
        print(f"      signed:    QK={s['signed']['QK']:+.4f}, QV={s['signed']['QV']:+.4f}, KV={s['signed']['KV']:+.4f}")
        print(f"      magnitude: QK={s['magnitude']['QK']:.4f}, QV={s['magnitude']['QV']:.4f}, KV={s['magnitude']['KV']:.4f}")

    return scores


def test_coupling_interpretation():
    """Interpret Fisher coupling results."""
    print("\n=== Test: Interpretation of Coupling ===")
    print("\n  Interpretation guide:")
    print("    - magnitude ≈ 0: Matrices show no instantaneous coupling on any batch")
    print("    - magnitude >> 0, signed ≈ 0: Real per-batch coupling exists, but its")
    print("      DIRECTION is not consistent across different calibration text")
    print("      (this is the case to watch for -- averaging signed values would")
    print("      hide real coupling here)")
    print("    - magnitude ≈ |signed|: Coupling direction IS consistent across batches")
    print("\n  This distinction is what the Hessian trace CANNOT see at all --")
    print("  and what a naive signed-average of Fisher cross-terms can also miss.")


def main():
    print("TESTING FISHER COUPLING WITH GPT-2")
    test_single_block_gpt2()
    test_all_blocks_gpt2()
    #test_coupling_interpretation()
    print("ALL TESTS COMPLETED!")


if __name__ == "__main__":
    main()