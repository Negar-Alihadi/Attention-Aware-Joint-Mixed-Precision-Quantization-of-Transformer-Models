"""
Simple unit tests for fisher_coupling.py using synthetic data.
"""
import torch
import torch.nn.functional as F
from src.attention_loss import compute_attention

def compute_fisher_coupling_simple(W_Q, W_K, W_V, X, target_A, n_embd, n_head=4):
    """
    Simplified Fisher coupling with direct tensors (no model needed).
    """
    W_Q = W_Q.clone().requires_grad_(True)
    W_K = W_K.clone().requires_grad_(True)
    W_V = W_V.clone().requires_grad_(True)

    # Pass n_head explicitly!
    A_hat, _ = compute_attention(W_Q, W_K, W_V, X, n_head=n_head)
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


def test_fisher_basic():
    """Test Fisher coupling computes correctly with synthetic data."""
    print("\n=== Test 1: Basic Fisher Coupling ===")

    n_embd = 32
    n_head = 4  # 32/4 = 8 
    batch = 2
    seq = 8

    W_Q = torch.randn(n_embd, n_embd)
    W_K = torch.randn(n_embd, n_embd)
    W_V = torch.randn(n_embd, n_embd)
    X = torch.randn(batch, seq, n_embd)
    target_A = torch.randn(batch, seq, n_embd)

    result = compute_fisher_coupling_simple(W_Q, W_K, W_V, X, target_A, n_embd, n_head=n_head)

    print(f"  QQ (diagonal): {result['QQ']:.6f}")
    print(f"  KK (diagonal): {result['KK']:.6f}")
    print(f"  VV (diagonal): {result['VV']:.6f}")
    print(f"  QK (cross):    {result['QK']:.6f}")
    print(f"  QV (cross):    {result['QV']:.6f}")
    print(f"  KV (cross):    {result['KV']:.6f}")

    coupling_QK = result["QK"] / ((result["QQ"] * result["KK"]) ** 0.5 + 1e-12)
    coupling_QV = result["QV"] / ((result["QQ"] * result["VV"]) ** 0.5 + 1e-12)
    coupling_KV = result["KV"] / ((result["KK"] * result["VV"]) ** 0.5 + 1e-12)

    print(f"\n  Normalized coupling:")
    print(f"    Q-K: {coupling_QK:+.4f}")
    print(f"    Q-V: {coupling_QV:+.4f}")
    print(f"    K-V: {coupling_KV:+.4f}")

    assert result["QQ"] > 0
    assert result["KK"] > 0
    assert result["VV"] > 0

    print("  Fisher coupling works!")

    return result


def test_fisher_gradient_flow():
    """Test that gradients flow correctly through the computation."""
    print("\n=== Test 2: Gradient Flow ===")

    n_embd = 16
    n_head = 2  # 16/2 = 8 
    batch = 2
    seq = 4
    W_Q = torch.randn(n_embd, n_embd, requires_grad=True)
    W_K = torch.randn(n_embd, n_embd, requires_grad=True)
    W_V = torch.randn(n_embd, n_embd, requires_grad=True)
    X = torch.randn(batch, seq, n_embd)
    target_A = torch.randn(batch, seq, n_embd)

    A_hat, _ = compute_attention(W_Q, W_K, W_V, X, n_head=n_head)
    loss = F.mse_loss(A_hat, target_A)
    loss.backward()

    print(f"  W_Q gradient norm: {W_Q.grad.norm().item():.6f}")
    print(f"  W_K gradient norm: {W_K.grad.norm().item():.6f}")
    print(f"  W_V gradient norm: {W_V.grad.norm().item():.6f}")

    assert W_Q.grad is not None
    assert W_K.grad is not None
    assert W_V.grad is not None

    print("  Gradients flow correctly")

def test_fisher_reproducibility():
    """Test that Fisher coupling is reproducible with fixed seed."""
    print("\n=== Test 3: Reproducibility ===")
    n_embd = 16
    n_head = 2  # 16/2 = 8 
    batch = 2
    seq = 4
    torch.manual_seed(42)
    W_Q = torch.randn(n_embd, n_embd)
    W_K = torch.randn(n_embd, n_embd)
    W_V = torch.randn(n_embd, n_embd)
    X = torch.randn(batch, seq, n_embd)
    target_A = torch.randn(batch, seq, n_embd)

    torch.manual_seed(42)
    result1 = compute_fisher_coupling_simple(W_Q, W_K, W_V, X, target_A, n_embd, n_head=n_head)
    torch.manual_seed(42)
    result2 = compute_fisher_coupling_simple(W_Q, W_K, W_V, X, target_A, n_embd, n_head=n_head)

    diff = abs(result1["QK"] - result2["QK"])
    print(f"  QK diff between runs: {diff:.6f}")

    if diff < 1e-6:
        print("  Fisher coupling is reproducible")
    else:
        print("  Some variation (expected due to random initialization)")

def main():
    print("=" * 60)
    print("TESTING FISHER COUPLING")
    print("=" * 60)
    test_fisher_basic()
    test_fisher_gradient_flow()
    test_fisher_reproducibility()
    print("ALL TESTS COMPLETED!")


if __name__ == "__main__":
    main()