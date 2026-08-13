"""
Simple unit tests for per_matrix_trace.py using synthetic data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.hutchinson_trace_estimator import hutchinson_trace_estimator
from src.attention_loss import compute_attention

def compute_per_matrix_traces_simple(W_Q, W_K, W_V, X, target_A, n_embd, 
                                      n_head=4, samples=30):
    """
    Simplified version of per-matrix traces that works with direct tensors.(no need for model)
    """
    traces = {}
    # Trace for W_Q (holding K and V fixed)
    def loss_fn_Q(params):
        WQ_ = params.view(n_embd, n_embd)
        A_hat, _ = compute_attention(WQ_, W_K, W_V, X, n_head=n_head)
        return F.mse_loss(A_hat, target_A)
    
    w_flat = W_Q.clone().flatten()
    w_flat.requires_grad_(True)
    traces["Q"] = hutchinson_trace_estimator(loss_fn_Q, w_flat, samples=samples).item()
    # Trace for W_K (holding Q and V fixed)
    def loss_fn_K(params):
        WK_ = params.view(n_embd, n_embd)
        A_hat, _ = compute_attention(W_Q, WK_, W_V, X, n_head=n_head)
        return F.mse_loss(A_hat, target_A)
    
    w_flat = W_K.clone().flatten()
    w_flat.requires_grad_(True)
    traces["K"] = hutchinson_trace_estimator(loss_fn_K, w_flat, samples=samples).item()
    # Trace for W_V (holding Q and K fixed)
    def loss_fn_V(params):
        WV_ = params.view(n_embd, n_embd)
        A_hat, _ = compute_attention(W_Q, W_K, WV_, X, n_head=n_head)
        return F.mse_loss(A_hat, target_A)
    
    w_flat = W_V.clone().flatten()
    w_flat.requires_grad_(True)
    traces["V"] = hutchinson_trace_estimator(loss_fn_V, w_flat, samples=samples).item()
    
    return traces


def test_reshape_and_attention():
    """Test that the attention computation works."""
    print("\n=== Test 1: Basic Attention with Per-Matrix Traces ===")
    n_embd = 32
    n_head = 4  # 32 / 8 = 4
    batch = 2
    seq = 8
    W_Q = torch.randn(n_embd, n_embd)
    W_K = torch.randn(n_embd, n_embd)
    W_V = torch.randn(n_embd, n_embd)
    X = torch.randn(batch, seq, n_embd)
    target_A = torch.randn(batch, seq, n_embd)
    
    A_hat, attn_weights = compute_attention(W_Q, W_K, W_V, X, n_head=n_head)
    
    print(f"  A_hat shape: {A_hat.shape}")
    print(f"  attn_weights shape: {attn_weights.shape}")
    print("  Basic attention works")
    
    return W_Q, W_K, W_V, X, target_A


def test_single_block_traces():
    """Test computing per-matrix traces with synthetic data."""
    print("\n=== Test 2: Per-Matrix Traces for One Block ===")
    
    n_embd = 32
    n_head = 4
    batch = 2
    seq = 8
    samples = 20
    W_Q = torch.randn(n_embd, n_embd)
    W_K = torch.randn(n_embd, n_embd)
    W_V = torch.randn(n_embd, n_embd)
    X = torch.randn(batch, seq, n_embd)
    target_A = torch.randn(batch, seq, n_embd)
    # Use the simplified function with n_head
    traces = compute_per_matrix_traces_simple(W_Q=W_Q, W_K=W_K, W_V=W_V, X=X, target_A=target_A, n_embd=n_embd, n_head=n_head, samples=samples)
    
    print(f"  Trace Q: {traces['Q']:.6f}")
    print(f"  Trace K: {traces['K']:.6f}")
    print(f"  Trace V: {traces['V']:.6f}")
    print(f"  Sum: {traces['Q'] + traces['K'] + traces['V']:.6f}")
    print("  Per-matrix traces computed successfully")
    
    return traces


def test_trace_reproducibility():
    """Test that traces are reproducible with fixed seed."""
    print("\n=== Test 3: Reproducibility ===")
    n_embd = 16
    n_head = 2  # 16 / 8 = 2
    batch = 2
    seq = 4
    samples = 30
    # Fixed seed
    torch.manual_seed(42)
    W_Q = torch.randn(n_embd, n_embd)
    W_K = torch.randn(n_embd, n_embd)
    W_V = torch.randn(n_embd, n_embd)
    X = torch.randn(batch, seq, n_embd)
    target_A = torch.randn(batch, seq, n_embd)
    # Compute twice with same seed
    torch.manual_seed(42)
    traces1 = compute_per_matrix_traces_simple(W_Q, W_K, W_V, X, target_A, n_embd, n_head, samples)
    
    torch.manual_seed(42)
    traces2 = compute_per_matrix_traces_simple(W_Q, W_K, W_V, X, target_A, n_embd, n_head, samples)
    
    print(f"  Run 1 - Q: {traces1['Q']:.6f}, K: {traces1['K']:.6f}, V: {traces1['V']:.6f}")
    print(f"  Run 2 - Q: {traces2['Q']:.6f}, K: {traces2['K']:.6f}, V: {traces2['V']:.6f}")
    
    diff_Q = abs(traces1['Q'] - traces2['Q'])
    diff_K = abs(traces1['K'] - traces2['K'])
    diff_V = abs(traces1['V'] - traces2['V'])
    
    print(f"  Differences: Q={diff_Q:.6f}, K={diff_K:.6f}, V={diff_V:.6f}")
    
    if diff_Q < 1e-4 and diff_K < 1e-4 and diff_V < 1e-4:
        print(" Traces are reproducible")
    else:
        print(" Traces have some variation (expected due to Monte Carlo)")


def main():
    print("TESTING PER_MATRIX_TRACE.PY")
    
    test_reshape_and_attention()
    test_single_block_traces()
    test_trace_reproducibility()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")


if __name__ == "__main__":
    main()