"""
Test attention_loss.py with random data.
"""
import torch
from src.attention_loss import (
    reshape_weights,
    compute_attention,
    mse_loss,
    kl_loss,
    attention_loss
)

def test_reshape_weights():
    print("\n=== Test 1: Reshape Weights ===")
    n_embd = 768
    total = 3 * n_embd * n_embd    
    # Create random flat weights
    w_flat = torch.randn(total)    
    # Reshape
    W_Q, W_K, W_V = reshape_weights(w_flat, n_embd)
    # Check shapes
    assert W_Q.shape == (n_embd, n_embd)
    assert W_K.shape == (n_embd, n_embd)
    assert W_V.shape == (n_embd, n_embd)
    # Check we can reconstruct
    reconstructed = torch.cat([W_Q.flatten(), W_K.flatten(), W_V.flatten()])
    assert torch.allclose(w_flat, reconstructed, atol=1e-6)    
    print("Reshape works correctly")
    print(f"   Input shape: {w_flat.shape}")
    print(f"   W_Q shape: {W_Q.shape}")

def test_compute_attention():
    print("\n=== Test 2: Compute Attention ===")
    n_embd = 768
    batch = 2
    seq = 4
    # Create random inputs
    W_Q = torch.randn(n_embd, n_embd)
    W_K = torch.randn(n_embd, n_embd)
    W_V = torch.randn(n_embd, n_embd)
    X = torch.randn(batch, seq, n_embd)
    # Compute attention
    A_hat, attn_weights = compute_attention(W_Q, W_K, W_V, X)
    # Check shapes
    assert A_hat.shape == (batch, seq, n_embd)
    assert attn_weights.shape == (batch, seq, seq)
    # Check attention weights sum to 1 (per row)
    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch, seq), atol=1e-5)
    print("Attention computation works")
    print(f"   A_hat shape: {A_hat.shape}")
    print(f"   Attn weights sum: {sums[0, 0].item():.6f}")

def test_mse_loss():
    print("\n=== Test 3: MSE Loss ===")
    n_embd = 768
    batch = 2
    seq = 4
    # Create random data
    w_flat = torch.randn(3 * n_embd * n_embd)
    X = torch.randn(batch, seq, n_embd)
    target_A = torch.randn(batch, seq, n_embd)
    # Compute loss
    loss = attention_loss(w_flat, X, target_A, n_embd)
    # Check it's a scalar and non-negative
    assert loss.shape == ()  # Scalar
    assert loss.item() >= 0  # Non-negative
    print(f"MSE loss works")
    print(f"   Loss: {loss.item():.6f}")

def test_gradients():
    print("\n=== Test 4: Gradient Flow ===")
    n_embd = 768
    batch = 2
    seq = 4
    # Create data with requires_grad
    w_flat = torch.randn(3 * n_embd * n_embd, requires_grad=True)
    X = torch.randn(batch, seq, n_embd)
    target_A = torch.randn(batch, seq, n_embd)
    # Forward pass
    loss = attention_loss(w_flat, X, target_A, n_embd)
    # Backward pass
    loss.backward()
    # Check gradients
    assert w_flat.grad is not None
    assert w_flat.grad.shape == w_flat.shape
    assert torch.any(w_flat.grad != 0)  # Non-zero gradients
    print(f"Gradients flow correctly")
    print(f"   Gradient shape: {w_flat.grad.shape}")
    print(f"   Gradient norm: {w_flat.grad.norm().item():.6f}")

def test_kl_loss():
    print("\n=== Test 5: KL Divergence Loss ===")
    n_embd = 768
    batch = 2
    seq = 4
    # Create random data
    w_flat = torch.randn(3 * n_embd * n_embd)
    X = torch.randn(batch, seq, n_embd)
    target_A = torch.randn(batch, seq, n_embd)
    # Create random target attention weights (valid probability distribution)
    target_attn = torch.softmax(torch.randn(batch, seq, seq), dim=-1)
    # MSE only
    loss_mse = attention_loss(w_flat, X, target_A, n_embd)
    # MSE + KL
    loss_combined = attention_loss(w_flat, X, target_A, n_embd, target_attn, lambda_kl=0.1)
    # Combined loss should be >= MSE loss (since KL >= 0)
    assert loss_combined.item() >= loss_mse.item()
    print(f"KL loss works")
    print(f"   MSE only: {loss_mse.item():.6f}")
    print(f"   MSE + KL: {loss_combined.item():.6f}")

def test_different_shapes():
    print("\n=== Test 6: Different Shapes ===")    
    n_embd = 768
    # Test different batch sizes
    for batch in [1, 2, 4]:
        seq = 4
        w_flat = torch.randn(3 * n_embd * n_embd)
        X = torch.randn(batch, seq, n_embd)
        target_A = torch.randn(batch, seq, n_embd)
        
        loss = attention_loss(w_flat, X, target_A, n_embd)
        print(f"   Batch {batch}: loss = {loss.item():.6f}")
    # Test different sequence lengths
    for seq in [1, 4, 8]:
        batch = 2
        w_flat = torch.randn(3 * n_embd * n_embd)
        X = torch.randn(batch, seq, n_embd)
        target_A = torch.randn(batch, seq, n_embd)
        
        loss = attention_loss(w_flat, X, target_A, n_embd)
        print(f"   Seq {seq}: loss = {loss.item():.6f}")
    
    print("Works with different shapes")

def test_numerical_stability():
    print("\n=== Test 7: Numerical Stability ===")    
    n_embd = 768
    batch = 2
    seq = 4
    # Test with large values
    w_flat = torch.randn(3 * n_embd * n_embd) * 1000
    X = torch.randn(batch, seq, n_embd) * 1000
    target_A = torch.randn(batch, seq, n_embd) * 1000
    loss = attention_loss(w_flat, X, target_A, n_embd)
    # Should not be NaN or Inf
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    print(f"Stable with extreme values: {loss.item():.6f}")

def test_attention_loss_consistency():
    print("\n=== Test 8: Loss Consistency ===")    
    n_embd = 768
    batch = 2
    seq = 4
    w_flat = torch.randn(3 * n_embd * n_embd)
    X = torch.randn(batch, seq, n_embd)
    target_A = torch.randn(batch, seq, n_embd)
    # Method 1: Direct mse_loss
    loss1 = mse_loss(w_flat, X, target_A, n_embd)
    # Method 2: attention_loss without KL
    loss2 = attention_loss(w_flat, X, target_A, n_embd)
    # Method 3: attention_loss with target_attn=None
    loss3 = attention_loss(w_flat, X, target_A, n_embd, target_attn=None)
    assert torch.allclose(loss1, loss2, atol=1e-6)
    assert torch.allclose(loss1, loss3, atol=1e-6)
    print(f"All methods give consistent results")
    print(f"   mse_loss: {loss1.item():.6f}")
    print(f"   attention_loss: {loss2.item():.6f}")

if __name__ == "__main__":
    print("TESTING ATTENTION_LOSS.PY (No GPT-2 Required)")
    
    test_reshape_weights()
    test_compute_attention()
    test_mse_loss()
    test_gradients()
    test_kl_loss()
    test_different_shapes()
    test_numerical_stability()
    test_attention_loss_consistency()    
    print("ALL TESTS PASSED!")
    print("\nThe loss function is ready to use with GPT-2.")