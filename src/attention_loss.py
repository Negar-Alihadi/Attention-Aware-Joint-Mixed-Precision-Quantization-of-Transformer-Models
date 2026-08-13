"""
Shared loss function for attention-aware quantization.
Combined Loss Function: L = ||A(X) - Â(X)||_F^2 + λ·KL(attention_maps)
"""
import torch
import torch.nn.functional as F
import math

def reshape_weights(w_flat, n_embd):
    """
    Convert flattened weights back to Q, K, V matrices.
    Inputs:
        w_flat: Flattened [W_Q, W_K, W_V]
        n_embd: Model dimension (768 for GPT-2 small)
    Outputs:
        W_Q, W_K, W_V: Each of shape (n_embd, n_embd)
    """
    # Each matrix has n_embd * n_embd parameters
    size = n_embd * n_embd
    
    # Split into 3 parts
    W_Q = w_flat[0:size].reshape(n_embd, n_embd)
    W_K = w_flat[size:2*size].reshape(n_embd, n_embd)
    W_V = w_flat[2*size:3*size].reshape(n_embd, n_embd)
    
    return W_Q, W_K, W_V

def compute_attention(W_Q, W_K, W_V, X, n_head = 12, b_Q=None, b_K=None, b_V=None):
    """
    Compute attention output and attention weights.
    Inputs:
        W_Q, W_K, W_V: Weight matrices (n_embd, n_embd)
        X: Input activations (batch, seq_len, n_embd)
    Outputs:
        A_hat: Attention output (batch, seq_len, n_embd)
        attn_weights: Attention weights (batch, seq_len, seq_len)
    """
    B, T, n_embd = X.shape #batch size, sequence length, 768
    d_head = n_embd // n_head #per-head dimension
    # Compute Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    if b_Q is not None:
        Q = Q + b_Q
        K = K + b_K
        V = V + b_V

    Q = Q.view(B, T, n_head, d_head).transpose(1, 2)
    K = K.view(B, T, n_head, d_head).transpose(1, 2)
    V = V.view(B, T, n_head, d_head).transpose(1, 2)
    # Scaled dot-product attention
    #d_k = Q.shape[-1]
    #scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=Q.device))
    scale = math.sqrt(d_head)    
    scores = Q @ K.transpose(-2, -1) / scale
    
    causal_mask = torch.tril(torch.ones(T, T, device=X.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    # Attention weights 
    attn_weights = torch.softmax(scores, dim=-1)
    # Attention output (weighted sum of values)
    #A_hat = attn_weights @ V
    A_hat = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, n_embd)

    return A_hat, attn_weights

def mse_loss(w_flat, X, target_A, n_embd, b_Q=None, b_K=None, b_V=None):
    """
    L_mse = ||A(X) - A_hat(X)||^2
    This is the primary loss function.
    """
    # Reshape and compute attention
    W_Q, W_K, W_V = reshape_weights(w_flat, n_embd)
    A_hat, _ = compute_attention(W_Q, W_K, W_V, X, b_Q=b_Q, b_K=b_K, b_V=b_V)
    # MSE loss
    return F.mse_loss(A_hat, target_A)

def kl_loss(w_flat, X, target_attn, n_embd, b_Q=None, b_K=None, b_V=None):
    """
    L_kl = KL(attention_weights || target_attention_weights)    
    """
    # Reshape and compute attention
    W_Q, W_K, W_V = reshape_weights(w_flat, n_embd)
    _, attn_weights = compute_attention(W_Q, W_K, W_V, X, b_Q=b_Q, b_K=b_K, b_V=b_V)
    # KL divergence: KL(P || Q) = sum(P * log(P / Q))
    # P = target_attn, Q = attn_weights
    log_q = torch.log(attn_weights + 1e-8)  # Add epsilon for stability
    
    return F.kl_div(log_q, target_attn, reduction='batchmean') #batchmean = sum(KL) / batch_size (like in Q-BERT & APTQ)

def attention_loss(w_flat, X, target_A, n_embd, target_attn=None, lambda_kl=0.1, b_Q=None, b_K=None, b_V=None):
    """
    If target_attn is None: Returns MSE only O.W. :Returns MSE + lambda_kl * KL
    """
    # Always compute MSE
    loss = mse_loss(w_flat, X, target_A, n_embd, b_Q=b_Q, b_K=b_K, b_V=b_V)
    # Add KL if target attention weights are provided
    if target_attn is not None:
        kl = kl_loss(w_flat, X, target_attn, n_embd, b_Q=b_Q, b_K=b_K, b_V=b_V)
        loss = loss + lambda_kl * kl
    
    return loss


if __name__ == "__main__":
    print("Testing attention_loss.py...")
    n_embd = 768
    n_head = 12
    batch = 2
    seq = 4
    # Create test data
    w_flat = torch.randn(3 * n_embd * n_embd)
    X = torch.randn(batch, seq, n_embd)
    target_A = torch.randn(batch, seq, n_embd)
    # Test MSE only
    loss1 = attention_loss(w_flat, X, target_A, n_embd)
    print(f"MSE loss: {loss1.item():.6f}")
    # Test with KL
    target_attn = torch.softmax(torch.randn(batch, n_head, seq, seq), dim=-1)  # (2, 12, 4, 4)    
    loss2 = attention_loss(w_flat, X, target_A, n_embd, target_attn, lambda_kl=0.1)
    print(f"MSE + KL loss: {loss2.item():.6f}")
    # Test gradients
    w_flat.requires_grad_(True)
    loss = attention_loss(w_flat, X, target_A, n_embd)
    loss.backward()
    print(f"Gradients flow: {w_flat.grad is not None}")
    print("All tests passed!")