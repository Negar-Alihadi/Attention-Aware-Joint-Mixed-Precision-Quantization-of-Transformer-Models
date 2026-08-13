import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.calibration import get_calibration_data_for_block
from src.attention_loss import mse_loss, compute_attention, reshape_weights

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()
n_embd = model.config.n_embd

# a small real batch
text = "The quick brown fox jumps over the lazy dog. " * 20
batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)

block_idx = 0
X, target_A, _ = get_calibration_data_for_block(model, batch, device, block_idx)

# reconstruct w_flat from the REAL, unquantized weights of this block
block = model.transformer.h[block_idx]
W = block.attn.c_attn.weight.data  # (768, 2304) = [W_Q | W_K | W_V]
W_Q, W_K, W_V = W.split(n_embd, dim=1)
w_flat = torch.cat([W_Q.flatten(), W_K.flatten(), W_V.flatten()])

# reconstruct the real bias too — same split pattern as the weights
bias = block.attn.c_attn.bias.data  # (2304,)
b_Q, b_K, b_V = bias.split(n_embd)

with torch.no_grad():
    err = mse_loss(w_flat, X, target_A, n_embd, b_Q=b_Q, b_K=b_K, b_V=b_V)

    # also grab attn_weights directly, just to confirm the head split is real
    W_Q_r, W_K_r, W_V_r = reshape_weights(w_flat, n_embd)
    A_hat, attn_weights = compute_attention(W_Q_r, W_K_r, W_V_r, X, b_Q=b_Q, b_K=b_K, b_V=b_V)

mse_value = err.item()
target_scale = (target_A**2).mean().item()

print(f"attn_weights shape (expect (batch, 12, T, T)): {tuple(attn_weights.shape)}")
print(f"MSE between real attention output and compute_attention's output: {mse_value:.10f}")
print(f"Scale reference — mean squared magnitude of target_A itself: {target_scale:.10f}")
print(f"Ratio (MSE / target scale): {mse_value / target_scale:.10f}")

if mse_value < 1e-6:
    print("\nPASS — compute_attention reconstructs the real attention output almost exactly.")
else:
    print("\nFAIL — still a meaningful gap; compute_attention does not match GPT-2's real attention yet.")