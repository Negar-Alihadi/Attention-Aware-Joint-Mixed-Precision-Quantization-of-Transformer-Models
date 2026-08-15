# ===== CELL 26 (markdown) =====
# ## 9. Full pipeline: uniform GPTQ baseline
# Quantizes every block's `c_attn` to a flat 4 bits -- the R7 comparison point for adaptive allocation. Both this and Section 10 reuse the *same* calibration batches, so the comparison is apples-to-apples.
#
# **Note (adopted from Untitled10.ipynb's `gptq_core.py`):** like that notebook's `quantize_all_attention_blocks`, each block's Hessian is collected and quantized in sequence on the *same* live model object, so later blocks' calibration activations already reflect earlier blocks' quantization error -- this is not fully independent per-block quantization. The next cell also reuses Untitled10's saved GPTQ checkpoint (`week1_gptq_qkv_init.pt`) when present, instead of always re-running GPTQ from scratch.

# ===== CELL 27 (code) =====
import os

print("Building the shared calibration set (used for all experiments below)...")
calibration_batches = build_calibration_batches(tokenizer, n_samples=128, seq_len=512)#32,128
print(f"{len(calibration_batches)} calibration batches ready.\n")

print("Loading a fresh full-precision GPT-2 for the uniform baseline...")
model_uniform = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
model_uniform.eval()

# --- Checkpoint-first: reuse Untitled10.ipynb's saved GPTQ init if present on disk ---
# Untitled10's `save_quantized_qkv` step writes exactly this filename. Loading it here
# instead of re-running GPTQ avoids a redundant multi-minute pass when both notebooks
# are run in the same environment. NOTE: Untitled10's own `gptq_quantize_layer` does
# NOT support `group_size`/`act_order`, so a checkpoint it produced was quantized
# WITHOUT those refinements -- slightly different from what this cell's own GPTQ call
# below would produce. If you want this baseline to reflect this notebook's improved
# quantizer (group_size=128, act_order=True), delete the checkpoint file and let this
# cell regenerate it.
CHECKPOINT_PATH = "week1_gptq_qkv_init.pt"

if os.path.exists(CHECKPOINT_PATH):
    print(f"Found {CHECKPOINT_PATH} -- loading GPTQ-quantized Q/K/V instead of re-running GPTQ...")
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    for idx, block in enumerate(model_uniform.transformer.h):
        saved = state[f"block_{idx}"]
        W_cat = torch.cat([saved["W_Q"], saved["W_K"], saved["W_V"]], dim=1).to(DEVICE)
        block.attn.c_attn.weight.data.copy_(W_cat)
    print("Loaded.\n")
else:
    print(f"No {CHECKPOINT_PATH} found -- quantizing ALL blocks to 4 bits uniformly...")
    for idx in range(len(model_uniform.transformer.h)):
        block = model_uniform.transformer.h[idx]
        c_attn = block.attn.c_attn
        print(f"  Block {idx}: quantizing to 4 bits...")
        H = collect_hessian_via_hook(model_uniform, c_attn, calibration_batches, DEVICE)
        gptq_quantize_layer(c_attn.weight.data, H, bits=4, group_size=128, act_order=True)

    # Save so this pass doesn't need to be repeated -- same format as Untitled10's Step 4,
    # so either notebook can load either notebook's checkpoint going forward.
    save_state = {}
    for idx, block in enumerate(model_uniform.transformer.h):
        W_Q, W_K, W_V = block.attn.c_attn.weight.data.split(n_embd, dim=1)
        save_state[f"block_{idx}"] = {
            "W_Q": W_Q.clone().cpu(), "W_K": W_K.clone().cpu(), "W_V": W_V.clone().cpu()
        }
    torch.save(save_state, CHECKPOINT_PATH)
    print(f"Saved GPTQ-initialized Q/K/V to {CHECKPOINT_PATH} for reuse.\n")

print("\nEvaluating perplexity...")
ppl_uniform = evaluate_perplexity(model_uniform, tokenizer)
print(f"\nUniform 4-bit perplexity: {ppl_uniform:.3f}")
