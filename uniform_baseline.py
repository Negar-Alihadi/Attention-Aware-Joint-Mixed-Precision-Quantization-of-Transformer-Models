"""
Run uniform 4-bit quantization for comparison.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.gptq_core import gptq_quantize_layer, collect_hessian_via_hook
#from apply_allocation_to_pipeline import build_calibration_batches, evaluate_perplexity
from src.utils import build_calibration_batches, evaluate_perplexity

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

print("Creating calibration data...")
calibration_batches = build_calibration_batches(tokenizer, n_samples=128, seq_len=512)

print("\nQuantizing ALL blocks to 6 bits uniformly...")
for idx in range(len(model.transformer.h)):
    block = model.transformer.h[idx]
    c_attn = block.attn.c_attn
    
    print(f"  Block {idx}: quantizing to 6 bits...")
    H = collect_hessian_via_hook(model, c_attn, calibration_batches, device)
    gptq_quantize_layer(c_attn.weight.data, H, bits=6)

print("\nEvaluating perplexity...")
ppl = evaluate_perplexity(model, tokenizer)
print(f"\nUniform 6-bit Perplexity: {ppl:.2f}")