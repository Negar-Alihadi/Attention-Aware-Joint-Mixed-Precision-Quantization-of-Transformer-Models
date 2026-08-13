"""
Apply adaptive bit allocation to GPT-2 (gptq_code pipeline).
"""
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.jab_hessian import run_jab_allocation
from src.utils import build_calibration_batches, evaluate_perplexity
from src.gptq_core import gptq_quantize_layer, collect_hessian_via_hook
""""
def build_calibration_batches(tokenizer, n_samples=32, seq_len=64):
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in raw["text"][:50] if t.strip())
    ids = tokenizer(text, return_tensors="pt", truncation=True).input_ids[0]
    batches = []
    for i in range(n_samples):
        start = i * seq_len
        if start + seq_len > ids.shape[0]:
            break
        batches.append(ids[start:start + seq_len].unsqueeze(0))
    return batches
"""
"""
def evaluate_perplexity(model, tokenizer):
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in raw["text"] if t.strip())
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    words = text.split()    # Process in chunks
    for i in range(0, len(words), 512):
        chunk = " ".join(words[i:i+512])
        if not chunk.strip():
            continue
        
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs.input_ids.to(model.device)
        
        with torch.no_grad():
            output = model(input_ids, labels=input_ids)
        
        total_loss += output.loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)
    
    return math.exp(total_loss / total_tokens)
    """

def apply_allocation(model, assignment, calibration_batches, device):
    print("\nApplying allocation")
    
    for idx in range(len(model.transformer.h)):
        block_name = f"block_{idx}_QKV"
        if block_name in assignment:
            bits = assignment[block_name]
            block = model.transformer.h[idx]
            c_attn = block.attn.c_attn
            
            H = collect_hessian_via_hook(model, c_attn, calibration_batches, device)# Quantize this block
            gptq_quantize_layer(c_attn.weight.data, H, bits=bits)
            
            print(f"Block {idx}: {bits} bits")
    
    print("Allocation is Done!")
    return model

if __name__ == "__main__":
    print("Adaptive Allocation for GPT-2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("\nLoading GPT-2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    n_embd = model.config.n_embd
    print(f"Dimension: {n_embd}")
    
    print("\nCreating calibration data")
    calibration_batches = build_calibration_batches(tokenizer, n_samples=128, seq_len=512) # n_samples=32, seq_len=64
    print(f"{len(calibration_batches)} batches")
    
    print("\nGetting JAB allocation")
    torch.manual_seed(42)
    assignment = run_jab_allocation(model=model, batches=calibration_batches, device=device, n_embd=n_embd, target_avg_bits=6.0, samples=20)
    
    print("\nAllocation Details:")
    for block, bits in assignment.items():
        print(f"{block}: {bits} bits")
    
    print("\nQuantizing model")
    model = apply_allocation(model, assignment, calibration_batches, device)
    
    print("\nEvaluating perplexity")
    ppl = evaluate_perplexity(model, tokenizer)
    print(f"\nPerplexity: {ppl:.2f}")
    
    print("\nDone!")