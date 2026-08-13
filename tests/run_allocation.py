"""
Run JAB-Hessian adaptive allocation on GPT-2 small.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.jab_hessian import run_jab_allocation

print("JAB-Hessian Adaptive Allocation")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("\nLoading model(GPT-2)")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()
n_embd = model.config.n_embd
print(f"Model dimension: {n_embd}")

print("\nCreating calibration batches")
raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
text = " ".join(t for t in raw["text"][:10] if t.strip())
batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids
batch = batch.to(device)
batches = [batch]
print(f"Batch shape: {batch.shape}")

print("\nRunning allocation for different budgets")
for target_avg_bits in [3.0, 4.0, 6.0, 8.0]:
    print(f"\nTarget average: {target_avg_bits} bits")    
    assignment = run_jab_allocation(model=model, batches=batches, device=device, n_embd=n_embd, target_avg_bits=target_avg_bits, samples=20)
    bit_counts = {} # which blocks get which bits
    for block, bits in assignment.items():
        bit_counts[bits] = bit_counts.get(bits, 0) + 1
    print(f"\nBit distribution:")
    for bits, count in sorted(bit_counts.items()):
        print(f"{bits} bits: {count} blocks")
