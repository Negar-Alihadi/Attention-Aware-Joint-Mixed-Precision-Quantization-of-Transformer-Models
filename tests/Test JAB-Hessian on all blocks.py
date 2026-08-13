"""
Test JAB-Hessian on all blocks.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.jab_hessian import compute_all_jab_scores

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()
n_embd = model.config.n_embd

raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train") # Create calibration batches
text = " ".join(t for t in raw["text"][:10] if t.strip())
batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids
batch = batch.to(device)
batches = [batch]
print("\nComputing JAB scores for ALL blocks...")
scores = compute_all_jab_scores(model=model, batches=batches, device=device, n_embd=n_embd, samples=20)
print("\nAll scores:")
for block, score in scores.items():
    print(f"  {block}: {score:.4f}")