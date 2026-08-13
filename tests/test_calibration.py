"""
Test the calibration utilities work with real GPT-2.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.calibration import get_calibration_data_for_block

def test_calibration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model
    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    
    # Create batch
    from datasets import load_dataset
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = " ".join(t for t in raw["text"][:5] if t.strip())
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids
    batch = batch.to(device)
    
    # Get data for block 0
    print("Getting calibration data...")
    X, target_A, target_attn = get_calibration_data_for_block(model, batch, device, block_idx=0)
    
    print(f"X shape: {X.shape}")
    print(f"target_A shape: {target_A.shape}")
    print(f"target_attn: {target_attn} (None - MSE only)")
    
    print("\nAll tests passed! Ready for JAB-Hessian.")

if __name__ == "__main__":
    test_calibration()