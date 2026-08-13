"""
Helper functions for calibration and evaluation.
"""

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def build_calibration_batches(tokenizer, n_samples=128, seq_len=512):
    """
    Pulls `n_samples` chunks of `seq_len` tokens each from WikiText-2 train,
    as a list of (1, seq_len) input_id tensors.
    """
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in raw["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]

    batches = []
    stride = seq_len
    for i in range(n_samples):
        start = i * stride
        if start + seq_len > ids.shape[0]:
            break
        chunk = ids[start:start + seq_len].unsqueeze(0)
        batches.append(chunk)
    return batches


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, max_length=1024, stride=512):
    """
    Sliding-window perplexity on WikiText-2 test.
    """
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in raw["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    seq_len = ids.shape[1]

    model.eval()
    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        input_ids = ids[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        out = model(input_ids, labels=target_ids)
        nll_sum += out.loss.item() * trg_len
        n_tokens += trg_len

        prev_end = end
        if end == seq_len:
            break

    return math.exp(nll_sum / n_tokens)