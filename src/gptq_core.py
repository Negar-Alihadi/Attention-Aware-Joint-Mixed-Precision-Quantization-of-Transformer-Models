"""
GPTQ core implementation from the notebook.
A minimal, from-scratch implementation of the GPTQ per-layer quantization
algorithm (Frantar et al., 2023), written to be simple to read and adapt.
"""
import torch

def collect_hessian_via_hook(model: torch.nn.Module, module: torch.nn.Module,
                              calibration_batches, device) -> torch.Tensor:
    """
    Registers a forward pre-hook on `module` (e.g. one block's attn.c_attn)
    to capture its input activations, runs `calibration_batches` through the
    *whole model* in no_grad mode, and returns the accumulated Hessian
    H = 2 X^T X for that layer.

    `calibration_batches` should be an iterable of input_ids tensors of shape
    (batch, seq_len), already on `device`.

    Returns: H, a (d_in, d_in) double-precision tensor.
    """
    d_in = module.weight.shape[0]  # Conv1D weight is (in_features, out_features)
    H = torch.zeros(d_in, d_in, dtype=torch.float64, device=device)
    n_samples = [0]

    def _hook(mod, inputs):
        x = inputs[0].detach()
        x = x.reshape(-1, x.shape[-1]).to(torch.float64)  # (tokens, d_in)
        H.add_(2.0 * x.T @ x)
        n_samples[0] += x.shape[0]

    handle = module.register_forward_pre_hook(_hook)
    try:
        model.eval()
        with torch.no_grad():
            for input_ids in calibration_batches:
                model(input_ids.to(device))
    finally:
        handle.remove()

    if n_samples[0] > 0:
        H /= n_samples[0]
    return H


def _quantize_to_grid(w_col: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Symmetric per-output-row fake quantization of a single input-column
    (shape: d_out) using a fixed per-row scale (shape: d_out) computed
    up front from the original weight statistics.
    """
    qmax = 2 ** (bits - 1) - 1
    q = torch.clamp(torch.round(w_col / scale), -qmax, qmax)
    return q * scale


@torch.no_grad()
def gptq_quantize_layer(weight_in_out: torch.Tensor, H: torch.Tensor, bits: int = 4,
                         damp_percent: float = 0.01) -> torch.Tensor:
    """
    Quantizes a weight matrix in the (d_in, d_out) "Conv1D" convention
    (GPT-2 style: forward is x @ weight) using the GPTQ algorithm.

    weight_in_out: (d_in, d_out) float tensor -- e.g. c_attn.weight.data
    H: (d_in, d_in) Hessian from collect_hessian_via_hook
    bits: target bit-width
    damp_percent: Hessian damping factor for numerical stability (GPTQ default ~0.01)

    Returns the fake-quantized weight, same shape, and also writes it into
    weight_in_out in place.
    """
    device = weight_in_out.device
    # Work in the (d_out, d_in) "row = output channel" convention internally,
    # since GPTQ quantizes input-columns one at a time.
    W = weight_in_out.detach().clone().to(torch.float64).T.contiguous()  # (d_out, d_in)
    d_out, d_in = W.shape

    # Fixed per-output-row scale, computed once from the original weights.
    qmax = 2 ** (bits - 1) - 1
    scale = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)  # (d_out, 1)

    # Damp and invert the Hessian.
    H = H.clone()
    mean_diag = H.diagonal().mean()
    H += damp_percent * mean_diag * torch.eye(d_in, dtype=torch.float64, device=device)
    H_inv = torch.linalg.inv(H)  # (d_in, d_in)

    for i in range(d_in):
        w_col = W[:, i]
        q_col = _quantize_to_grid(w_col, scale.squeeze(1), bits)
        err = (w_col - q_col) / H_inv[i, i]
        if i + 1 < d_in:
            W[:, i + 1:] -= torch.outer(err, H_inv[i, i + 1:])
        W[:, i] = q_col

    W_final = W.T.contiguous().to(weight_in_out.dtype)  # back to (d_in, d_out)
    weight_in_out.copy_(W_final)
    return W_final