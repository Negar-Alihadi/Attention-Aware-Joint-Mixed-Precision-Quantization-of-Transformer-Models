"""
Capturing calibration data from GPT-2
1. AttentionOutputCapture (for MSE loss)
2. ActivationCapture (for X)
3. Combined function to get all data
"""
import torch

class AttentionOutputCapture:
    """
    Captures attention output A(X) from each block, used for MSE loss.
    """
    def __init__(self, model, detach=True):
        self.outputs = {}
        self.handles = []
        self.detach = detach

        for idx, block in enumerate(model.transformer.h):
            # Hook before c_proj (this is A(X) before output projection)
            handle = block.attn.c_proj.register_forward_pre_hook(self._make_hook(idx))
            self.handles.append(handle)

    def _make_hook(self, idx):
        def hook(module, inputs):
            # input[0] is A(X) - the attention output
            if self.detach:
                self.outputs[idx] = inputs[0].detach().clone()
            else:
                self.outputs[idx] = inputs[0]  # ← Keep gradients!
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


class ActivationCapture:
    """
    Captures input activations (X) to each block's attention.
    """
    def __init__(self, model, detach=True):
        self.activations = {}
        self.handles = []
        self.detach = detach

        for idx, block in enumerate(model.transformer.h):
            # Hook before c_attn (this is X, the input to attention)
            handle = block.attn.c_attn.register_forward_pre_hook(self._make_hook(idx))
            self.handles.append(handle)

    def _make_hook(self, idx):
        def hook(module, input):
            # input[0] is X
            if self.detach:
                self.activations[idx] = input[0].detach().clone()
            else:
                self.activations[idx] = input[0]  # ← Keep gradients!
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def get_calibration_data(model, calibration_batch, device, with_grad=False):
    """
    Get all calibration data for one forward pass.
    Returns X, target_A, target_attn.

    # ===== CHANGED (KL divergence) =====
    # target_attn used to be hard-coded to None ("MSE only for now"). It is
    # now populated with each block's REAL post-softmax attention-weight
    # matrix, obtained via output_attentions=True on this same forward pass
    # (no extra forward pass needed). This is what kl_loss/attention_loss's
    # target_attn argument expects, so the KL term defined in Section 3 can
    # actually be used instead of always silently falling back to MSE-only.
    # NOTE: requires the model to have been loaded with
    # attn_implementation="eager" (sdpa/flash attention do not expose
    # attention-weight tensors).
    """
    # Create captures
    act_capture = ActivationCapture(model, detach=not with_grad)
    out_capture = AttentionOutputCapture(model, detach=not with_grad)

    # Run model
    model.eval()
    if with_grad:
        # Enable gradients for Fisher coupling
        with torch.enable_grad():
            outputs = model(calibration_batch.to(device), output_attentions=True)
            # Keep gradients for target_attn
            target_attn = {idx: attn for idx, attn in enumerate(outputs.attentions)}
    else:
        # Default: no gradients (for Hessian trace)
        with torch.no_grad():
            outputs = model(calibration_batch.to(device), output_attentions=True)
            target_attn = {idx: attn.detach().clone() for idx, attn in enumerate(outputs.attentions)}

    # Get data
    X = act_capture.activations
    target_A = out_capture.outputs
    # ===== CHANGED: target_attn is now real data, not None =====
    # outputs.attentions: tuple of (batch, n_head, T, T) tensors, one per block,
    # already post-softmax -- same quantity compute_attention returns as attn_weights.
    #target_attn = {idx: attn.detach().clone() for idx, attn in enumerate(outputs.attentions)}

    # Clean up
    act_capture.remove()
    out_capture.remove()

    return X, target_A, target_attn


def get_calibration_data_for_block(model, calibration_batch, device, block_idx, with_grad=False):
    """
    Get calibration data for a specific block.
    """
    X_dict, target_A_dict, target_attn_dict = get_calibration_data(
        model, calibration_batch, device, with_grad=with_grad
    )

    return (
        X_dict[block_idx],
        target_A_dict[block_idx],
        target_attn_dict[block_idx],  # ===== CHANGED: was hard-coded None =====
    )