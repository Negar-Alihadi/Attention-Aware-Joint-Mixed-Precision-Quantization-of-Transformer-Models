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
    def __init__(self, model):
        self.outputs = {}
        self.handles = []
        
        for idx, block in enumerate(model.transformer.h):
            # Hook before c_proj (this is A(X) before output projection)
            handle = block.attn.c_proj.register_forward_pre_hook(self._make_hook(idx))
            self.handles.append(handle)
    
    def _make_hook(self, idx):
        def hook(module, inputs):
            # input[0] is A(X) - the attention output
            self.outputs[idx] = inputs[0].detach().clone()
        return hook
    
    def remove(self):
        for h in self.handles:
            h.remove()


class ActivationCapture:
    """
    Captures input activations (X) to each block's attention.
    """
    def __init__(self, model):
        self.activations = {}
        self.handles = []
        
        for idx, block in enumerate(model.transformer.h):
            # Hook before c_attn (this is X, the input to attention)
            handle = block.attn.c_attn.register_forward_pre_hook(self._make_hook(idx))
            self.handles.append(handle)
    
    def _make_hook(self, idx):
        def hook(module, input):
            # input[0] is X
            self.activations[idx] = input[0].detach().clone()
        return hook
    
    def remove(self):
        for h in self.handles:
            h.remove()


def get_calibration_data(model, calibration_batch, device):
    """
    Get all calibration data for one forward pass.
    Returns X, target_A, target_attn (None for now - MSE only).
    """
    # Create captures
    act_capture = ActivationCapture(model)
    out_capture = AttentionOutputCapture(model)
    
    # Run model
    model.eval()
    with torch.no_grad():
        model(calibration_batch.to(device))
    
    # Get data
    X = act_capture.activations
    target_A = out_capture.outputs
    target_attn = None  # MSE only for now (KL can be added later)
    
    # Clean up
    act_capture.remove()
    out_capture.remove()
    
    return X, target_A, target_attn


def get_calibration_data_for_block(model, calibration_batch, device, block_idx):
    """
    Get calibration data for a specific block.
    """
    X_dict, target_A_dict, target_attn_dict = get_calibration_data(
        model, calibration_batch, device
    )
    
    return (
        X_dict[block_idx],
        target_A_dict[block_idx],
        None  # This will be None #target_attn_dict[block_idx]
    )