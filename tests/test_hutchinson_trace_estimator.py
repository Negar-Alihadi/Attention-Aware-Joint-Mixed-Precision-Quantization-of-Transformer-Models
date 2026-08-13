import torch
import torch.nn as nn
from torch.autograd.functional import hessian
from src.hutchinson_trace_estimator import hutchinson_trace_estimator

def test_hutchinson():
    
    #toy layer
    layer = nn.Linear(5, 3)
    x = torch.randn(4, 5) #input data
    y = torch.randn(4, 3) #target data
        
    weights = list(layer.parameters())[0].clone()
    weights_flat = weights.flatten()
    weights_flat.requires_grad_(True)
    print(f"shape of weights: {weights.shape}")
    print(f"number of weights: {weights.numel()}")
    
    def loss_fn(w):
        w_reshaped = w.reshape(3, 5)
        output = torch.matmul(x, w_reshaped.T) + layer.bias
        return nn.MSELoss()(output, y)
    
    test_loss = loss_fn(weights_flat)
    print(f"Test loss: {test_loss.item():.6f}")
    
    H_matrix = hessian(loss_fn, weights_flat)
    H_2d = H_matrix.reshape(weights_flat.numel(), weights_flat.numel())
    exact_trace = torch.trace(H_2d)
    print(f"exact trace: {exact_trace:0.6f}")
    
    hutchinson_trace = hutchinson_trace_estimator(loss_fn=loss_fn, params=weights_flat, samples=100)
    print(f"Hutchinson estimated trace: {hutchinson_trace:0.6f}") 
    
    print("\n3. Comparing results...")
    error = abs(hutchinson_trace - exact_trace)
    print(f"Exact trace:      {exact_trace:0.6f}")
    print(f"Hutchinson:       {hutchinson_trace:0.6f}")
    print(f"Error:            {error:0.6f}")
    
    if error < 0.1:
        print("Test passed!")
    else:
        print(f"Error too large: {error:0.6f}")

if __name__ == "__main__":
    test_hutchinson()