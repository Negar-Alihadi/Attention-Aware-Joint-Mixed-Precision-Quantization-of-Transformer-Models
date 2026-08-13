import torch
import torch.autograd as autograd

def hessian_vector_product(loss_fn, params, vector, retain_graph = True):
    #first order grad
    grad = autograd.grad(loss_fn(params), params, create_graph=True, retain_graph=True)[0] #retain_graph = true to keep the graph for second grad, do not release it!
    #second order grad
    hvp = autograd.grad(grad, params, grad_outputs=vector, retain_graph=True)[0]
    return hvp

def hutchinson_trace_estimator(loss_fn, params, samples=50): #number of iterations mentioned in HAWQ-V2 article
    #trace(H) ≈ (1/n) * Σ(v_i^T * H * v_i)
    if not params.requires_grad:
        params.requires_grad_(True)
    
    device = params.device
    estimated_trace = 0.0
    
    for _ in range(samples):
        #Rademacher Vector(mentioned in HAWQ-V2)
        vec = torch.randint(0, 2, params.shape, device=device) * 2 -1 
        vec = vec.float()
        hvp_result = hessian_vector_product(loss_fn, params, vec)
        estimated_trace += torch.dot(vec.flatten(), hvp_result.flatten())
    
    return estimated_trace/samples 