import torch
import numpy as np
import torch.nn as nn
from torch.autograd.functional import jacobian


def grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )

def multi_output_grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input.
    Args:
        outputs: (N, D) tensor
        inputs: (N, D) tensor
    """
    for i in range(outputs.shape[1]):
        grads = grad(outputs[:, i].reshape(-1, 1), inputs)[0]
        if i == 0:
            all_grads = grads
        else:
            all_grads = torch.cat([all_grads, grads], dim=1)
    return all_grads


if __name__ == '__main__':
    
    torch.random.manual_seed(42)
    net = nn.Sequential(
        nn.Linear(1, 15),
        nn.Softplus(),
        nn.Linear(15, 15),
        nn.Softplus(),
        nn.Linear(15, 3)
    )
    
    def f(x):
        return torch.cat([x, x**2, x**3], dim=1)
    
    def f2(x):
        y = x[:,0] + x[:,1]
        y = y.reshape(-1,1) 
        return torch.cat([y, y**2, y**3], dim=1)
    
    x = torch.arange(1, 6).reshape(-1,1).float().requires_grad_()
    

    
    y1 = f(x)[...,0].reshape(-1,1)
    y2 = f(x)[...,1].reshape(-1,1)
    y3 = f(x)[...,2].reshape(-1,1) 
    
    dy1 = grad(y1, x)[0]
    dy2 = grad(y2, x)[0]
    dy3 = grad(y3, x)[0]
    
    
    print(dy1)
    print(dy2)
    print(dy3)
    
    
    x2 = torch.arange(1, 7).reshape(-1,2).float().requires_grad_()
    print(x2.shape)
    y = f2(x2)
    dy = multi_output_grad(y, x2)
    print(dy)
