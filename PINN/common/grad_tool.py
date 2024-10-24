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
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

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
    
    def n2one(x):
        y = x[:, 0] ** 2 + x[:, 1] ** 3
        y = y.reshape(-1, 1)
        return y
    
    x = torch.arange(1, 7).reshape(-1, 2).float().requires_grad_(True)
    
    y = n2one(x)
    print(x)
    print(y)
    
    dy_dx = grad(y, x)
    dy_dx1 = dy_dx[0]
    print(dy_dx)
    
    # dy_d2x = grad(dy_dx, x)
    print(grad(dy_dx[0][:,0], x))
    print(grad(dy_dx[0][:,1], x))