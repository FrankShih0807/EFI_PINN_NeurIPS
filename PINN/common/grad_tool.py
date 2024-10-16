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



if __name__ == '__main__':
    
    
    net = nn.Sequential(
        nn.Linear(1, 15),
        nn.Softplus(),
        nn.Linear(15, 15),
        nn.Softplus(),
        nn.Linear(15, 2)
    )
    
    N = 5
    x = torch.randn(N, 1).requires_grad_(True)
    y = net(x)
    
    d = jacobian(net, x)
    print(d[range(x.size(0)), :, range(x.size(0)), :].shape)