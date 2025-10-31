import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np

class SGLD(Optimizer):
    def __init__(self, params, lr=required, temperature=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, tau=temperature)
        super(SGLD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            tau = group['tau']
            for p in group['params']:
                if p.grad is None:
                    continue
                # Add noise
                noise = torch.randn_like(p.grad)
                # SGLD update
                p.data.sub_(p.grad * lr)
                p.data.add_(noise * (2 * lr * tau) ** 0.5)

        return loss

class SGHMC(Optimizer):
    def __init__(self, 
                params,
                lr=1e-3, 
                alpha=1.0,
                ):

        defaults = dict(lr=lr, alpha=alpha)
        super().__init__(params, defaults)
        


    @torch.no_grad()
    def step(self, closure=None):
        ''' One sigle step of LKTD algorithm
            observation (Tensor):  
            measurement (Tensor):
        '''
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                v = state['momentum']
                v = (1 - alpha) * v - lr * p.grad + np.sqrt(2 * alpha * lr) * torch.randn_like(p, device=p.device)
                p.add_(v)
