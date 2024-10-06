import torch
from torch.optim.optimizer import Optimizer, required

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

