import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from PINN.common.base_pinn import BasePINN
from PINN.common.torch_layers import BayesianNN



class PINN_BNN(BasePINN):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        physics_loss_weight=1,
        save_path=None,
        device='cpu'
    ) -> None:
        super().__init__(physics_model, dataset, hidden_layers, activation_fn, lr, physics_loss_weight, save_path, device)
    
    

    
    def pde_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'differential':
                diff_o = self.differential_operator(self.net, d['X'])
                loss += self.mse_loss(diff_o, d['y'])
        return loss
    
    def solution_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'solution':
                loss += self.mse_loss(d['y'], self.net(d['X']))
        return loss
    
    def update(self):
        self.optimiser.zero_grad()
        # outputs = self.net(self.X)
        # loss = self.mse_loss(self.y, outputs)
        loss = self.solution_loss() + self.physics_loss_weight * self.pde_loss()
        # loss += self.physics_loss_weight * self.physics_loss(self.net, self.physics_X)
        
        loss.backward()
        self.optimiser.step()
        
        return loss

