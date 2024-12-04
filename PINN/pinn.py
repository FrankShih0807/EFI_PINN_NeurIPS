import torch
import torch.nn as nn
from PINN.common.base_pinn import BasePINN
from PINN.models.poisson import Poisson
from PINN.common.torch_layers import DropoutDNN
from torch import optim

class PINN(BasePINN):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        lambda_pde=1,
        dropout_rate=0.0,
        save_path=None,
        device='cpu'
    ) -> None:
        
        self.dropout_rate = dropout_rate
        
        super().__init__(
            physics_model=physics_model,
            dataset=dataset,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            lr=lr,
            lambda_pde=lambda_pde,
            save_path=save_path,
            device=device,
        )

    def _pinn_init(self):
        # init pinn net and optimiser
        self.net = DropoutDNN(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
            dropout_rate=self.dropout_rate,
        )
        self.net.to(self.device)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        
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
        sol_loss = self.solution_loss()
        pde_loss = self.pde_loss()
        loss = sol_loss + self.lambda_pde * pde_loss

        loss.backward()
        self.optimiser.step()

        return sol_loss.item(), pde_loss.item()
