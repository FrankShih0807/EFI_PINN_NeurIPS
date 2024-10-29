import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from PINN.common.base_pinn import BasePINN
from PINN.common.torch_layers import BayesianNN
from PINN.models.cooling import Cooling
import seaborn as sns
import matplotlib.pyplot as plt

class PINN_BNN(BasePINN):
    def __init__(
        self,
        physics_model,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        physics_loss_weight=1,
        kl_weight=1e-1,
    ) -> None:
        super().__init__(physics_model, hidden_layers, activation_fn, lr, physics_loss_weight)
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = kl_weight
        

    def _pinn_init(self):
        self.net = BayesianNN(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
    
    def update(self):
        self.optimiser.zero_grad()
        outputs = self.net(self.X)
        loss = self.mse_loss(self.y, outputs) + self.kl_weight * self.kl_loss(self.net)
        loss += self.physics_loss_weight * self.physics_loss(self.net)
        
        loss.backward()
        self.optimiser.step()
        
        return loss


