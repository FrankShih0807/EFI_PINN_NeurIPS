import torch
import torch.nn as nn
from PINN.common.base_pinn import BasePINN

class PINN(BasePINN):
    def __init__(
        self,
        physics_model,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        physics_loss_weight=1,
        save_path=None,
        device='cpu'
    ) -> None:
        super().__init__(physics_model, hidden_layers, activation_fn, lr, physics_loss_weight, save_path, device)
        

    def update(self):
        self.optimiser.zero_grad()
        outputs = self.net(self.X)
        loss = self.mse_loss(self.y, outputs)
        loss += self.physics_loss_weight * self.physics_loss(self.net, self.physics_X)
        
        loss.backward()
        self.optimiser.step()
        
        return loss

