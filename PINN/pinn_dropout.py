import torch.nn as nn
import torch.optim as optim
from PINN.common.base_pinn import BasePINN
from PINN.common.torch_layers import DropoutDNN


class PINN_DROPOUT(BasePINN):
    def __init__(
        self,
        physics_model,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        physics_loss_weight=10,
        dropout_rate=0.01,
        save_path=None,
    ) -> None:
        super().__init__(physics_model, hidden_layers, activation_fn, lr, physics_loss_weight, save_path)

        # Dropout config
        self.dropout_rate = dropout_rate

    def _pinn_init(self):
        self.net = DropoutDNN(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
            dropout_rate=self.dropout_rate,
        )
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)

    def update(self):
        self.optimiser.zero_grad()
        outputs = self.net(self.X)
        loss = self.mse_loss(self.y, outputs)
        loss += self.physics_loss_weight * self.physics_loss(self.net)

        loss.backward()
        self.optimiser.step()

        return loss
    
