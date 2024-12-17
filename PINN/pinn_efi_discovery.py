import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from PINN.common import SGLD
from PINN.common.torch_layers import EFI_Net_PE
from PINN.common.base_pinn import BasePINN


class PINN_EFI_Discovery(BasePINN):
    def __init__(
        self,
        physics_model,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        physics_loss_weight=10,
        sgld_lr=1e-3,
        lambda_y=1,
        lambda_theta=1,
        save_path=None,
    ) -> None:
        super().__init__(physics_model, hidden_layers, activation_fn, lr, physics_loss_weight, save_path)
        
        # EFI configs
        self.sgld_lr = sgld_lr
        self.lambda_y = lambda_y
        self.lambda_theta = lambda_theta
        
        self.noise_sd = physics_model.noise_sd
        self.n_update = 0

    
    def _pinn_init(self):
        # init EFI net and optimiser
        self.net = EFI_Net_PE(input_dim=self.input_dim, output_dim=self.output_dim, variable_dim=1, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        
        # init latent noise and sampler
        self.Z = (self.noise_sd * torch.randn_like(self.y)).requires_grad_()
        self.sampler = SGLD([self.Z], self.sgld_lr)
        

    def update(self):
        ## 1. Latent variable sampling (Sample Z)
        self.net.eval()
        theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        Z_loss = self.lambda_y * y_loss + self.lambda_theta * theta_loss + torch.mean(self.Z**2)/2/self.noise_sd**2
        

        self.sampler.zero_grad()
        Z_loss.backward()
        self.sampler.step()

        ## 2. DNN weights update (Optimize W)
        
        self.net.train()
        theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        prior_loss = - self.net.gmm_prior_loss() / self.n_samples
        
        w_loss = self.lambda_y * (y_loss + prior_loss) + self.physics_loss_weight * self.physics_loss(self.net, self.net.variable_tensor[0]) + self.lambda_theta * theta_loss 

        self.optimiser.zero_grad()
        w_loss.backward()
        self.optimiser.step()

        self.n_update += 1
        if self.n_update % 100 == 0:
            print("r discovery: ", self.net.variable_tensor[0].item())
