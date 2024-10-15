import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import utils
import torch.optim as optim
import seaborn as sns
from PINN.common import SGLD
from PINN.common.torch_layers import EFI_Net
# from PINN.common.grad_tool import grad
from PINN.common.base_pinn import BasePINN

from PINN.examples.cooling import Cooling
# from collections import deque

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sns.set_theme()
# torch.manual_seed(42)
torch.manual_seed(1234)

np.random.seed(10)


class PINN_EFI(BasePINN):
    def __init__(
        self,
        physics_model,
        hidden_layers=[15, 15],
        lr=1e-3,
        physics_loss_weight=10,
        sgld_lr=1e-3,
        lambda_y=10,
        lambda_theta=10,
    ) -> None:
        super().__init__(physics_model, hidden_layers, lr, physics_loss_weight)
        
        # EFI configs
        self.sgld_lr = sgld_lr
        self.lambda_y = lambda_y
        self.lambda_theta = lambda_theta

    
    def _pinn_init(self):
        # init EFI net and optimiser
        self.net = EFI_Net(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        
        # init latent noise and sampler
        self.Z = torch.randn(self.n_samples, 1).requires_grad_()
        self.sampler = SGLD([self.Z], self.sgld_lr)
        

    def update(self):
        ## 1. Latent variable sampling (Sample Z)
        self.net.eval()
        theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        Z_loss = self.lambda_y * y_loss + self.lambda_theta * theta_loss + torch.mean(self.Z**2)/2
        

        self.sampler.zero_grad()
        Z_loss.backward()
        self.sampler.step()

        ## 2. DNN weights update (Optimize W)
        
        self.net.train()
        theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        prior_loss = - self.net.gmm_prior_loss() / self.n_samples
        
        w_loss = self.lambda_y * (y_loss + prior_loss + self.physics_loss_weight * self.physics_loss(self.net)) + self.lambda_theta * theta_loss 

        self.optimiser.zero_grad()
        w_loss.backward()
        self.optimiser.step()


if __name__ == '__main__':
    
    Tenv = 25
    T0 = 100
    R = 0.005
    t_end = 300
    t_extend = 1500
    physics_model = Cooling()
    
    times = torch.linspace(0, t_extend, t_extend)
    temps = physics_model.physics_law(times)

    pinn_efi = PINN_EFI(physics_model=physics_model, physics_loss_weight=10, lr=1e-5, sgld_lr=1e-4)

    losses = pinn_efi.train(epochs=10000, eval_x=times.view(-1,1))



    # preds = pinn_efi.predict(times.reshape(-1,1))
    preds_upper, preds_lower, preds_mean = pinn_efi.summary()
    
    # print(preds.shape)

    plt.plot(times, temps, alpha=0.8, color='b', label='Equation')
    # plt.plot(t, T, 'o')
    plt.plot(times, preds_mean, alpha=0.8, color='g', label='PINN-EFI')
    plt.vlines(t_end, Tenv, T0, color='r', linestyles='dashed', label='no data beyond this point')
    plt.fill_between(times, preds_upper, preds_lower, alpha=0.2, color='g', label='95% CI')
    plt.legend()
    plt.ylabel('Temperature (C)')
    plt.xlabel('Time (s)')
    plt.savefig('temp_pred_efi.png')