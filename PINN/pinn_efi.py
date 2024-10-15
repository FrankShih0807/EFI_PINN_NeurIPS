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
from PINN.common.grad_tool import grad

from PINN.examples.cooling import Cooling
# from collections import deque

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sns.set_theme()
# torch.manual_seed(42)
torch.manual_seed(1234)

np.random.seed(10)



def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * torch.exp(-R * time)
    return T


class PINN_EFI(object):
    def __init__(
        self,
        physics_model,
        net_arch=[15, 15],
        lr=1e-3,
        physics_loss_weight=10,
        sgld_lr=1e-3,
        lambda_y=10,
        lambda_theta=10,
    ) -> None:
        super().__init__()
        self.physics_model = physics_model
        # X, y = self.physics_model.X, self.physics_model.y
        print('transfering model params')
        for key, value in self.physics_model.__dict__.items():
            setattr(self, key, value)
            # print('{}: {}'.format(key, value))

        
        # self.input_dim = self.physics_model.input_dim
        # self.output_dim = self.physics_model.output_dim
        
        # Physics loss
        self.physics_loss = self.physics_model.physics_loss
        self.physics_loss_weight = physics_loss_weight
        
        # Common configs
        self.lr = lr
        self.net_arch = net_arch
        self.mse_loss = nn.MSELoss()
        self.activation_fn = F.softplus
        
        # EFI configs
        self.sgld_lr = sgld_lr
        self.lambda_y = lambda_y
        self.lambda_theta = lambda_theta
        
        
        self.net = EFI_Net(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=net_arch, activation_fn=self.activation_fn)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        
        self.collection = []
        
    
    def _pinn_init(self):
        self.Z = torch.randn(self.n_samples, 1).requires_grad_()
        self.sampler = SGLD([self.Z], self.sgld_lr)
        

    # def forward(self, x):
    #     return self.net(x)

    def train(self, epochs):
        # X, y = self.physics_model.X, self.physics_model.y
        # n_samples = y.shape[0]
        
        
        
        # self._build_encoder(encoder_input_dim)
        self._pinn_init()
        

        # self.train()
        losses = []
        for ep in range(epochs):
            
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
            
            
            
            ## 3. Loss calculation
            if ep % int(epochs / 100) == 0:
                loss = self.mse_loss(self.y, self.net(self.X))
                losses.append(loss.item())
                print(f"Epoch {ep}/{epochs}, loss: {losses[-1]:.2f}")
                
            if ep > epochs - 1000:
                y_pred = self.evaluate()
                self.collection.append(y_pred)
        return losses
    
    def evaluate(self):
        x = torch.linspace(0, T_extend, T_extend).view(-1,1)
        y = self.net(x).detach().flatten()
        return y
    
    def summary(self):
        y_pred_mat = torch.stack(self.collection, dim=0)
        y_pred_upper = torch.quantile(y_pred_mat, 0.975, dim=0)
        y_pred_lower = torch.quantile(y_pred_mat, 0.025, dim=0)
        y_pred_mean = torch.mean(y_pred_mat, dim=0)
        return y_pred_upper, y_pred_lower, y_pred_mean
        
    
    def predict(self, X):
        self.net.eval()
        out = self.net(X)
        return out.detach().cpu().numpy()
    
    
Tenv = 25
T0 = 100
R = 0.005
T_end = 300
T_extend = 1500

physics_model = Cooling()
pinn_efi = PINN_EFI(physics_model=physics_model, physics_loss_weight=10, lr=1e-5, sgld_lr=1e-4)

# t, T = physics_model.X, physics_model.y
losses = pinn_efi.train(epochs=10000)


times = torch.linspace(0, T_extend, T_extend)
temps = physics_model.physics_law(times)

preds = pinn_efi.predict(times.reshape(-1,1))
preds_upper, preds_lower, preds_mean = pinn_efi.summary()

plt.plot(times, temps, alpha=0.8, color='b', label='Equation')
# plt.plot(t, T, 'o')
plt.plot(times, preds_mean, alpha=0.8, color='g', label='PINN-EFI')
plt.vlines(T_end, Tenv, T0, color='r', linestyles='dashed', label='no data beyond this point')
plt.fill_between(times, preds_upper, preds_lower, alpha=0.2, color='g', label='95% CI')
# plt.legend(labels=['Equation','Training data', 'PINN-EFI'])
plt.legend()
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.savefig('temp_pred_efi.png')