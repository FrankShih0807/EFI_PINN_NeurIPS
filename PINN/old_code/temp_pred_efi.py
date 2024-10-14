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

from PINN.examples.cooling import CoolingModel
# from collections import deque

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sns.set_theme()
# torch.manual_seed(42)
torch.manual_seed(1234)

np.random.seed(10)



def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * torch.exp(-R * time)
    return T


class PINN_EFI(nn.Module):
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        net_arch=[15, 15],
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        sgld_lr=1e-3,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.sgld_lr = sgld_lr
        self.net_arch = net_arch
        self.activation_fn = F.softplus

        self.net = EFI_Net(input_dim=input_dim, output_dim=output_dim, hidden_layers=net_arch, activation_fn=self.activation_fn)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        
        # self.parameter_size = self.net.parameter_size
        self.collection = []
        
    
    def _initialize_latent(self, n_samples):
        self.Z = torch.randn(n_samples, 1).requires_grad_()
        self.sampler = SGLD([self.Z], self.sgld_lr)
        

    def forward(self, x):
        return self.net(x)

    def fit(self, X, y):
        n_samples = y.shape[0]
        
        lambda_1 = 10
        lambda_2 = 10
        
        # self._build_encoder(encoder_input_dim)
        self._initialize_latent(n_samples)
        

        # self.train()
        losses = []
        for ep in range(self.epochs):
            
            ## 1. Latent variable sampling (Sample Z)
            self.net.eval()
            theta_loss = self.net.theta_encode(X, y, self.Z)
            y_loss = self.loss(y, self.net(X) + self.Z)
            Z_loss = lambda_2 * y_loss + lambda_1 * theta_loss + torch.mean(self.Z**2)/2
            

            self.sampler.zero_grad()
            Z_loss.backward()
            self.sampler.step()

            ## 2. DNN weights update (Optimize W)
            
            self.net.train()
            theta_loss = self.net.theta_encode(X, y, self.Z)
            y_loss = self.loss(y, self.net(X) + self.Z)
            prior_loss = - self.net.gmm_prior_loss() / n_samples
            
            w_loss = lambda_2 * (y_loss + prior_loss + self.loss2_weight * self.loss2(self.net)) + lambda_1 * theta_loss 

            self.optimiser.zero_grad()
            w_loss.backward()
            self.optimiser.step()
            
            
            
            ## 3. Loss calculation
            if ep % int(self.epochs / 100) == 0:
                loss = self.loss(y, self.net(X))
                losses.append(loss.item())
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
                
            if ep > self.epochs - 1000:
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
        
    def latent_loss(self, n_samples, lambda_1, lambda_2, y , tilde_y, thetas, bar_theta, Z):
        loss = lambda_2 * self.loss(y,tilde_y) + lambda_1 * self.loss(thetas,bar_theta.repeat(n_samples,1)) + torch.mean(Z**2)/2
        if self.loss2 is not None:
            loss += lambda_2 * self.loss2_weight * self.loss2(self.net)
        return loss
    
    def encoder_loss(self, n_samples, lambda_1, lambda_2, y , tilde_y, thetas, bar_theta):
        loss = lambda_2 * self.loss(y,tilde_y) + lambda_1 * self.loss(thetas,bar_theta.repeat(n_samples,1))
        prior_loss = - self.encoder.mixture_gaussian_prior()
        loss += prior_loss / n_samples
        if self.loss2 is not None:
            loss += lambda_2 * self.loss2_weight * self.loss2(self.net)
        return loss
    
    def predict(self, X):
        self.net.eval()
        out = self.net(X)
        return out.detach().cpu().numpy()
    
    
Tenv = 25
T0 = 100
R = 0.005
T_end = 300
T_extend = 1500
times = torch.linspace(0, T_extend, T_extend)
eq = functools.partial(cooling_law, Tenv=Tenv, T0=T0, R=R)
temps = eq(times)

# Make training data
n_samples = 200
noise_sd = 1
t = torch.linspace(0, T_end, n_samples).reshape(n_samples, -1)
T = eq(t) +  noise_sd * torch.randn(n_samples).reshape(n_samples, -1)



def physics_loss(model: torch.nn.Module):
    ts = torch.linspace(0, T_extend, steps=T_extend,).view(-1,1).requires_grad_(True)
    temps = model(ts)
    # print(temps)
    # raise
    dT = grad(temps, ts)[0]
    pde = R*(Tenv - temps) - dT
    
    return torch.mean(pde**2)

# net_arch = [100, 100, 100, 100]
net_arch = [15, 15]
net = PINN_EFI(1,1, net_arch, loss2=physics_loss, epochs=10000, loss2_weight=10, lr=1e-5, sgld_lr=1e-4)


losses = net.fit(t, T)
# plt.plot(losses)
# plt.yscale('log')

preds = net.predict(times.reshape(-1,1))
preds_upper, preds_lower, preds_mean = net.summary()

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