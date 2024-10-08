import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import utils
import torch.optim as optim
import seaborn as sns
from PINN.common import BaseNetwork, SparseDNN, SGLD
from PINN.common.net_in_net import EFI_Net

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sns.set_theme()
# torch.manual_seed(42)

np.random.seed(10)


def grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * torch.exp(-R * time)
    return T


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        net_arch,
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
        self.activation = nn.Softplus
        # self.activation = nn.ReLU

        # self.net = BaseNetwork(input_size=input_dim, output_size=output_dim, hidden_layers=net_arch, activation_fn=self.activation)
        self.net = EFI_Net(activation=self.activation)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        
        # self.parameter_size = self.net.parameter_size
        

    def _build_encoder(self, encoder_input_dim):
        hidden_size = self.parameter_size
        net_arch = [hidden_size]
        self.encoder = SparseDNN(input_size=encoder_input_dim, output_size=self.parameter_size, hidden_layers=net_arch, activation_fn=nn.ReLU, prior_sd=0.01, sparse_sd=0.001, sparsity=1.0)
    
    def _initialize_latent(self, n_samples):
        self.Z = torch.randn(n_samples, 1).requires_grad_()
        self.sampler = SGLD([self.Z], self.sgld_lr)
        

    def forward(self, x):
        return self.net(x)

    def fit(self, X, y):
        n_samples = y.shape[0]
        variable_dim = X.shape[1]
        encoder_input_dim = 2 + variable_dim
        
        lambda_1 = 10
        lambda_2 = 10
        
        # self._build_encoder(encoder_input_dim)
        self._initialize_latent(n_samples)
        

        self.train()
        losses = []
        for ep in range(self.epochs):
            
            ## 1. Latent variable sampling (Sample Z)
            self.net.eval()
            theta_loss = self.net.theta_loss(torch.cat([X, y, self.Z], dim=1))
            y_loss = self.loss(y, self.net(X) + self.Z)
            # print(self.net(X), y)
            # raise
            Z_loss = lambda_2 * y_loss + lambda_1 * theta_loss + torch.mean(self.Z**2)/2
            
            
            # thetas = self.encoder(torch.cat([X, y, self.Z], dim=1))
            # bar_theta = thetas.mean(dim=0)
            # utils.vector_to_parameters(bar_theta, self.net.parameters())
            # tilde_y = self.net(X) + self.Z
            
            # latent_loss = self.latent_loss(n_samples, lambda_1, lambda_2, y, tilde_y, thetas, bar_theta, self.Z)

            self.sampler.zero_grad()
            Z_loss.backward()
            self.sampler.step()

            ## 2. DNN weights update (Optimize W)
            
            self.net.train()
            theta_loss = self.net.theta_loss(torch.cat([X, y, self.Z], dim=1))
            y_loss = self.loss(y, self.net(X) + self.Z)
            prior_loss = - self.net.mixture_gaussian_prior() / n_samples
            
            w_loss = lambda_2 * (y_loss + prior_loss + self.loss2_weight * self.loss2(self.net)) + lambda_1 * theta_loss 
            # w_loss = self.loss2(self.net)

            self.optimiser.zero_grad()
            w_loss.backward()
            # for p in self.net.parameters():
            #     print(p.grad)
                
            # raise
            self.optimiser.step()
            
            
            
            ## 3. Loss calculation
            
            loss = self.loss(y, self.net(X))
            # if self.loss2:
            #     loss += self.loss2_weight * self.loss2(self)
            losses.append(loss.item())
            if ep % int(self.epochs / 100) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses
    
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
        self.eval()
        out = self.forward(X)
        return out.detach().cpu().numpy()
    
    
Tenv = 25
T0 = 100
R = 0.005
times = torch.linspace(0, 1000, 1000)
eq = functools.partial(cooling_law, Tenv=Tenv, T0=T0, R=R)
temps = eq(times)

# Make training data
n_samples = 100
obs_sd = 1
t = torch.linspace(0, 300, n_samples).reshape(n_samples, -1)
T = eq(t) +  obs_sd * torch.randn(n_samples).reshape(n_samples, -1)



def physics_loss(model: torch.nn.Module):
    ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True)
    temps = model(ts)
    # print(temps)
    # raise
    dT = grad(temps, ts)[0]
    pde = R*(Tenv - temps) - dT
    
    return torch.mean(pde**2)

# net_arch = [100, 100, 100, 100]
net_arch = [15, 15]
net = Net(1,1, net_arch, loss2=physics_loss, epochs=10000, loss2_weight=1, lr=1e-4, sgld_lr=1e-4)


losses = net.fit(t, T)
# plt.plot(losses)
# plt.yscale('log')

preds = net.predict(times.reshape(-1,1))

plt.plot(times, temps, alpha=0.8)
plt.plot(t, T, 'o')
plt.plot(times, preds, alpha=0.8)
plt.legend(labels=['Equation','Training data', 'PINN-EFI'])
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.savefig('temp_pred_efi.png')