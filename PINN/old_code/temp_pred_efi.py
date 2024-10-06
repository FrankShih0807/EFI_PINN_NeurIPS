import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from PINN.common import BaseNetwork, SparseDNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sns.set_theme()
torch.manual_seed(42)

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
        self.activation = nn.ReLU

        self.net = BaseNetwork(input_size=input_dim, output_size=output_dim, hidden_layers=net_arch, activation_fn=nn.ReLU)
        
        # self.inverse_net = SparseDNN(input_size=output_dim, output_size=input_dim, hidden_layers=net_arch, activation_fn=nn.ReLU)


    def forward(self, x):
        return self.net(x)

    def fit(self, X, y):
        optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(X)
            loss = self.loss(y, outputs)
            if self.loss2:
                loss += self.loss2_weight + self.loss2_weight * self.loss2(self)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses

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
n_samples = 10
obs_noise = 2
t = torch.linspace(0, 300, 10).reshape(n_samples, -1)
T = eq(t) +  obs_noise * torch.randn(10).reshape(n_samples, -1)



def physics_loss(model: torch.nn.Module):
    ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True).to(DEVICE)
    temps = model(ts)
    dT = grad(temps, ts)[0]
    pde = R*(Tenv - temps) - dT
    
    return torch.mean(pde**2)

# net_arch = [100, 100, 100, 100]
net_arch = [64, 64]
net = Net(1,1, net_arch, loss2=physics_loss, epochs=20000, loss2_weight=1, lr=1e-4).to(DEVICE)

losses = net.fit(t, T)
# plt.plot(losses)
# plt.yscale('log')

preds = net.predict(times.reshape(-1,1))

plt.plot(times, temps, alpha=0.8)
plt.plot(t, T, 'o')
plt.plot(times, preds, alpha=0.8)
plt.legend(labels=['Equation','Training data', 'PINN'])
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.savefig('temp_pred_efi.png')