import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from PINN.common.torch_layers import BaseDNN

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
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.net_arch = net_arch
        self.activation = F.softplus
        # self.activation = nn.ReLU

        self.net = BaseDNN(input_dim=input_dim, output_dim=output_dim, hidden_layers=net_arch, activation_fn=self.activation)


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
T_end = 300
T_extend = 1500
times = torch.linspace(0, T_extend, T_extend)
eq = functools.partial(cooling_law, Tenv=Tenv, T0=T0, R=R)
temps = eq(times)

# Make training data
n_samples = 200
t = torch.linspace(0, T_end, n_samples).reshape(n_samples, -1)
T = eq(t) +  torch.randn(n_samples).reshape(n_samples, -1)



def physics_loss(model: torch.nn.Module):
    ts = torch.linspace(0, T_extend, steps=T_extend,).view(-1,1).requires_grad_(True).to(DEVICE)
    temps = model(ts)
    dT = grad(temps, ts)[0]
    pde = R*(Tenv - temps) - dT
    
    return torch.mean(pde**2)

net_arch = [15, 15]
net = Net(1,1, net_arch, loss2=physics_loss, epochs=10000, loss2_weight=10, lr=1e-3).to(DEVICE)
# net = Net(1,1, net_arch, loss2=None, epochs=30000, loss2_weight=1, lr=1e-3).to(DEVICE)

losses = net.fit(t, T)
# plt.plot(losses)
# plt.yscale('log')

preds = net.predict(times.reshape(-1,1))

plt.plot(times, temps, alpha=0.8, color='b', label='Equation')
# plt.plot(t, T, 'o')
plt.plot(times, preds, alpha=0.8, color='g', label='PINN')
plt.vlines(T_end, Tenv, T0, color='r', linestyles='dashed', label='no data beyond this point')
# plt.legend(labels=['Equation','Training data', 'PINN'])
plt.legend()
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.savefig('temp_pred.png')