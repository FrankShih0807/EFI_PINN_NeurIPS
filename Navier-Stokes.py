import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
# -------------------------
# Physics-Informed Network
# -------------------------
class PINN_NS(nn.Module):
    def __init__(self, layers=[3, 50, 50, 50, 3]):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)-2):
            self.net.append(nn.Linear(layers[i], layers[i+1]))
            self.net.append(nn.Tanh())
        self.net.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, tx_y):
        return self.net(tx_y)  # Output: [u, v, p]

# -------------------------
# Autograd Utilities
# -------------------------
def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def compute_residuals(model, tx_y, nu=0.01):
    tx_y.requires_grad_(True)
    uvp = model(tx_y)
    u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
    
    u_t = grad(u, tx_y)[:, 0:1]
    u_x = grad(u, tx_y)[:, 1:2]
    u_y = grad(u, tx_y)[:, 2:3]
    u_xx = grad(u_x, tx_y)[:, 1:2]
    u_yy = grad(u_y, tx_y)[:, 2:3]

    v_t = grad(v, tx_y)[:, 0:1]
    v_x = grad(v, tx_y)[:, 1:2]
    v_y = grad(v, tx_y)[:, 2:3]
    v_xx = grad(v_x, tx_y)[:, 1:2]
    v_yy = grad(v_y, tx_y)[:, 2:3]

    p_x = grad(p, tx_y)[:, 1:2]
    p_y = grad(p, tx_y)[:, 2:3]

    cont = u_x + v_y
    mom_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    mom_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return mom_u, mom_v, cont

# -------------------------
# Generate Training Points
# -------------------------
def gen_domain(n=10000):
    t = torch.rand(n, 1) * 1.0
    x = torch.rand(n, 1) * 2 - 1
    y = torch.rand(n, 1) * 2 - 1
    return torch.cat([t, x, y], dim=1)

def taylor_green_solution(t, x, y, nu):
    u = -torch.cos(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-2 * np.pi**2 * nu * t)
    v =  torch.sin(np.pi * x) * torch.cos(np.pi * y) * torch.exp(-2 * np.pi**2 * nu * t)
    p = -0.25 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y)) * torch.exp(-4 * np.pi**2 * nu * t)
    return u, v, p

# -------------------------
# Training
# -------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = PINN_NS().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
nu = 0.01

# Collocation points
tx_y_col = gen_domain(10000).to(device)

# Initial condition (t=0)
n_ic = 1000
x_ic = torch.rand(n_ic, 1) * 2 - 1
y_ic = torch.rand(n_ic, 1) * 2 - 1
t_ic = torch.zeros_like(x_ic)
tx_y_ic = torch.cat([t_ic, x_ic, y_ic], dim=1).to(device)
u_ic, v_ic, _ = taylor_green_solution(t_ic, x_ic, y_ic, nu)
uv_ic = torch.cat([u_ic, v_ic], dim=1).to(device)

# -------------------------
# Training Loop
# -------------------------

for step in range(2000):
    optimizer.zero_grad()

    # PDE residuals
    mom_u, mom_v, cont = compute_residuals(model, tx_y_col, nu)
    loss_pde = (mom_u**2).mean() + (mom_v**2).mean() + (cont**2).mean()

    # Initial condition
    uv_pred_ic = model(tx_y_ic)[:, :2]
    loss_ic = ((uv_pred_ic - uv_ic)**2).mean()

    loss = loss_pde + loss_ic
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss.item():.5f}")

# -------------------------
# Visualization
# -------------------------
# Grid for plotting at t=0.5
n_plot = 100
x = torch.linspace(-1, 1, n_plot)
y = torch.linspace(-1, 1, n_plot)
xv, yv = torch.meshgrid(x, y, indexing='xy')
t = torch.full_like(xv, 0.5)
tx_y_plot = torch.stack([t.flatten(), xv.flatten(), yv.flatten()], dim=1).to(device)

model.eval()
with torch.no_grad():
    uvp = model(tx_y_plot)
    u = uvp[:, 0].view(n_plot, n_plot).cpu().numpy()
    v = uvp[:, 1].view(n_plot, n_plot).cpu().numpy()
    p = uvp[:, 2].view(n_plot, n_plot).cpu().numpy()

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Velocity field
ax[0].quiver(xv.numpy(), yv.numpy(), u, v, scale=5)
ax[0].set_title("Velocity Field at t=0.5")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")

# Pressure field
cf = ax[1].contourf(xv.numpy(), yv.numpy(), p, cmap='coolwarm')
plt.colorbar(cf, ax=ax[1])
ax[1].set_title("Pressure Field at t=0.5")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")

plt.tight_layout()
plt.show()