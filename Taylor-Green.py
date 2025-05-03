import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Set device (MPS or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Neural Network Architecture
class PINN_TaylorGreen(nn.Module):
    def __init__(self, layers=[3, 50, 50, 3]):
        super().__init__()
        net = []
        for i in range(len(layers) - 2):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)

    def forward(self, tx_y):
        return self.net(tx_y)

# Utility functions for autograd
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

def taylor_green_solution(t, x, y, nu):
    u = -torch.cos(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-2 * np.pi**2 * nu * t)
    v = torch.sin(np.pi * x) * torch.cos(np.pi * y) * torch.exp(-2 * np.pi**2 * nu * t)
    p = -0.25 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y)) * torch.exp(-4 * np.pi**2 * nu * t)
    return u, v, p

def generate_domain(n):
    t = torch.rand(n, 1)
    x = 2 * torch.rand(n, 1) - 1
    y = 2 * torch.rand(n, 1) - 1
    return torch.cat([t, x, y], dim=1)

# Initialize model
model = PINN_TaylorGreen().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
nu = 0.01

# Generate data
collocation_points = generate_domain(10000).to(device)
x_ic = 2 * torch.rand(1000, 1) - 1
y_ic = 2 * torch.rand(1000, 1) - 1
t_ic = torch.zeros_like(x_ic)
txy_ic = torch.cat([t_ic, x_ic, y_ic], dim=1).to(device)
u_ic, v_ic, _ = taylor_green_solution(t_ic, x_ic, y_ic, nu)
uv_ic = torch.cat([u_ic, v_ic], dim=1).to(device)

# Training
start_time = time.time()
for step in range(10000):
    optimizer.zero_grad()
    mom_u, mom_v, cont = compute_residuals(model, collocation_points, nu)
    loss_pde = (mom_u**2).mean() + (mom_v**2).mean() + (cont**2).mean()

    uv_pred_ic = model(txy_ic)[:, :2]
    loss_ic = ((uv_pred_ic - uv_ic) ** 2).mean()

    loss = loss_pde + loss_ic
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss.item():.5f}")

if device.type == "mps": torch.mps.synchronize()
print(f"Training time: {time.time() - start_time:.2f} seconds")

# Create output directory
os.makedirs("figures", exist_ok=True)

# Evaluation and visualization at multiple time steps
time_steps = [0.0, 0.5, 1.0]
grid_size = 100
x = torch.linspace(-1, 1, grid_size)
y = torch.linspace(-1, 1, grid_size)
xv, yv = torch.meshgrid(x, y, indexing='xy')

model.eval()

for t_val in time_steps:
    t = torch.full_like(xv, t_val)
    txy_eval = torch.stack([t.flatten(), xv.flatten(), yv.flatten()], dim=1).to(device)

    with torch.no_grad():
        uvp_pred = model(txy_eval)
        u_pred = uvp_pred[:, 0].view(grid_size, grid_size).cpu().numpy()
        v_pred = uvp_pred[:, 1].view(grid_size, grid_size).cpu().numpy()
        p_pred = uvp_pred[:, 2].view(grid_size, grid_size).cpu().numpy()

    # Ground truth
    x_flat = xv.flatten().unsqueeze(1)
    y_flat = yv.flatten().unsqueeze(1)
    t_tensor = torch.full_like(x_flat, t_val)
    u_true, v_true, p_true = taylor_green_solution(t_tensor, x_flat, y_flat, nu)
    u_true = u_true.view(grid_size, grid_size).numpy()
    v_true = v_true.view(grid_size, grid_size).numpy()
    p_true = p_true.view(grid_size, grid_size).numpy()

    # Error
    u_err = np.abs(u_pred - u_true)
    v_err = np.abs(v_pred - v_true)
    p_err = np.abs(p_pred - p_true)

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs[0, 0].quiver(xv, yv, u_pred, v_pred)
    axs[0, 0].set_title(f"Predicted Velocity at t={t_val}")

    axs[0, 1].quiver(xv, yv, u_true, v_true)
    axs[0, 1].set_title(f"True Velocity at t={t_val}")

    axs[0, 2].contourf(xv, yv, np.sqrt(u_err**2 + v_err**2), cmap='viridis')
    axs[0, 2].set_title(f"Velocity Error at t={t_val}")

    cf1 = axs[1, 0].contourf(xv, yv, p_pred, cmap='coolwarm')
    axs[1, 0].set_title("Predicted Pressure")
    fig.colorbar(cf1, ax=axs[1, 0])

    cf2 = axs[1, 1].contourf(xv, yv, p_true, cmap='coolwarm')
    axs[1, 1].set_title("True Pressure")
    fig.colorbar(cf2, ax=axs[1, 1])

    cf3 = axs[1, 2].contourf(xv, yv, p_err, cmap='plasma')
    axs[1, 2].set_title("Pressure Error")
    fig.colorbar(cf3, ax=axs[1, 2])

    plt.tight_layout()
    plt.savefig(f"figures/taylor_green_t{t_val:.1f}.png")
    plt.close()

print("Evaluation and plots saved to 'figures/' folder.")
