import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from matplotlib.colors import TwoSlopeNorm
# Define true Taylor-Green vortex solution
def taylor_green_solution(t, x, y, nu):
    u = -torch.cos(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-2 * np.pi**2 * nu * t)
    v = torch.sin(np.pi * x) * torch.cos(np.pi * y) * torch.exp(-2 * np.pi**2 * nu * t)
    p = -0.25 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y)) * torch.exp(-4 * np.pi**2 * nu * t)
    return u, v, p

# Create output directory
os.makedirs("figures", exist_ok=True)

# Generate GIF of true pressure field from t=0 to t=1
grid_size = 100
x = torch.linspace(-1, 1, grid_size)
y = torch.linspace(-1, 1, grid_size)
xv, yv = torch.meshgrid(x, y, indexing='xy')
x_flat = xv.flatten().unsqueeze(1)
y_flat = yv.flatten().unsqueeze(1)

nu = 0.01

# Use theoretical bounds for pressure from Taylor-Green solution
p_min, p_max = -0.6, 0.6  # conservative fixed range

# Plot and save frames
frames = []
tmp_files = []

for t_val in np.linspace(0, 1, 30):
    t_tensor = torch.full_like(x_flat, t_val)
    _, _, p_true = taylor_green_solution(t_tensor, x_flat, y_flat, nu)
    p_true = p_true.view(grid_size, grid_size).numpy()

    fig, ax = plt.subplots()
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)
    im = ax.contourf(xv.numpy(), yv.numpy(), p_true, levels=50, cmap='coolwarm', norm=norm)
    # cbar = plt.colorbar(im, ax=ax)
    cbar = plt.colorbar(im, ax=ax, ticks=[-0.6, -0.3, 0.0, 0.3, 0.6])
    cbar.set_label("Pressure")
    plt.title(f"True Pressure at t={t_val:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    fname = f"figures/p_true_{t_val:.2f}.png"
    plt.savefig(fname)
    plt.close()
    frames.append(imageio.v2.imread(fname))
    tmp_files.append(fname)

# Save GIF
gif_path = "figures/true_pressure.gif"
imageio.mimsave(gif_path, frames, fps=5)
print(f"Saved true pressure GIF as '{gif_path}'")

# Delete temporary figures
for fname in tmp_files:
    os.remove(fname)
print("Temporary figure files deleted.")
