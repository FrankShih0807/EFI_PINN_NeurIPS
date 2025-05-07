import torch

u = torch.randn(10, 3)

t = u[:, 0:1]
x = u[:, 1:2]
y = u[:, 2:3]

pde = torch.cat([t, x, y], dim=0)
print(pde.shape)