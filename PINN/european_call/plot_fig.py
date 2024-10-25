import matplotlib.pyplot as plt
import numpy as np


data = np.load('PINN/european_call/output.npz')

preds_upper = data['preds_upper']
preds_lower = data['preds_lower']
preds_mean = data['preds_mean']
S_grid = data['S_grid']
t_grid = data['t_grid']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
im = ax.plot_surface(S_grid,
                        t_grid, 
                        preds_mean, 
                        cmap='plasma')



fig.colorbar(im, shrink=0.5, aspect=5, pad=0.07)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Option Price')
ax.view_init(elev=30, azim=225)
# plt.savefig('PINN/european_call/european_call_efi.png')
plt.show()