import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns

from PINN.pinn import PINN
from PINN.models.european_call import EuropeanCall



if __name__ == '__main__':
    sns.set_theme()
    torch.manual_seed(1234)


    physics_model = EuropeanCall()
    

    pinn = PINN(physics_model=physics_model, 
                        physics_loss_weight=5, 
                        lr=1e-3, 
                        hidden_layers=[20, 20, 20]
                        )


    losses = pinn.train(epochs=20000)


    grids = 100
    
    preds_upper, preds_lower, preds_mean = pinn.summary()
    preds_upper = preds_upper.flatten().reshape(grids,grids).numpy()
    preds_lower = preds_lower.flatten().reshape(grids,grids).numpy()
    preds_mean = preds_mean.flatten().reshape(grids,grids).numpy()
    
    S_grid = physics_model.eval_X[:,1].reshape(grids,grids).numpy()
    t_grid = 1-physics_model.eval_X[:,0].reshape(grids,grids).numpy()
    
    np.savez('PINN/european_call/pinn_output.npz', preds_upper=preds_upper, preds_lower=preds_lower, preds_mean=preds_mean, S_grid=S_grid, t_grid=t_grid)
    
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
    plt.savefig('PINN/european_call/european_call_pinn.png')
    plt.show()
    

