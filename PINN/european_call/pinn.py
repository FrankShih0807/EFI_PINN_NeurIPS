import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from PINN.common import SGLD
from PINN.common.torch_layers import EFI_Net
from PINN.common.base_pinn import BasePINN
from PINN.models.european_call import EuropeanCall


class PINN(BasePINN):
    def __init__(
        self,
        physics_model,
        hidden_layers=[15, 15],
        lr=1e-3,
        physics_loss_weight=10,
    ) -> None:
        super().__init__(physics_model, hidden_layers, lr, physics_loss_weight)
        

    def update(self):
        self.optimiser.zero_grad()
        outputs = self.net(self.X)
        loss = self.mse_loss(self.y, outputs)
        loss += self.physics_loss_weight * self.physics_loss(self.net)
        
        loss.backward()
        self.optimiser.step()
        
        return loss
    


if __name__ == '__main__':
    sns.set_theme()
    torch.manual_seed(1234)


    physics_model = EuropeanCall()
    

    pinn_efi = PINN(physics_model=physics_model, 
                        physics_loss_weight=5, 
                        lr=1e-3, 
                        hidden_layers=[20, 20, 20]
                        )

    # print(pinn_efi.eval_X.shape)
    
    losses = pinn_efi.train(epochs=20000)


    # preds = pinn_efi.predict(times.reshape(-1,1))
    grids = 100
    
    preds_upper, preds_lower, preds_mean = pinn_efi.summary()
    preds_upper = preds_upper.flatten().reshape(grids,grids).numpy()
    preds_lower = preds_lower.flatten().reshape(grids,grids).numpy()
    preds_mean = preds_mean.flatten().reshape(grids,grids).numpy()
    
    S_grid = physics_model.eval_X[:,1].reshape(grids,grids).numpy()
    t_grid = 1-physics_model.eval_X[:,0].reshape(grids,grids).numpy()
    
    np.savez('PINN/european_call/output.npz', preds_upper=preds_upper, preds_lower=preds_lower, preds_mean=preds_mean, S_grid=S_grid, t_grid=t_grid)
    
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
    

