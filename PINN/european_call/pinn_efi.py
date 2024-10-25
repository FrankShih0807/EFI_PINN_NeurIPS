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



class PINN_EFI(BasePINN):
    def __init__(
        self,
        physics_model,
        hidden_layers=[15, 15],
        lr=1e-3,
        physics_loss_weight=10,
        sgld_lr=1e-3,
        lambda_y=1,
        lambda_theta=1,
    ) -> None:
        super().__init__(physics_model, hidden_layers, lr, physics_loss_weight)
        
        # EFI configs
        self.sgld_lr = sgld_lr
        self.lambda_y = lambda_y
        self.lambda_theta = lambda_theta
        
        self.noise_sd = physics_model.noise_sd
        self.activation_fn = nn.Softplus(beta=5)
    
    def _pinn_init(self):
        # init EFI net and optimiser
        self.net = EFI_Net(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        
        # init latent noise and sampler
        self.Z = torch.randn_like(self.y).requires_grad_()
        self.sampler = SGLD([self.Z], self.sgld_lr)
        

    def update(self):
        ## 1. Latent variable sampling (Sample Z)
        self.net.eval()
        theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        # y_loss = self.mse_loss(self.y, self.net(self.X + torch.cat([torch.zeros_like(self.Z), self.Z], dim=1)))
        Z_loss = self.lambda_y * y_loss + self.lambda_theta * theta_loss + torch.mean(self.Z**2)/2/self.noise_sd**2
        

        self.sampler.zero_grad()
        Z_loss.backward()
        self.sampler.step()

        ## 2. DNN weights update (Optimize W)
        
        self.net.train()
        theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        # y_loss = self.mse_loss(self.y, self.net(self.X + torch.cat([torch.zeros_like(self.Z), self.Z], dim=1)))
        prior_loss = - self.net.gmm_prior_loss() / self.n_samples
        
        w_loss = self.lambda_y * (y_loss + prior_loss) + self.physics_loss_weight * self.physics_loss(self.net) + self.lambda_theta * theta_loss 

        self.optimiser.zero_grad()
        w_loss.backward()
        self.optimiser.step()
    


if __name__ == '__main__':
    sns.set_theme()
    torch.manual_seed(1234)


    physics_model = EuropeanCall()
    

    pinn_efi = PINN_EFI(physics_model=physics_model, 
                        physics_loss_weight=5, 
                        lr=1e-4, 
                        sgld_lr=1e-4, 
                        lambda_y=10, 
                        lambda_theta=1,
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
    plt.savefig('PINN/european_call/european_call_efi.png')
    plt.show()
    

