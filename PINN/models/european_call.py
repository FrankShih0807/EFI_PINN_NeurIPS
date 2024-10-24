import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd.functional import jacobian
import seaborn as sns

from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

    
    
        
class EuropeanCall(PhysicsModel):
    def __init__(self, 
                 S_range = [0, 160],
                 t_range = [0, 1],
                 sigma = 0.4,
                 r = 0.05,
                 K = 80,
                 noise_sd=1
                 ):
        self.norm_dist = dist.Normal(0, 1)
        super().__init__(S_range=S_range, t_range=t_range, sigma=sigma, r=r, K=K, noise_sd=noise_sd)


        
    def _data_generation(self, n_samples=500):
        ivp_x, ivp_y = self.get_ivp_data(n_samples)
        bvp_x1, bvp_y1, bvp_x2, bvp_y2 = self.get_bvp_data(n_samples)
        X = torch.cat([ivp_x, bvp_x1, bvp_x2], dim=0)
        y = torch.cat([ivp_y, bvp_y1, bvp_y2], dim=0)
        y += self.noise_sd * torch.randn_like(y)
        
        self.physics_X = self.get_diff_data(4 * n_samples).requires_grad_(True)
        return X, y
    
    def get_diff_data(self, n_samples):
        ts = torch.rand(n_samples, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        Ss = torch.rand(n_samples, 1) * (self.S_range[1] - self.S_range[0]) + self.S_range[0]
        X = torch.cat([ts, Ss], dim=1)
        # y = np.zeros((n_samples, 1))
        return X
    
    def get_ivp_data(self, n_samples):
        ts = torch.ones(n_samples, 1) * self.t_range[1]
        
        Ss = torch.rand(n_samples, 1) * (self.S_range[1] - self.S_range[0]) + self.S_range[0]
        X = torch.cat([ts, Ss], dim=1)
        y = F.relu(Ss - self.K)
        return X, y
    
    def get_bvp_data(self, n_samples):
        t1 = torch.rand(n_samples, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        S1 = torch.ones(n_samples, 1) * self.S_range[0]
        X1 = torch.cat([t1, S1], dim=1)
        y1 = torch.zeros(n_samples, 1)
        
        t2 = torch.rand(n_samples, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        S2 = torch.ones(n_samples, 1) * self.S_range[1]
        X2 = torch.cat([t2, S2], dim=1)
        y2 =  (self.S_range[1] - self.K * torch.exp(-self.r * (self.t_range[1] - t2)))
        
        return X1, y1, X2, y2
        
    
    def physics_law(self, s, t2m)->torch.Tensor:
        d1 = (torch.log(s/self.K) + (self.r + self.sigma**2/2) * (t2m)) / (self.sigma * torch.sqrt(t2m))
        d2 = d1 - self.sigma * torch.sqrt(t2m)
        Nd1 = self.norm_dist.cdf(d1)
        Nd2 = self.norm_dist.cdf(d2)
        V = s * Nd1 - self.K * torch.exp(-self.r * (t2m)) * Nd2
        return V
    
    
    def physics_loss(self, model: torch.nn.Module):
        ''' Compute the Black-Scholes loss
        Args:
            model (torch.nn.Module): torch network model
        '''
        y_pred = model(self.physics_X)
        grads = grad(y_pred, self.physics_X)[0]
        dVdt = grads[:, 0].view(-1, 1)
        dVdS = grads[:, 1].view(-1, 1)
        grads2nd = grad(dVdS, self.physics_X)[0]
        d2VdS2 = grads2nd[:, 1].view(-1, 1)
        S1 = self.physics_X[:, 1].view(-1, 1)
        bs_pde = dVdt + 0.5 * self.sigma**2 * S1**2 * d2VdS2 + self.r * S1 * dVdS - self.r * y_pred
        
        return bs_pde.pow(2).mean() 

    
    
    def plot(self, s_range=[0, 160], t_range=[0, 1], grid_size=100):
        s = torch.linspace(s_range[0], s_range[1], grid_size)
        t = torch.linspace(t_range[0], t_range[1], grid_size)
        
        S, T = torch.meshgrid(s, t)
        C = self.physics_law(S, T)

        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection='3d')
        
        im = ax.plot_surface(S.numpy(), T.numpy(), C.numpy(), cmap='plasma')
        # ax.contourf(S.numpy(), T.numpy(), C.numpy(), zdir='z', offset=-20, cmap='plasma')
        
        fig.colorbar(im, shrink=0.5, aspect=5, pad=0.07)
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Option Price')
        ax.view_init(elev=30, azim=225)
        plt.show()


if __name__ == "__main__":
    
    call = EuropeanCall()
    
    # call.plot()
    # print(call.X.shape, call.y.shape)
    
    # bvp_x1,bvp_y1,bvp_x2,bvp_y2 = call.get_bvp_data(500)
    # ivp_x1,ivp_y1 = call.get_ivp_data(500)
    # diff_x1,diff_y1 = call.get_diff_data(2000)
    # plt.scatter(bvp_x1[:,0],bvp_x1[:,1], label= "BVP 1", color = "red",marker="o")
    # plt.scatter(bvp_x2[:,0],bvp_x2[:,1], label= "BVP 2", color = "green",marker="x")
    # plt.scatter(ivp_x1[:,0],ivp_x1[:,1], label= "IVP", color = "blue")
    # plt.scatter(diff_x1[:,0],diff_x1[:,1], label= "PDE sample", color = "grey", alpha = 0.3)
    # plt.xlabel("time to expiry, ")
    # plt.ylabel("stock price ")
    # plt.title("Data Sampling for European Call")
    # plt.legend()
    # plt.show()
        
    
    net = nn.Sequential(
        nn.Linear(2, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 1)
    )
    pde_loss = call.physics_loss(net)
    print(pde_loss)