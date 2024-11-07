import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import os

from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel

    
        
class EuropeanCall(PhysicsModel):
    def __init__(self, 
                 S_range = [0, 160],
                 t_range = [0, 1],
                 sigma = 0.5,
                 r = 0.05,
                 K = 80,
                 noise_sd=1
                 ):
        self.norm_dist = dist.Normal(0, 1)
        super().__init__(S_range=S_range, t_range=t_range, sigma=sigma, r=r, K=K, noise_sd=noise_sd)


        
    def _data_generation(self, n_samples=200):
        ivp_x, ivp_y = self.get_ivp_data(n_samples)
        bvp_x1, bvp_y1, bvp_x2, bvp_y2 = self.get_bvp_data(n_samples)
        X = torch.cat([ivp_x, bvp_x1, bvp_x2], dim=0)
        y = torch.cat([ivp_y, bvp_y1, bvp_y2], dim=0)        
        # bvp_x2, bvp_y2 = self.get_bvp_data(n_samples)
        # X = torch.cat([ivp_x, bvp_x2], dim=0)
        # y = torch.cat([ivp_y, bvp_y2], dim=0)

        
        self.physics_X = self.get_diff_data(4 * n_samples).requires_grad_(True)
        return X, y
    
    def _eval_data_generation(self):
        eval_t = torch.linspace(self.t_range[0], self.t_range[1], 100)
        eval_S = torch.linspace(self.S_range[0], self.S_range[1], 100)
        S, T = torch.meshgrid(eval_S, eval_t, indexing='ij')
        eval_X = torch.cat([T.reshape(-1, 1), S.reshape(-1, 1)], dim=1)
        return eval_X
    
    def get_diff_data(self, n_samples):
        ts = torch.rand(n_samples, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        Ss = torch.rand(n_samples, 1) * (self.S_range[1] - self.S_range[0]) + self.S_range[0]
        X = torch.cat([ts, Ss], dim=1)
        return X
    
    def get_ivp_data(self, n_samples):
        ts = torch.ones(n_samples, 1) * self.t_range[1]
        Ss = torch.rand(n_samples, 1) * (self.S_range[1] - self.S_range[0]) + self.S_range[0]
        X = torch.cat([ts, Ss], dim=1)
        y = F.relu(Ss - self.K)
        return X, y
    
    def get_bvp_data(self, n_samples):
        # Boundary condition at S = 0
        t1 = torch.rand(n_samples, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        S1 = torch.ones(n_samples, 1) * self.S_range[0]
        X1 = torch.cat([t1, S1], dim=1)
        y1 = torch.zeros(n_samples, 1)
        
        # Boundary condition at S = S_max/2
        t2 = torch.rand(n_samples, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        S2 = torch.ones(n_samples, 1) * self.S_range[1]/2
        X2 = torch.cat([t2, S2], dim=1)
        # y2 =  (self.S_range[1] - self.K * torch.exp(-self.r * (self.t_range[1] - t2)))
        y2 = self.physics_law(S2, self.t_range[1] - t2)
        y2 += self.noise_sd * torch.randn_like(y2)
        
        return X1, y1, X2, y2
        # return X2, y2
        
    
    def physics_law(self, s, t2m)->torch.Tensor:
        s = torch.as_tensor(s)
        t2m = torch.as_tensor(t2m)
        d1 = (torch.log(s/self.K) + (self.r + self.sigma**2/2) * (t2m)) / (self.sigma * torch.sqrt(t2m))
        d2 = d1 - self.sigma * torch.sqrt(t2m)
        Nd1 = self.norm_dist.cdf(d1)
        Nd2 = self.norm_dist.cdf(d2)
        V = s * Nd1 - self.K * torch.exp(-self.r * (t2m)) * Nd2
        return V
    
    
    def physics_loss(self, model: torch.nn.Module, physics_X):
        ''' Compute the Black-Scholes loss
        Args:
            model (torch.nn.Module): torch network model
        '''
        # self.physics_X = self.get_diff_data(800).requires_grad_(True)
        y_pred = model(physics_X)
        grads = grad(y_pred, physics_X)[0]
        dVdt = grads[:, 0].view(-1, 1)
        dVdS = grads[:, 1].view(-1, 1)
        grads2nd = grad(dVdS, physics_X)[0]
        d2VdS2 = grads2nd[:, 1].view(-1, 1)
        S1 = physics_X[:, 1].view(-1, 1)
        bs_pde = dVdt + 0.5 * self.sigma**2 * S1**2 * d2VdS2 + self.r * S1 * dVdS - self.r * y_pred
        
        return bs_pde.pow(2).mean() 

    
    
    def plot(self, s_range=[0, 160], t_range=[0, 1], grid_size=100):
        s = torch.linspace(s_range[0], s_range[1], grid_size)
        t = torch.linspace(t_range[0], t_range[1], grid_size)
        
        S, T = torch.meshgrid(s, t, indexing='ij')
        C = self.physics_law(S, T)

        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection='3d')
        
        # print(S.shape, T.shape, C.shape)   
        # raise
        ax.plot_surface(S.numpy(), T.numpy(), C.numpy(), cmap='plasma')
        # ax.contourf(S.numpy(), T.numpy(), C.numpy(), zdir='z', offset=-20, cmap='plasma')
        
        # fig.colorbar(im, shrink=0.5, aspect=5, pad=0.07)
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Option Price')
        ax.view_init(elev=15, azim=-125)
        plt.tight_layout()
        plt.show()
        
    def plot_true_solution(self, save_path=None):
        self.grids = 100
        s = torch.linspace(self.S_range[0], self.S_range[1], self.grids)
        t = torch.linspace(self.t_range[0], self.t_range[1], self.grids)
        
        S, T = torch.meshgrid(s, t, indexing='ij')
        C = self.physics_law(S, T)

        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection='3d')
        
        # print(S.shape, T.shape, C.shape)   
        # raise
        ax.plot_surface(S.numpy(), T.numpy(), C.numpy(), cmap='plasma')
        # ax.contourf(S.numpy(), T.numpy(), C.numpy(), zdir='z', offset=-20, cmap='plasma')
        
        # fig.colorbar(im, shrink=0.5, aspect=5, pad=0.07)
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Option Price')
        ax.view_init(elev=15, azim=-125)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'true_solution.png'))
        plt.close()
        
    def save_evaluation(self, model, save_path=None):
        preds_upper, preds_lower, preds_mean = model.summary()
        preds_upper = preds_upper.flatten().reshape(self.grids,self.grids).numpy()
        preds_lower = preds_lower.flatten().reshape(self.grids,self.grids).numpy()
        preds_mean = preds_mean.flatten().reshape(self.grids,self.grids).numpy()
        
        S_grid = self.eval_X[:,1].reshape(self.grids,self.grids).numpy()
        t_grid = 1-self.eval_X[:,0].reshape(self.grids,self.grids).numpy()
        
        np.savez(os.path.join(save_path, 'evaluation_data.npz'), preds_upper=preds_upper, preds_lower=preds_lower, preds_mean=preds_mean, S_grid=S_grid, t_grid=t_grid)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') 
        ax.plot_surface(S_grid,
                        t_grid, 
                        preds_mean, 
                        cmap='plasma')

        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Option Price')
        ax.view_init(elev=15, azim=-125)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pred_solution.png'))
        plt.close()
        
        true_price = self.physics_law(S_grid[:,0], t_grid[:,0])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(S_grid[:,0], preds_mean[:,0], label='Predicted Price')
        ax.plot(S_grid[:,0], true_price, label='True Price')
        ax.fill_between(S_grid[:,0], preds_upper[:,0], preds_lower[:,0], alpha=0.2, color='g', label='95% CI')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Option Price')
        ax.legend()
        plt.savefig(os.path.join(save_path, 'slice_prediction.png'))
        plt.close()
            
if __name__ == "__main__":
    
    call = EuropeanCall(K=40)
    
    # call.plot()

    
    bvp_x1, bvp_y1, bvp_x2,bvp_y2 = call.get_bvp_data(200)
    ivp_x1,ivp_y1 = call.get_ivp_data(200)
    diff_x1 = call.get_diff_data(800)
    plt.scatter(bvp_x1[:,0],bvp_x1[:,1], label= "BVP 1", color = "red",marker="o")
    plt.scatter(bvp_x2[:,0],bvp_x2[:,1], label= "BVP 2", color = "green",marker="x")
    plt.scatter(ivp_x1[:,0],ivp_x1[:,1], label= "IVP", color = "blue")
    plt.scatter(diff_x1[:,0],diff_x1[:,1], label= "PDE sample", color = "grey", alpha = 0.3)
    plt.xlabel("time to expiry, ")
    plt.ylabel("stock price ")
    plt.title("Data Sampling for European Call")
    plt.legend()
    plt.show()
        
    
