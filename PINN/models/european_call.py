import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import seaborn as sns
import os

from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel
from PINN.common.utils import PINNDataset
from PIL import Image
from PINN.common.callbacks import BaseCallback

    
        
class EuropeanCall(PhysicsModel):
    def __init__(self, 
                 S_range = [0, 160],
                 t_range = [0, 1],
                 sigma = 0.5,
                 r = 0.05,
                 K = 80,
                 noise_sd=1,
                 n_price_sensors=5,
                 n_price_replicates=10,
                 n_boundary_samples=50,
                 n_diff_samples=400,
                 is_inverse=False,
                 ):
        self.norm_dist = dist.Normal(0, 1)
        super().__init__(S_range=S_range, 
                         t_range=t_range, 
                         sigma=sigma, 
                         r=r, 
                         K=K, 
                         noise_sd=noise_sd,
                         n_price_sensors=n_price_sensors,
                         n_price_replicates=n_price_replicates,
                         n_boundary_samples=n_boundary_samples,
                         n_diff_samples=n_diff_samples,
                         is_inverse=is_inverse
                         )
        if is_inverse:
            self.pe_dim = 1
        else:
            self.pe_dim = 0

    def generate_data(self, device):
        dataset = PINNDataset(device)
        # get solution data
        price_X, price_y, true_price_y = self.get_price_data()
        # get boundary data
        boundary_X, boundary_y = self.get_boundary_data()
        # get differential data
        diff_X, diff_y = self.get_diff_data()
        # get evaluation data
        eval_X, eval_y = self.get_eval_data()
        
        dataset.add_data(price_X, price_y, true_price_y, 'solution', self.noise_sd)
        dataset.add_data(boundary_X, boundary_y, boundary_y, 'solution', 0.0)
        dataset.add_data(diff_X, diff_y, diff_y, 'differential', 0.0)
        dataset.add_data(eval_X, eval_y, eval_y, 'evaluation', 0.0)
        return dataset
    
    def get_diff_data(self):
        ts = torch.rand(self.n_diff_samples, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        Ss = torch.rand(self.n_diff_samples, 1) * (self.S_range[1] - self.S_range[0]) + self.S_range[0]
        X = torch.cat([ts, Ss], dim=1)
        y = torch.zeros(self.n_diff_samples, 1)
        return X, y
    
    def get_eval_data(self):
        eval_t = torch.linspace(self.t_range[0], self.t_range[1], 100)
        eval_S = torch.linspace(self.S_range[0], self.S_range[1], 100)
        S, T = torch.meshgrid(eval_S, eval_t, indexing='ij')
        eval_X = torch.cat([T.reshape(-1, 1), S.reshape(-1, 1)], dim=1)
        eval_y = self.physics_law(S, self.t_range[1] - T).reshape(-1, 1)
        return eval_X, eval_y
    
    def get_price_data(self):
        # ts = torch.rand(n_samples, 1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        # Ss = torch.ones(n_samples, 1) * self.S_range[1]/2
        
        # ts = torch.zeros(n_samples).reshape(-1, 1)
        Ss = torch.linspace(self.S_range[0], self.S_range[1], self.n_price_sensors + 1)[1::].repeat_interleave(self.n_price_replicates).reshape(-1, 1)
        ts = torch.zeros_like(Ss)
        
        X = torch.cat([ts, Ss], dim=1)
        true_y = self.physics_law(Ss, self.t_range[1] - ts)
        y = true_y + self.noise_sd * torch.randn_like(true_y)
        return X, y, true_y
    
    def get_boundary_data(self):
        # Boundary condition at S = 0
        
        t1 = torch.linspace(self.t_range[0], self.t_range[1], self.n_boundary_samples).reshape(-1, 1)
        S1 = torch.ones(self.n_boundary_samples, 1) * self.S_range[0]
        X1 = torch.cat([t1, S1], dim=1)
        y1 = torch.zeros(self.n_boundary_samples, 1)
        
        # Boundary condition at time to maturity
        
        t2 = torch.ones(self.n_boundary_samples, 1) * self.t_range[1]
        S2 = torch.linspace(self.S_range[0], self.S_range[1], self.n_boundary_samples).reshape(-1, 1)
        X2 = torch.cat([t2, S2], dim=1)
        y2 = F.relu(S2 - self.K)
        
        boundary_X = torch.cat([X1, X2], dim=0)
        boundary_y = torch.cat([y1, y2], dim=0)
        return boundary_X, boundary_y
    
    def physics_law(self, s, t2m)->torch.Tensor:
        s = torch.as_tensor(s)
        t2m = torch.as_tensor(t2m)
        d1 = (torch.log(s/self.K) + (self.r + self.sigma**2/2) * (t2m)) / (self.sigma * torch.sqrt(t2m))
        d2 = d1 - self.sigma * torch.sqrt(t2m)
        Nd1 = self.norm_dist.cdf(d1)
        Nd2 = self.norm_dist.cdf(d2)
        V = s * Nd1 - self.K * torch.exp(-self.r * (t2m)) * Nd2
        return V

    def differential_operator(self, model: torch.nn.Module, physics_X, pe_variables=None):
        ''' Compute the Black-Scholes loss
        Args:
            model (torch.nn.Module): torch network model
        '''
        # self.physics_X = self.get_diff_data(800).requires_grad_(True)
        if pe_variables is None:
            r = self.r
        else:
            r = pe_variables[0]
        
        y_pred = model(physics_X)
        grads = grad(y_pred, physics_X)[0]
        dVdt = grads[:, 0].view(-1, 1)
        dVdS = grads[:, 1].view(-1, 1)
        grads2nd = grad(dVdS, physics_X)[0]
        d2VdS2 = grads2nd[:, 1].view(-1, 1)
        S1 = physics_X[:, 1].view(-1, 1)
        bs_pde = dVdt + 0.5 * self.sigma**2 * S1**2 * d2VdS2 + r * S1 * dVdS - r * y_pred
        
        return bs_pde
    
    ##########################
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
        
    ##########################    
    def save_evaluation(self, model, save_path=None):
        # preds_upper, preds_lower, preds_mean = model.summary()
        pred_dict = model.summary()
        
        preds_upper = pred_dict['y_preds_upper'].flatten().reshape(self.grids,self.grids).numpy()
        preds_lower =pred_dict['y_preds_lower'].flatten().reshape(self.grids,self.grids).numpy()
        preds_mean = pred_dict['y_preds_mean'].flatten().reshape(self.grids,self.grids).numpy()
        
        S_grid = model.eval_X[:,1].reshape(self.grids,self.grids).numpy()
        t_grid = 1-model.eval_X[:,0].reshape(self.grids,self.grids).numpy()
        
        # np.savez(os.path.join(save_path, 'evaluation_data.npz'), preds_upper=preds_upper, preds_lower=preds_lower, preds_mean=preds_mean, S_grid=S_grid, t_grid=t_grid)
        np.savez(os.path.join(save_path, 'evaluation_data.npz') , **pred_dict, S_grid=S_grid, t_grid=t_grid)
        
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

class EuropeanCallCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _init_callback(self) -> None:
        self.eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        self.eval_y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        
        self.eval_X_cpu = self.eval_X.clone().detach().cpu()
        self.eval_y_cpu = self.eval_y.clone().detach().cpu()

        self.grids = self.physics_model.grids

    def _on_training(self):
        pred_y = self.model.net(self.eval_X).detach().cpu()
        # print(pred_y)
        # raise
        self.eval_buffer.add(pred_y)

    def _on_eval(self):
        pred_y_mean = self.eval_buffer.get_mean()

        ci_low, ci_high = self.eval_buffer.get_ci()
        ci_range = (ci_high - ci_low).mean().item()
        cr = ((ci_low <= self.eval_y_cpu.flatten()) & (self.eval_y_cpu.flatten() <= ci_high)).float().mean().item()
        mse = F.mse_loss(pred_y_mean, self.eval_y_cpu.flatten(), reduction='mean').item()
        
        self.logger.record('eval/ci_range', ci_range)
        self.logger.record('eval/coverage_rate', cr)
        self.logger.record('eval/mse', mse)
        
        self.save_evaluation()
        try:
            self.plot_latent_Z()
        except:
            pass    
        
    def _on_training_end(self):
        self.save_gif()
    
    def plot_latent_Z(self):
        true_y = self.dataset[0]['true_y'].flatten()
        sol_y = self.dataset[0]['y'].flatten()
        true_Z = sol_y - true_y
        
        latent_Z = self.model.latent_Z[0].flatten().detach().cpu().numpy()
        
        np.save(os.path.join(self.save_path, 'true_Z.npy'), true_Z)
        np.save(os.path.join(self.save_path, 'latent_Z.npy'), latent_Z)
        
        plt.subplots(figsize=(6, 6))
        plt.scatter(true_Z, latent_Z, label='Latent Z')
        plt.xlabel('True Z')
        plt.ylabel('Latent Z')
        plt.xlim(-2.0, 2.0)
        plt.ylim(-2.0, 2.0)
        plt.savefig(os.path.join(self.save_path, 'latent_Z.png'))
        plt.close()

    def save_evaluation(self):
        subset_indices = torch.arange(0, self.grids * self.grids, self.grids)

        S = self.eval_X_cpu[:,1].reshape(self.grids,self.grids).numpy()
        S_eval = S[:,0]
        # t = 1 - self.eval_X_cpu[:,0].reshape(self.grids,self.grids).numpy()
        # t_eval = t[:,0]
        # X = self.eval_X_cpu.flatten().numpy()
        # y = self.eval_y_cpu.flatten().numpy()
        # true_price = self.physics_law(S_eval, t_eval)
        true_price = self.eval_y_cpu[subset_indices,:].flatten().numpy()

        preds_mean = self.eval_buffer.get_mean()
        preds_upper, preds_lower = self.eval_buffer.get_ci()


        sns.set_theme()
        plt.subplots(figsize=(8, 6))
        plt.plot(S_eval, true_price, alpha=0.8, color='b', label='True')
        plt.plot(S_eval, preds_mean[subset_indices], alpha=0.8, color='g', label='Mean')
        # plt.plot(self.model.sol_X.clone().cpu().numpy() , self.model.sol_y.clone().cpu().numpy(), 'x', label='Training data', color='orange')

        plt.fill_between(S_eval, preds_upper[subset_indices], preds_lower[subset_indices], alpha=0.2, color='g', label='95% CI')
        plt.xlabel('Stock Price')
        plt.ylabel('Option Price')
        plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
        plt.savefig(os.path.join(self.save_path, 'slice_prediction.png'))

        # save temp frames
        temp_dir = os.path.join(self.save_path, 'temp_frames')
        os.makedirs(temp_dir, exist_ok=True)
        frame_path = os.path.join(temp_dir, f"frame_{self.n_evals}.png")
        plt.savefig(frame_path)
        
        plt.close()

    def save_gif(self):
        frames = []
        temp_dir = os.path.join(self.save_path, 'temp_frames')
        n_frames = len(os.listdir(temp_dir))
        for epoch in range(n_frames):
            frame_path = os.path.join(temp_dir, f"frame_{epoch}.png")
            frames.append(Image.open(frame_path))
        # frame_files = sorted(os.listdir(temp_dir))  # Sort by file name to maintain order
        # print(frame_files)
        # frames = [Image.open(os.path.join(temp_dir, frame)) for frame in frame_files]
        
        frames[0].save(
            os.path.join(self.save_path, "training_loss.gif"),
            save_all=True,
            append_images=frames[1:],
            duration=500,
            loop=0
        )
        for frame_path in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, frame_path))
        os.rmdir(temp_dir)



if __name__ == "__main__":
    
    call = EuropeanCall(K=40)
    
    # call.plot()
        # get solution data
    price_X, price_y, true_price_y = call.get_price_data()
    # get boundary data
    boundary_X, boundary_y = call.get_boundary_data()
    # get differential data
    diff_X, diff_y = call.get_diff_data()
    
    plt.scatter(price_X[:,0], price_X[:,1], label= "Price", color = "red", marker="x")
    plt.scatter(boundary_X[:,0],boundary_X[:,1], label= "Boundary", color = "green")
    plt.scatter(diff_X[:,0],diff_X[:,1], label= "Differential", color = "blue")
    
    # plt.scatter(bvp_x1[:,0],bvp_x1[:,1], label= "BVP 1", color = "red",marker="o")
    # plt.scatter(bvp_x2[:,0],bvp_x2[:,1], label= "BVP 2", color = "green",marker="x")
    # plt.scatter(ivp_x1[:,0],ivp_x1[:,1], label= "IVP", color = "blue")
    # plt.scatter(diff_x1[:,0],diff_x1[:,1], label= "PDE sample", color = "grey", alpha = 0.3)
    plt.xlabel("time to expiry, ")
    plt.ylabel("stock price ")
    plt.title("Data Sampling for European Call")
    plt.legend()
    plt.show()
        
    
