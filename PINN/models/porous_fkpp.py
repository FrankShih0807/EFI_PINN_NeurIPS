import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel
from PINN.common.utils import PINNDataset
from PIL import Image
from PINN.common.callbacks import BaseCallback
from PINN.common.buffers import EvaluationBuffer, ScalarBuffer
import random
from matplotlib import gridspec
import pandas as pd
from matplotlib import cm

def load_cell_migration_data(file_path, initial_density, plot=False, path=None):
    
    densities = ['dens_10000', 'dens_12000', 'dens_14000', 
                 'dens_16000', 'dens_18000', 'dens_20000']
    density = densities[initial_density]
    
    # load data
    file = np.load(file_path, allow_pickle=True).item()

    # extract data
    density = densities[initial_density]
    x = file[density]['x'].copy()[1:, :] 
    t = file[density]['t'].copy()
    X = file[density]['X'].copy()[1:, :]
    T = file[density]['T'].copy()[1:, :]
    U = file[density]['U_mean'].copy()[1:, :]
    shape = U.shape

    # variable scales
    x_scale = 1/1000 # micrometer -> millimeter
    t_scale = 1/24 # hours -> days
    u_scale = 1/(x_scale**2) # cells/um^2 -> cells/mm^2

    # scale variables
    x *= x_scale
    t *= t_scale
    X *= x_scale
    T *= t_scale
    U *= u_scale

    # flatten for MLP
    inputs = np.concatenate([X.reshape(-1)[:, None],
                             T.reshape(-1)[:, None]], axis=1)
    outputs = U.reshape(-1)[:, None]

    if plot:
        # plot surface
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(X, T, U, cmap=cm.coolwarm, alpha=1)
        ax.scatter(X.reshape(-1), T.reshape(-1), U.reshape(-1), s=5, c='k')
        plt.title('Initial density: '+density[5:])
        ax.set_xlabel('Position (millimeters)')
        ax.set_ylabel('Time (days)')
        ax.set_zlabel('Cell density (cells/mm^2)')
        ax.set_zlim(0, 2.2e3)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()
        
    return inputs, outputs, shape
'''
PorousFKPP model for tumor growth (real data)
'''
class PorousFKPP(PhysicsModel):
    def __init__(self, 
                 n_diff_sensors=200,
                 k = 1.7e3,
                 D = None,
                 R = None,
                 M = None,
                 ):
        super().__init__(n_diff_sensors=n_diff_sensors,
                         k=k,
                         D=D,
                         R=R,
                         M=M,
                         )

        self.pe_dim = 3
        

    def generate_data(self, device):
        dataset = PINNDataset(device=device)
        sol_X, sol_y, sol_true_y = self.get_sol_data()
        diff_X, diff_y, diff_true_y = self.get_diff_data()
        eval_X, eval_y = self.get_eval_data()
        dataset.add_data(sol_X, sol_y, sol_true_y, category='solution', noise_sd=1.0)
        dataset.add_data(diff_X, diff_y, diff_true_y, category='differential', noise_sd=0)
        dataset.add_data(eval_X, eval_y, eval_y, category='evaluation', noise_sd=0)
        
        return dataset
    
    def get_eval_data(self):
        x = torch.linspace(0, 2, 100)
        t = torch.linspace(0, 2, 5)
        
        # print(x.shape, t.shape)

        x_mesh, t_mesh = torch.meshgrid(x, t, indexing='ij')
        
        X = torch.cat([x_mesh.reshape(-1, 1), t_mesh.reshape(-1, 1)], dim=1)
        # print(X)
        y = torch.zeros(X.shape[0], 1)
        return X, y
    
    def get_sol_data(self):
        file_path = os.path.join(os.path.dirname(__file__), 'data','cell_density_profiles.npy')
        # file_path = os.path.join(os.path.dirname(__file__), 'data','porous_fisher_KPP_data.npy')
        initial_density = 5  # Change this to the desired density index
        inputs, outputs, shape = load_cell_migration_data(file_path, initial_density, plot=False)


        X = torch.tensor(inputs, dtype=torch.float32 )
        y = torch.tensor(outputs, dtype=torch.float32 )/1000
        
        true_y = y.clone()
        
        return X, y, true_y
    
    def get_diff_data(self):
        x = torch.rand(self.n_diff_sensors, 1) * 2
        t = torch.rand(self.n_diff_sensors, 1) * 2
        X = torch.cat([x, t], dim=1)
        true_y = torch.zeros(X.shape[0], 1)
        y = true_y.clone()
        return X, y, true_y
    
    
    def differential_operator(self, model: torch.nn.Module, physics_X, pe_variables=None):
        if pe_variables is None:
            raise ValueError("pe_variables should be provided for inverse problems.")
        else:
            D = pe_variables[0].exp()
            R = pe_variables[1].exp()
            M = pe_variables[2].exp()
        

        
        u = model(physics_X)
        du = grad(u, physics_X)[0]
        dudt = du[:, 1:2]
        dudx = du[:, 0:1]
        
        T = D * (u/self.k) ** M * dudx
        dTdx = grad(T, physics_X)[0][:, 0:1]
        
        # pde = p_t - k * p * (1 - (p / C).abs() ** theta)
        
        pde = dudt - dTdx + R * (1 - (u/self.k)) * u
        
        if torch.isnan(pde).any():
            print("u:", u)
            print("u_t:", dudt)
            print("pde:", pde)
            raise
        
        # print(k, C, theta)
        # print(p_t, p, pde)
        
        return pde
    
    def plot_data(self, X, y):

        x_vals = X[:, 0]
        t_vals = X[:, 1]

        # Choose time points to visualize (e.g., 0, 0.5, 1.0, 1.5, 2.0)
        time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
        markers = ['x', 'o', 's', 'd', '^']
        colors = ['blue', 'orange', 'green', 'red', 'purple']

        plt.figure(figsize=(8, 5))

        for i, t in enumerate(time_points):
            idx = np.where(np.isclose(t_vals, t, atol=1e-3))[0]
            x_plot = x_vals[idx]
            y_plot = y[idx]
            
            # Sort by x for smoother plotting
            sort_idx = np.argsort(x_plot)
            x_plot = x_plot[sort_idx]
            y_plot = y_plot[sort_idx]

            plt.plot(x_plot, y_plot, marker=markers[i], color=colors[i], label=f"{t} days", linestyle='-')

        plt.title("P-FKPP Data")
        plt.xlabel("Position (mm)")
        plt.ylabel("Cell density (cells/mm²)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class PorousFKPPCallback(BaseCallback):
    def __init__(self):
        super().__init__()
    
    def _init_callback(self) -> None:
        self.eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        self.eval_X_cpu = self.eval_X.clone().detach().cpu()
        
        self.X = self.dataset[0]['X'].flatten()
        self.y = self.dataset[0]['y'].flatten()
        
        if self.physics_model.D is None:
            self.D_buffer = ScalarBuffer(burn=self.burn)
        if self.physics_model.R is None:
            self.R_buffer = ScalarBuffer(burn=self.burn)
        if self.physics_model.M is None:
            self.M_buffer = ScalarBuffer(burn=self.burn)
        
        try:
            self.sd_buffer = ScalarBuffer(burn=self.burn)
        except:
            pass
        # self.sd_buffer = ScalarBuffer(burn=self.burn)
    
    def _on_training(self):
        
        
        pred_y = self.model.net(self.eval_X).detach().cpu()
        self.eval_buffer.add(pred_y)
        
        if self.physics_model.D is None:
            self.D_buffer.add(np.exp(self.model.pe_variables[0].detach().cpu().numpy()))
        if self.physics_model.R is None:
            self.R_buffer.add(np.exp(self.model.pe_variables[1].detach().cpu().numpy()))
        if self.physics_model.M is None:
            self.M_buffer.add(np.exp(self.model.pe_variables[2].detach().cpu().numpy()))

        try:
            self.sd_buffer.add(self.model.net.log_sd.exp().item())
        except:
            pass
        
    
    def _on_eval(self):
        ci_low, ci_high = self.eval_buffer.get_ci()
        ci_range = (ci_high - ci_low).mean().item()

        self.logger.record('eval/ci_range', ci_range)
            
        if self.physics_model.k is None:
            k_mean = self.k_buffer.get_mean()
            k_low, k_high = self.k_buffer.get_ci()
            self.logger.record('eval/k_low', k_low)
            self.logger.record('eval/k_high', k_high)
            self.logger.record('eval/k_mean', k_mean)
        
        if self.physics_model.C is None:
            C_mean = self.C_buffer.get_mean()
            C_low, C_high = self.C_buffer.get_ci()
            self.logger.record('eval/C_low', C_low)
            self.logger.record('eval/C_high', C_high)
            self.logger.record('eval/C_mean', C_mean)
        
        if self.physics_model.theta is None:
            theta_mean = self.theta_buffer.get_mean()
            theta_low, theta_high = self.theta_buffer.get_ci()
            self.logger.record('eval/theta_low', theta_low)
            self.logger.record('eval/theta_high', theta_high)
            self.logger.record('eval/theta_mean', theta_mean)
        
        try:
            sd_mean = self.sd_buffer.get_mean()
            sd_low, sd_high = self.sd_buffer.get_ci()
            self.logger.record('eval/sd_mean', sd_mean)
            self.logger.record('eval/sd_low', sd_low)
            self.logger.record('eval/sd_high', sd_high)
        except:
            pass
        
        self.save_evaluation()
        
        if self.physics_model.k is None and self.physics_model.theta is None:
            self.plot_k_vs_theta()
        try:
            self.plot_latent_Z()
        except:
            pass    
        
        if self.model.progress <= self.eval_buffer.burn:
            self.eval_buffer.reset()
            if self.physics_model.k is None:
                self.k_buffer.reset()
            if self.physics_model.C is None:
                self.C_buffer.reset()
            if self.physics_model.theta is None:
                self.theta_buffer.reset()
            try:
                self.sd_buffer.reset()
            except:
                pass

    
    def _on_training_end(self) -> None:
        self.save_gif()
        
    def plot_k_vs_theta(self):
        n = 100000
        k_samples = self.k_buffer.last(n)
        theta_samples = self.theta_buffer.last(n)
        plt.figure(figsize=(8, 6))
        plt.scatter(k_samples, theta_samples, alpha=0.8, color='b')
        plt.xlabel(r'$k$')
        plt.ylabel(r'$\theta$')
        plt.title(r'$k$ vs $\theta$')
        plt.savefig(os.path.join(self.save_path, 'k_vs_theta.png'))
        plt.close()
        
    def plot_latent_Z(self):
        sol_X = self.dataset[0]['X'].flatten() * 60
        latent_Z = self.model.latent_Z[0].flatten().detach().cpu().numpy()
        
        
        plt.subplots(figsize=(6, 6))
        plt.scatter(sol_X, latent_Z, label='Latent Z')
        plt.xlabel('Time')
        plt.ylabel('Latent Z')
        plt.savefig(os.path.join(self.save_path, 'latent_Z.png'))
        plt.close()
        
        
    def save_evaluation(self):
        X = self.eval_X_cpu.flatten().numpy() * 60
        
        preds_mean = self.eval_buffer.get_mean()
        preds_upper, preds_lower = self.eval_buffer.get_ci()
        
        sns.set_theme()
        plt.subplots(figsize=(8, 6))
        # plt.plot(X, y, alpha=0.8, color='b', label='True')
        plt.plot(X, preds_mean, alpha=0.8, color='g', label='Predicted')
        plt.scatter(self.X, self.y, marker='o', linestyle='-', color='r', label='Data')
        # plt.plot(self.model.sol_X.clone().cpu().numpy() , self.model.sol_y.clone().cpu().numpy(), 'x', label='Training data', color='orange')
        
        plt.fill_between(X, preds_upper, preds_lower, alpha=0.2, color='g', label='95% CI')
        plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
        plt.ylabel('Volume')
        plt.xlabel('Time')
        # plt.ylim(-1.5, 1.5)
        plt.savefig(os.path.join(self.save_path, 'pred_solution.png'))
        
        
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
        
        
if __name__ == '__main__':
    model = PorousFKPP()
    dataset = model.generate_data(device='cpu')
    
    X = dataset[0]['X']
    y = dataset[0]['y']
    
    # print(X.shape, y.shape)
    
    diff_X = dataset[1]['X']
    diff_y = dataset[1]['y']
    # print(diff_X, diff_y)
    
    model.plot_data(X, y)
    
    # sns.set_theme()
    # plt.figure(figsize=(10, 6))
    # plt.scatter(X, y, marker='o', linestyle='-', color='r', label='Data')
    # plt.xlabel('Time')
    # plt.ylabel('Volume')
    # plt.title('Tumor Cell Data Over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # diff_X = dataset[1]['X']
    # diff_y = dataset[1]['y']
    
    # print(diff_X) 
    # print(diff_y)