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
from matplotlib.lines import Line2D


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
                 initial_density=5,
                 y_scale=2000,
                 ):
        super().__init__(n_diff_sensors=n_diff_sensors,
                         k=k,
                         D=D,
                         R=R,
                         M=M,
                         initial_density=initial_density,
                         y_scale=y_scale,
                         )

        self.pe_dim = 3
        
        file_path = os.path.join(os.path.dirname(__file__), 'data','cell_density_profiles.npy')# Change this to the desired density index
        inputs, outputs, shape = load_cell_migration_data(file_path, self.initial_density, plot=False)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)
        
    def generate_data(self, device):
        dataset = PINNDataset(device=device)
        sol_X, sol_y, sol_true_y = self.get_sol_data()
        diff_X = sol_X.clone()
        diff_y = torch.zeros(sol_X.shape[0], 1)
        diff_true_y = torch.zeros(sol_X.shape[0], 1)
        
        diff_X, diff_y, diff_true_y = self.get_diff_data()
        eval_X, eval_y = self.get_eval_data()
        dataset.add_data(sol_X, sol_y, sol_true_y, category='solution', noise_sd=1.0)
        dataset.add_data(diff_X, diff_y, diff_true_y, category='differential', noise_sd=0)
        dataset.add_data(eval_X, eval_y, eval_y, category='evaluation', noise_sd=0)
        
        return dataset
    
    def get_eval_data(self):
        # file_path = os.path.join(os.path.dirname(__file__), 'data','porous_fisher_KPP_data.npy')
        # data = np.load(file_path, allow_pickle=True).item()
        # inputs = torch.tensor(data['inputs'], dtype=torch.float32)
        x_min = self.inputs[:, 0].min()
        x_max = self.inputs[:, 0].max()
        
        x = torch.linspace(x_min, x_max, 100)
        t = torch.linspace(0, 2, 5)
        
        # print(x.shape, t.shape)

        x_mesh, t_mesh = torch.meshgrid(x, t, indexing='ij')
        
        X = torch.cat([x_mesh.reshape(-1, 1), t_mesh.reshape(-1, 1)], dim=1)
        y = torch.zeros(X.shape[0], 1)
        return X, y
    
    def get_sol_data(self):
        

        X = self.inputs.clone()
        y = self.outputs.clone() / self.y_scale
        
        true_y = y.clone()
        
        return X, y, true_y
    
    def get_diff_data(self):
        
        x_min = self.inputs[:, 0].min()
        x_max = self.inputs[:, 0].max()
        

        
        if self.n_diff_sensors > 0:
            x = torch.rand(self.n_diff_sensors, 1) * (x_max - x_min) + x_min
            t = torch.rand(self.n_diff_sensors, 1) * 2
            X = torch.cat([x, t], dim=1)
            
            # X = torch.cat([X, self.inputs], dim=0)
        else:
            x = torch.linspace(x_min, x_max, 50)
            t = torch.linspace(0, 2, 10)
            x_mesh, t_mesh = torch.meshgrid(x, t, indexing='ij')
            X = torch.cat([x_mesh.reshape(-1, 1), t_mesh.reshape(-1, 1)], dim=1)
            
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
        
        pde = dudt - dTdx - R * (1 - (u/self.k)) * u
        
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
            y_plot = y[idx] * self.y_scale
            
            # Sort by x for smoother plotting
            sort_idx = np.argsort(x_plot)
            x_plot = x_plot[sort_idx]
            y_plot = y_plot[sort_idx]

            plt.scatter(x_plot, y_plot, marker=markers[i], color=colors[i], label=f"{t} days")

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
        
        self.X = self.dataset[0]['X']
        self.y = self.dataset[0]['y']

        
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
        
        self.logger.record_mean('train/gls_loss', self.GLS_loss().item())
        
        pred_y = self.model.net(self.eval_X).detach().cpu() * self.physics_model.y_scale
        self.eval_buffer.add(pred_y)
        
        if self.physics_model.D is None:
            self.D_buffer.add(np.exp(self.model.pe_variables[0].detach().cpu().numpy()))
        if self.physics_model.R is None:
            self.R_buffer.add(np.exp(self.model.pe_variables[1].detach().cpu().numpy()))
        if self.physics_model.M is None:
            self.M_buffer.add(np.exp(self.model.pe_variables[2].detach().cpu().numpy()))

        try:
            self.sd_buffer.add(self.model.net.log_sd.exp().item() * self.physics_model.y_scale)
        except:
            pass
        
    
    def _on_eval(self):
        ci_low, ci_high = self.eval_buffer.get_ci()
        ci_range = (ci_high - ci_low).mean().item()

        self.logger.record('eval/ci_range', ci_range)
            
        if self.physics_model.D is None:
            D_mean = self.D_buffer.get_mean()
            D_low, D_high = self.D_buffer.get_ci()
            self.logger.record('eval/D_low', D_low)
            self.logger.record('eval/D_high', D_high)
            self.logger.record('eval/D_mean', D_mean)
        
        if self.physics_model.R is None:
            R_mean = self.R_buffer.get_mean()
            R_low, R_high = self.R_buffer.get_ci()
            self.logger.record('eval/R_low', R_low)
            self.logger.record('eval/R_high', R_high)
            self.logger.record('eval/R_mean', R_mean)
        
        if self.physics_model.M is None:
            M_mean = self.M_buffer.get_mean()
            M_low, M_high = self.M_buffer.get_ci()
            self.logger.record('eval/M_low', M_low)
            self.logger.record('eval/M_high', M_high)
            self.logger.record('eval/M_mean', M_mean)
        
        try:
            sd_mean = self.sd_buffer.get_mean()
            sd_low, sd_high = self.sd_buffer.get_ci()
            self.logger.record('eval/sd_mean', sd_mean)
            self.logger.record('eval/sd_low', sd_low)
            self.logger.record('eval/sd_high', sd_high)
        except:
            pass
        
        self.save_evaluation()
        
        # if self.physics_model.k is None and self.physics_model.theta is None:
        #     self.plot_k_vs_theta()
        # try:
        #     self.plot_latent_Z()
        # except:
        #     pass    
        
        if self.model.progress <= self.eval_buffer.burn:
            self.eval_buffer.reset()
            if self.physics_model.D is None:
                self.D_buffer.reset()
            if self.physics_model.R is None:
                self.R_buffer.reset()
            if self.physics_model.M is None:
                self.M_buffer.reset()
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
        sol_X = self.X[:, 0].flatten().detach().cpu().numpy()
        latent_Z = self.model.latent_Z[0].flatten().detach().cpu().numpy()
        
        
        plt.subplots(figsize=(6, 6))
        plt.scatter(sol_X, latent_Z, label='Latent Z')
        plt.xlabel('Time')
        plt.ylabel('Latent Z')
        plt.savefig(os.path.join(self.save_path, 'latent_Z.png'))
        plt.close()
        
        
    def save_evaluation(self):

        # plt.ylim(-1.5, 1.5)

        X_eval = self.eval_X_cpu.numpy()
        x_eval = X_eval[:, 0]
        t_eval = X_eval[:, 1]
        y_eval = self.eval_buffer.get_mean().numpy()
        
        y_upper, y_lower = self.eval_buffer.get_ci()
        
        X_data = self.X.clone().cpu().numpy()
        y_data = self.y.clone().cpu().numpy() * self.physics_model.y_scale
        # preds_mean = self.eval_buffer.get_mean()
        # preds_upper, preds_lower = self.eval_buffer.get_ci()
        
        x_data = X_data[:, 0]
        t_data = X_data[:, 1]

        # Choose time points to visualize (e.g., 0, 0.5, 1.0, 1.5, 2.0)
        time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
        markers = ['x', 'o', 's', 'd', '^']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        labels = [f"{t} days" for t in time_points]

        plt.figure(figsize=(8, 5))

        for i, t in enumerate(time_points):
            idx = np.where(np.isclose(t_data, t, atol=1e-3))[0]
            x_plot = x_data[idx]
            y_plot = y_data[idx]
            
            idx_eval = np.where(np.isclose(t_eval, t, atol=1e-3))[0]
            x_plot_eval = x_eval[idx_eval]
            y_plot_eval = y_eval[idx_eval]
            
            # Sort by x for smoother plotting
            sort_idx = np.argsort(x_plot)
            x_plot = x_plot[sort_idx]
            y_plot = y_plot[sort_idx]
            
            sort_idx_eval = np.argsort(x_plot_eval)
            x_plot_eval = x_plot_eval[sort_idx_eval]
            y_plot_eval = y_plot_eval[sort_idx_eval]

            plt.scatter(x_plot, y_plot, marker=markers[i], color=colors[i])
            plt.plot(x_plot_eval, y_plot_eval, color=colors[i], linestyle='--', alpha=0.5)
            
            plt.fill_between(x_plot_eval, y_lower[idx_eval], y_upper[idx_eval], color=colors[i], alpha=0.2)
            # plt.plot(x_plot, y_plot, marker=markers[i], color=colors[i], label=f"{t} days", linestyle='-')

        legend_elements = [
            Line2D([0], [0], linestyle='-', marker=marker, color=color, label=label)
            for marker, color, label in zip(markers, colors, labels)
        ]
        
        # plt.title("P-FKPP Data")
        plt.xlabel("Position (mm)")
        plt.ylabel("Cell density (cells/mm²)")
        plt.legend(handles=legend_elements)
        plt.grid(True)
        plt.tight_layout()
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
        
    def GLS_loss(self):
        sol_X = self.X
        sol_y = self.y
        
        loss = torch.mean(((self.physics_model.y_scale * sol_y - self.physics_model.y_scale * self.model.net(sol_X)) / (self.physics_model.y_scale * self.model.net(sol_X).abs()).pow(0.2) )**2).detach()

        return loss
        
        
if __name__ == '__main__':
    model = PorousFKPP(initial_density=0)
    dataset = model.generate_data(device='cpu')
    
    for d in dataset:
        print(d['category'])
        print(d['X'].shape, d['y'].shape)
    X = dataset[0]['X']
    y = dataset[0]['y']
    
    # print(X.shape, y.shape)
    
    diff_X = dataset[1]['X']
    diff_y = dataset[1]['y']
    # print(diff_X, diff_y)
    
    eval_X = dataset[2]['X']
    eval_y = dataset[2]['y']
    
    pred_y = eval_X[:, 0:1] + eval_X[:, 1:2]
    # print(eval_X)
    # print(pred_y.reshape())
    
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