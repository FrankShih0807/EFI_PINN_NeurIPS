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
'''
PoissonNonlinear: Poisson with parameter estimation 
'''

class Poisson2D(PhysicsModel):
    def __init__(self, 
                 t_start=-1.0,
                 t_end=1.0, 
                 boundary_sd=0.01,
                 sol_sd=0.01,
                 diff_sd=0.01,
                 n_boundary_sensors=25,
                 n_boundary_replicates=10,
                 n_sol_sensors=10,
                 n_sol_replicates=10,
                 n_diff_sensors=10,
                 n_diff_replicates=10,
                 k = 0.0,
                 is_inverse=False,
                 ):
        super().__init__(t_start=t_start, 
                         t_end=t_end, 
                         boundary_sd=boundary_sd,
                         sol_sd=sol_sd, 
                         diff_sd=diff_sd, 
                         n_boundary_sensors=n_boundary_sensors,
                         n_boundary_replicates=n_boundary_replicates,
                         n_sol_sensors=n_sol_sensors,
                         n_sol_replicates=n_sol_replicates,
                         n_diff_sensors=n_diff_sensors,
                         n_diff_replicates=n_diff_replicates,
                         k=k,
                         is_inverse=is_inverse
                         )
        if is_inverse:
            self.pe_dim = 1
        else:
            self.pe_dim = 0

    def generate_data(self, device):
        dataset = PINNDataset(device=device)
        if self.n_boundary_sensors > 0:
            bd_X, bd_y, bd_true_y = self.get_boundary_data()
            dataset.add_data(bd_X, bd_y, bd_true_y, category='solution', noise_sd=self.boundary_sd)
        if self.n_sol_sensors > 0:
            sol_X, sol_y, sol_true_y = self.get_sol_data()
            dataset.add_data(sol_X, sol_y, sol_true_y, category='solution', noise_sd=self.sol_sd)
        if self.n_diff_sensors > 0:
            diff_X, diff_y, diff_true_y = self.get_diff_data()
            dataset.add_data(diff_X, diff_y, diff_true_y, category='differential', noise_sd=self.diff_sd)
        eval_X, eval_y = self.get_eval_data()
        dataset.add_data(eval_X, eval_y, eval_y, category='evaluation', noise_sd=0)
        
        return dataset
    
    def get_eval_data(self):
        X1 = torch.linspace(self.t_start, self.t_end, steps=100)
        X2 = torch.linspace(self.t_start, self.t_end, steps=100)
        X1, X2 = torch.meshgrid(X1, X2)
        eval_X = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)
        eval_y = self.physics_law(eval_X)
        return eval_X, eval_y
    
    def get_boundary_data(self):
        
        X1_up = torch.linspace(self.t_start, self.t_end, steps=self.n_boundary_sensors+1)[:self.n_boundary_sensors]
        X2_up = torch.linspace(self.t_end, self.t_end, steps=self.n_boundary_sensors+1)[:self.n_boundary_sensors]
        X_up = torch.cat([X1_up.reshape(-1, 1), X2_up.reshape(-1, 1)], dim=1)
        
        X1_down = torch.linspace(self.t_start, self.t_end, steps=self.n_boundary_sensors+1)[1:]
        X2_down = torch.linspace(self.t_start, self.t_start, steps=self.n_boundary_sensors+1)[1:]
        X_down = torch.cat([X1_down.reshape(-1, 1), X2_down.reshape(-1, 1)], dim=1)
        
        X1_left = torch.linspace(self.t_start, self.t_start, steps=self.n_boundary_sensors+1)[:self.n_boundary_sensors]
        X2_left = torch.linspace(self.t_start, self.t_end, steps=self.n_boundary_sensors+1)[:self.n_boundary_sensors]
        X_left = torch.cat([X1_left.reshape(-1, 1), X2_left.reshape(-1, 1)], dim=1)
        
        X1_right = torch.linspace(self.t_end, self.t_end, steps=self.n_boundary_sensors+1)[1:]
        X2_right = torch.linspace(self.t_start, self.t_end, steps=self.n_boundary_sensors+1)[1:]
        X_right = torch.cat([X1_right.reshape(-1, 1), X2_right.reshape(-1, 1)], dim=1)
        
        boundary_X = torch.cat([X_up, X_down, X_left, X_right], dim=0).repeat_interleave(self.n_boundary_replicates, dim=0)
        true_boundary_y = self.physics_law(boundary_X)
        boundary_y = true_boundary_y + self.boundary_sd * torch.randn_like(true_boundary_y)

        return boundary_X, boundary_y, true_boundary_y
    
    def get_sol_data(self):
        X1 = torch.rand(self.n_sol_sensors, 1) * (self.t_end - self.t_start) + self.t_start
        X2 = torch.rand(self.n_sol_sensors, 1) * (self.t_end - self.t_start) + self.t_start
        X = torch.cat([X1, X2], dim=1).repeat_interleave(self.n_sol_replicates, dim=0)
        
        true_y = self.physics_law(X)
        y = true_y + self.sol_sd * torch.randn_like(true_y)
        return X, y, true_y

    
    def get_diff_data(self):
        X1 = torch.rand(self.n_diff_sensors, 1) * (self.t_end - self.t_start) + self.t_start
        X2 = torch.rand(self.n_diff_sensors, 1) * (self.t_end - self.t_start) + self.t_start
        X = torch.cat([X1, X2], dim=1).repeat_interleave(self.n_diff_replicates, dim=0)

        true_y = self.differential_function(X)
        y = true_y + self.diff_sd * torch.randn_like(true_y)
        return X, y, true_y
    
    def physics_law(self, X):
        y = torch.sin(torch.pi * X).prod(dim=1, keepdim=True)
        return y
    
    def differential_function(self, X):
        y = - 0.01 * torch.pi ** 2 * (torch.sin(torch.pi * X).prod(dim=1, keepdim=True)) + self.k * (torch.sin(torch.pi * X).prod(dim=1, keepdim=True)) ** 2
        return y
    
    def differential_operator(self, model: torch.nn.Module, physics_X, pe_variables=None):
        if pe_variables is None:
            k = self.k
        else:
            k = pe_variables[0]
            
        u = model(physics_X)
        grads = grad(u, physics_X)[0]
        dudx1 = grads[:, 0].view(-1, 1)
        dudx2 = grads[:, 1].view(-1, 1)
        u_x1x1 = grad(dudx1, physics_X)[0][:, 0].view(-1, 1)
        u_x2x2 = grad(dudx2, physics_X)[0][:, 1].view(-1, 1)
        
        # u_x = grad(u, physics_X)[0]
        # u_xx = grad(u_x, physics_X)[0]
        pde = 0.01 * u_x1x1 + 0.01 * u_x2x2 + k * u ** 2
        return pde

    def plot_true_solution(self, save_path=None):
        grids = 100
        X1 = torch.linspace(self.t_start, self.t_end, steps=grids)
        X2 = torch.linspace(self.t_start, self.t_end, steps=grids)
        X1, X2 = torch.meshgrid(X1, X2, indexing='ij')
        y = self.physics_law(torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1))
        y_reshaped = y.reshape(grids, grids)
        
        # sns.set_theme()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X1, X2, y.reshape(grids, grids), cmap='plasma')
        ax.contourf(X1, X2, y_reshaped, zdir='z', offset=y_reshaped.min()-0.5, cmap='plasma', alpha=0.7)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_zticks([-1, -0.5, 0, 0.5, 1])
        ax.view_init(elev=20, azim=-120)
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'true_solution.png'))
        else:
            plt.show()



class Poisson2DCallback(BaseCallback):
    def __init__(self):
        super().__init__()
    
    def _init_callback(self) -> None:
        self.eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        self.eval_y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        
        self.eval_X_cpu = self.eval_X.clone().detach().cpu()
        self.eval_y_cpu = self.eval_y.clone().detach().cpu()
        
        if self.physics_model.is_inverse:
            self.k_buffer = ScalarBuffer(burn=self.burn)

    
    def _on_training(self):
        pred_y = self.model.net(self.eval_X).detach().cpu()
        self.eval_buffer.add(pred_y)
        if self.physics_model.is_inverse:
            self.k_buffer.add(self.model.net.pe_variables[0].item())
        # print(len(self.eval_buffer))
        
    
    def _on_eval(self):
        pred_y_mean = self.eval_buffer.get_mean()
        ci_low, ci_high = self.eval_buffer.get_ci()
        ci_range = (ci_high - ci_low).mean().item()
        cr = ((ci_low <= self.eval_y_cpu.flatten()) & (self.eval_y_cpu.flatten() <= ci_high)).float().mean().item()
        mse = F.mse_loss(pred_y_mean, self.eval_y_cpu.flatten(), reduction='mean').item()
        
        self.logger.record('eval/ci_range', ci_range)
        self.logger.record('eval/coverage_rate', cr)
        self.logger.record('eval/mse', mse)
        if self.physics_model.is_inverse:
            k_mean = self.k_buffer.get_mean()
            k_low, k_high = self.k_buffer.get_ci()
            k_ci_range = k_high - k_low
            k_cr = ((k_low <= self.physics_model.k) & (self.physics_model.k <= k_high))
            
            self.logger.record('eval/k_ci_range', k_ci_range)
            self.logger.record('eval/k_coverage_rate', k_cr)
            self.logger.record('eval/k_mean', k_mean)
        
        self.save_evaluation()
        # self.plot_latent_Z()
        try:
            self.plot_latent_Z()
        except:
            pass    
        # self.physics_model.save_evaluation(self.model, self.save_path)
        # self.physics_model.save_temp_frames(self.model, self.n_evals, self.save_path)
    
    def _on_training_end(self) -> None:
        self.save_gif()
        pass
        
    def plot_latent_Z(self):
        true_y = self.dataset[0]['true_y'].flatten()
        sol_y = self.dataset[0]['y'].flatten()
        sd = self.dataset[0]['noise_sd']
        true_Z = sol_y - true_y
        
        latent_Z = self.model.latent_Z[0].flatten().detach().cpu().numpy()
        
        np.save(os.path.join(self.save_path, 'true_Z.npy'), true_Z)
        np.save(os.path.join(self.save_path, 'latent_Z.npy'), latent_Z)
        
        plt.subplots(figsize=(6, 6))
        plt.scatter(true_Z, latent_Z, label='Latent Z')
        plt.xlabel('True Z')
        plt.ylabel('Latent Z')
        plt.xlim(-3*sd, 3*sd)
        plt.ylim(-3*sd, 3*sd)
        plt.savefig(os.path.join(self.save_path, 'latent_Z.png'))
        plt.close()
        
        true_y = self.dataset[1]['true_y'].flatten()
        sol_y = self.dataset[1]['y'].flatten()
        sd = self.dataset[1]['noise_sd']
        true_Z = sol_y - true_y
        
        latent_Z = self.model.latent_Z[1].flatten().detach().cpu().numpy()
        
        np.save(os.path.join(self.save_path, 'true_Z_diff.npy'), true_Z)
        np.save(os.path.join(self.save_path, 'latent_Z_diff.npy'), latent_Z)
        
        plt.subplots(figsize=(6, 6))
        plt.scatter(true_Z, latent_Z, label='Latent Z')
        plt.xlabel('True Z')
        plt.ylabel('Latent Z')
        plt.xlim(-3*sd, 3*sd)
        plt.ylim(-3*sd, 3*sd)
        plt.savefig(os.path.join(self.save_path, 'latent_Z_diff.png'))
        plt.close()
        
    def save_evaluation(self):

        
        X = self.eval_X_cpu.numpy()
        # y = self.eval_y_cpu.numpy()
        grids = 100
        
        preds_mean = self.eval_buffer.get_mean()
        # preds_upper, preds_lower = self.eval_buffer.get_ci()

        X1 = X[:, 0].reshape(grids, grids)
        X2 = X[:, 1].reshape(grids, grids)
        # X1, X2 = torch.meshgrid(X1, X2, indexing='ij')
        # y = self.physics_law(torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1))
        y_reshaped = preds_mean.reshape(grids, grids)
        # print(X1.shape, X2.shape, y_reshaped.shape)
        # raise
        
        # sns.set_theme()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X1, X2, y_reshaped, cmap='plasma')
        ax.contourf(X1, X2, y_reshaped, zdir='z', offset=-1.5, cmap='plasma', alpha=0.7)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_zticks([-1, -0.5, 0, 0.5, 1])
        ax.view_init(elev=20, azim=-120)
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'pred_solution.png'))
        
        
            
            
        # sns.set_theme()
        # plt.subplots(figsize=(8, 6))
        # plt.plot(X, y, alpha=0.8, color='b', label='True')
        # plt.plot(X, preds_mean, alpha=0.8, color='g', label='Mean')
        # plt.plot(self.model.sol_X.clone().cpu().numpy() , self.model.sol_y.clone().cpu().numpy(), 'x', label='Training data', color='orange')
        
        # plt.fill_between(X, preds_upper, preds_lower, alpha=0.2, color='g', label='95% CI')
        # plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
        # plt.ylabel('u')
        # plt.xlabel('x')
        # plt.ylim(-1.5, 1.5)
        # plt.savefig(os.path.join(self.save_path, 'pred_solution.png'))
        
        
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
        
        
if __name__ == '__main__':
    # buffer = EvaluationBuffer()
    poisson = Poisson2D(is_inverse=False, n_boundary_replicates=2, n_boundary_sensors=10)


    poisson.plot_true_solution()
    