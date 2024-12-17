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

'''
PoissonNonlinear: Poisson with parameter estimation 
'''

class PoissonNonlinear(PhysicsModel):
    def __init__(self, 
                 t_start=-0.7,
                 t_end=0.7, 
                 sol_sd=0.01,
                 diff_sd=0.01,
                 n_sol_sensors=10,
                 n_sol_replicates=10,
                 n_diff_sensors=10,
                 n_diff_replicates=10,
                 k = 0.7
                 ):
        super().__init__(t_start=t_start, 
                         t_end=t_end, 
                         sol_sd=sol_sd, 
                         diff_sd=diff_sd, 
                         n_sol_sensor=n_sol_sensors,
                         n_sol_replicates=n_sol_replicates,
                         n_diff_sensors=n_diff_sensors,
                         n_diff_replicates=n_diff_replicates,
                         k=k
                         )

    def generate_data(self, device):
        dataset = PINNDataset(device=device)
        sol_X, sol_y, sol_true_y = self.get_sol_data()
        diff_X, diff_y, diff_true_y = self.get_diff_data()
        eval_X, eval_y = self.get_eval_data()
        dataset.add_data(sol_X, sol_y, sol_true_y, category='solution', noise_sd=self.sol_sd)
        dataset.add_data(diff_X, diff_y, diff_true_y, category='differential', noise_sd=self.diff_sd)
        dataset.add_data(eval_X, eval_y, eval_y, category='evaluation', noise_sd=0)
        
        return dataset
    
    def get_eval_data(self):
        X = torch.linspace(self.t_start, self.t_end, steps=100).reshape(100, -1)
        y = self.physics_law(X)
        return X, y
    
    def get_sol_data(self):
        # X = torch.tensor([self.t_start, self.t_end]).repeat_interleave(self.n_sol_samples).view(-1, 1)
        X = torch.linspace(self.t_start, self.t_end, steps=self.n_sol_sensor).repeat_interleave(self.n_sol_replicates).view(-1, 1)
        true_y = self.physics_law(X)
        y = true_y + self.sol_sd * torch.randn_like(true_y)
        return X, y, true_y
    
    def get_diff_data(self):
        X = torch.linspace(self.t_start, self.t_end, steps=self.n_diff_sensors).repeat_interleave(self.n_diff_replicates).view(-1, 1)
        true_y = self.differential_function(X)
        y = true_y + self.diff_sd * torch.randn_like(true_y)
        return X, y, true_y
    
    def physics_law(self, X):
        y = torch.sin(6 * X) ** 3
        return y
    
    def differential_function(self, X):
        y = -1.08 * torch.sin(6 * X) * (torch.sin(6 * X) ** 2 - 2 * torch.cos(6 * X) ** 2) + self.k * torch.tanh(torch.sin(6 * X) ** 3)
        return y
    
    def differential_operator(self, model: torch.nn.Module, physics_X, k=None):
        
        if k is None:
            k = self.k
            
        u = model(physics_X)
        u_x = grad(u, physics_X)[0]
        u_xx = grad(u_x, physics_X)[0]
        pde = 0.01 * u_xx
        pde += k * torch.tanh(u)
        
        return pde

    def plot_true_solution(self, save_path=None):
        X = torch.linspace(self.t_start, self.t_end, steps=100)
        y = self.physics_law(X)
        
        sns.set_theme()
        plt.plot(X, y, label='Equation')
        plt.legend()
        plt.ylabel('u')
        plt.xlabel('x')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'true_solution.png'))
        plt.close()
        



class PoissonNonlinearCallback(BaseCallback):
    def __init__(self):
        super().__init__()
    
    def _init_callback(self) -> None:
        self.eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        self.eval_y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        
        self.eval_X_cpu = self.eval_X.clone().detach().cpu()
        self.eval_y_cpu = self.eval_y.clone().detach().cpu()

    
    def _on_training(self):
        pred_y = self.model.net(self.eval_X).detach().cpu()
        self.eval_buffer.add(pred_y)
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
        X = self.eval_X_cpu.flatten().numpy()
        y = self.eval_y_cpu.flatten().numpy()
        
        preds_mean = self.eval_buffer.get_mean()
        preds_upper, preds_lower = self.eval_buffer.get_ci()
        
        sns.set_theme()
        plt.subplots(figsize=(8, 6))
        plt.plot(X, y, alpha=0.8, color='b', label='True')
        plt.plot(X, preds_mean, alpha=0.8, color='g', label='Mean')
        plt.plot(self.model.sol_X.clone().cpu().numpy() , self.model.sol_y.clone().cpu().numpy(), 'x', label='Training data', color='orange')
        
        plt.fill_between(X, preds_upper, preds_lower, alpha=0.2, color='g', label='95% CI')
        plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
        plt.ylabel('u')
        plt.xlabel('x')
        plt.ylim(-1.5, 1.5)
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
        
        
