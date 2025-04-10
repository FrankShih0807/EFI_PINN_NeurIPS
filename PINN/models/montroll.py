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


'''
Montroll model for tumor growth (real data)
'''
class Montroll(PhysicsModel):
    def __init__(self, 
                 t_start=0.0,
                 t_end=60.0, 
                 sol_sd=1,
                 n_diff_sensors=100,
                 is_inverse=True,
                 sd_known=False,
                 ):
        super().__init__(t_start=t_start, 
                         t_end=t_end, 
                         sol_sd=sol_sd, 
                         n_diff_sensors=n_diff_sensors,
                         is_inverse=is_inverse,
                         sd_known=sd_known,
                         )

        self.pe_dim = 3

    def generate_data(self, device):
        dataset = PINNDataset(device=device)
        sol_X, sol_y, sol_true_y = self.get_sol_data()
        diff_X, diff_y, diff_true_y = self.get_diff_data()
        eval_X, eval_y = self.get_eval_data()
        dataset.add_data(sol_X, sol_y, sol_true_y, category='solution', noise_sd=self.sol_sd)
        dataset.add_data(diff_X, diff_y, diff_true_y, category='differential', noise_sd=0)
        dataset.add_data(eval_X, eval_y, eval_y, category='evaluation', noise_sd=0)
        
        return dataset
    
    def get_eval_data(self):
        X = torch.linspace(self.t_start, self.t_end, steps=100).reshape(100, -1)
        y = torch.zeros_like(X)
        return X, y
    
    def get_sol_data(self):
        csv_path = os.path.join(os.path.dirname(__file__), 'Tumor_Cell_Data.csv')
        tumor_cell_data = pd.read_csv(csv_path)
        X = torch.tensor(tumor_cell_data['t'].values, dtype=torch.float32 ).reshape(-1, 1)
        y = torch.tensor(tumor_cell_data['V'].values, dtype=torch.float32 ).reshape(-1, 1)
        
        X = X / 60.0
        y = y / 8
        true_y = y.clone()
        
        return X, y, true_y
    
    def get_diff_data(self):
        X = torch.linspace(self.t_start, self.t_end, steps=self.n_diff_sensors).view(-1, 1)
        true_y = torch.zeros_like(X)
        y = true_y.clone()
        return X, y, true_y
    
    def physics_law(self, X):
        y = torch.sin(6 * X) ** 3
        return y
    
    def differential_operator(self, model: torch.nn.Module, physics_X, pe_variables=None):
        if pe_variables is None:
            raise ValueError("pe_variables should be provided for inverse problems.")
        else:
            k = pe_variables[0].exp()
            C = pe_variables[1].exp()
            theta = pe_variables[2].exp()
            
        
        p = model(physics_X)
        p_t = grad(p, physics_X)[0]
        
        pde = p_t - k * p * (1 - (p / C).abs() ** theta)
        
        if torch.isnan(pde).any():
            print("p:", p)
            print("p_t:", p_t)
            print("pde:", pde)
            raise
        
        # print(k, C, theta)
        # print(p_t, p, pde)
        
        return pde


class MontrollCallback(BaseCallback):
    def __init__(self):
        super().__init__()
    
    def _init_callback(self) -> None:
        self.eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        self.eval_X_cpu = self.eval_X.clone().detach().cpu()
        
        self.X = self.dataset[0]['X'].flatten()
        self.y = self.dataset[0]['y'].flatten()
        

        if self.physics_model.is_inverse:
            self.k_buffer = ScalarBuffer(burn=self.burn)
            self.C_buffer = ScalarBuffer(burn=self.burn)
            self.theta_buffer = ScalarBuffer(burn=self.burn)
        if self.model.net.sd_known==False:
            self.sd_buffer = ScalarBuffer(burn=self.burn)
    
    def _on_training(self):
        
        
        pred_y = self.model.net(self.eval_X).detach().cpu()
        self.eval_buffer.add(pred_y)
        if self.physics_model.is_inverse:
            self.k_buffer.add(self.model.net.pe_variables[0].exp().item())
            self.C_buffer.add(self.model.net.pe_variables[1].exp().item())
            self.theta_buffer.add(self.model.net.pe_variables[2].exp().item())
        if self.model.net.sd_known==False:
            self.sd_buffer.add(self.model.net.log_sd.exp().item())
        
    
    def _on_eval(self):
        pred_y_mean = self.eval_buffer.get_mean()
        ci_low, ci_high = self.eval_buffer.get_ci()
        ci_range = (ci_high - ci_low).mean().item()
        
        self.logger.record('eval/ci_range', ci_range)
        if self.physics_model.is_inverse:
            k_mean = self.k_buffer.get_mean()
            k_low, k_high = self.k_buffer.get_ci()
            k_ci_range = k_high - k_low
            
            self.logger.record('eval/k_ci_range', k_ci_range)
            self.logger.record('eval/k_mean', k_mean)
            
            C_mean = self.C_buffer.get_mean()
            C_low, C_high = self.C_buffer.get_ci()
            C_ci_range = C_high - C_low

            self.logger.record('eval/C_ci_range', C_ci_range)
            self.logger.record('eval/C_mean', C_mean)
            
            theta_mean = self.theta_buffer.get_mean()
            theta_low, theta_high = self.theta_buffer.get_ci()
            theta_ci_range = theta_high - theta_low
            
            self.logger.record('eval/theta_ci_range', theta_ci_range)
            self.logger.record('eval/theta_mean', theta_mean)
        
        if self.model.net.sd_known==False:
            sd_mean = self.sd_buffer.get_mean()
            sd_low, sd_high = self.sd_buffer.get_ci()
            sd_ci_range = sd_high - sd_low

            
            self.logger.record('eval/sd_ci_range', sd_ci_range)
            self.logger.record('eval/sd_mean', sd_mean)
        
        self.save_evaluation()
        # self.plot_latent_Z()
        try:
            self.plot_latent_Z()
        except:
            pass    
        
        if self.model.progress <= self.eval_buffer.burn:
            self.eval_buffer.reset()
            if self.physics_model.is_inverse:
                self.k_buffer.reset()
                self.C_buffer.reset()
                self.theta_buffer.reset()
            if self.model.net.sd_known==False:
                self.sd_buffer.reset()
        # self.physics_model.save_evaluation(self.model, self.save_path)
        # self.physics_model.save_temp_frames(self.model, self.n_evals, self.save_path)
    
    def _on_training_end(self) -> None:
        self.save_gif()
        
    def plot_latent_Z(self):
        # true_y = self.dataset[0]['true_y'].flatten()
        sol_X = self.dataset[0]['X'].flatten()
        sol_y = self.dataset[0]['y'].flatten()
        
        sd_hat = self.model.net.log_sd.exp().item()
        # true_Z = (sol_y - true_y) / sd
        # true_Z = (sol_y - true_y)
        
        latent_Z = self.model.latent_Z[0].flatten().detach().cpu().numpy() * sd_hat
        
        # np.save(os.path.join(self.save_path, 'true_Z.npy'), true_Z)
        # np.save(os.path.join(self.save_path, 'latent_Z.npy'), latent_Z)
        
        plt.subplots(figsize=(6, 6))
        plt.scatter(sol_X, latent_Z, label='Latent Z')
        plt.xlabel('Time')
        plt.ylabel('Latent Z')
        # plt.xlim(-3, 3)
        # plt.ylim(-3, 3)
        plt.savefig(os.path.join(self.save_path, 'latent_Z.png'))
        plt.close()
        
        
    def save_evaluation(self):
        X = self.eval_X_cpu.flatten().numpy()
        
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

        # if self.model.net.sd_known == False:
        #     sigma_samples = self.sd_buffer.samples
        #     sns.set_style("whitegrid")
            
        #     fig = plt.figure(figsize=(10, 6))
        #     gs = gridspec.GridSpec(1, 4)  # 4 columns: 3 for line plot, 1 for histogram

        #     # Line plot (left, wider)
        #     ax_main = plt.subplot(gs[0, :3])
        #     ax_main.plot(sigma_samples, color='steelblue')
        #     ax_main.set_xlabel('Iteration')
        #     ax_main.set_ylabel('Sigma')
        #     ax_main.set_title('Sigma Trace')

        #     # Histogram (right, narrow)
        #     ax_hist = plt.subplot(gs[0, 3], sharey=ax_main)
        #     ax_hist.hist(sigma_samples, bins=30, orientation='horizontal', density=True, color='lightcoral', edgecolor='black')
        #     ax_hist.set_xlabel('Density')
        #     ax_hist.tick_params(labelleft=False)  # hide y-tick labels to avoid clutter

        #     plt.tight_layout()
        #     plt.savefig(os.path.join(self.save_path, 'sigma.png'))
        #     plt.close()
        
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
    model = Montroll()
    dataset = model.generate_data(device='cpu')
    
    X = dataset[0]['X'].flatten().numpy()
    y = dataset[0]['y'].flatten().numpy()
    
    
    sns.set_theme()
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, marker='o', linestyle='-', color='r', label='Data')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Tumor Cell Data Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # diff_X = dataset[1]['X']
    # diff_y = dataset[1]['y']
    
    # print(diff_X) 
    # print(diff_y)