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
                 n_diff_sensors=100,
                 k = None,
                 C = None,
                 theta = None,
                #  is_inverse=True,
                #  sd_known=False,
                 ):
        super().__init__(t_start=t_start, 
                         t_end=t_end, 
                         n_diff_sensors=n_diff_sensors,
                         k=k,
                         C=C,
                         theta=theta,
                        #  is_inverse=is_inverse,
                        #  sd_known=sd_known,
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
    
    
    def differential_operator(self, model: torch.nn.Module, physics_X, pe_variables=None):
        if pe_variables is None:
            raise ValueError("pe_variables should be provided for inverse problems.")
        else:
            k = pe_variables[0].exp()
            C = pe_variables[1].exp()
            theta = pe_variables[2].exp()
        
        if self.k is not None:
            k = self.k * 60
        if self.C is not None:
            C = self.C / 8
        if self.theta is not None:
            theta = self.theta
        
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
        
        self.X = self.dataset[0]['X'].flatten() * 60
        self.y = self.dataset[0]['y'].flatten() * 8
        
        if self.physics_model.k is None:
            self.k_buffer = ScalarBuffer(burn=self.burn)
        if self.physics_model.C is None:
            self.C_buffer = ScalarBuffer(burn=self.burn)
        if self.physics_model.theta is None:
            self.theta_buffer = ScalarBuffer(burn=self.burn)
        
        try:
            self.sd_buffer = ScalarBuffer(burn=self.burn)
        except:
            pass
        # self.sd_buffer = ScalarBuffer(burn=self.burn)
    
    def _on_training(self):
        
        
        pred_y = self.model.net(self.eval_X).detach().cpu()
        self.eval_buffer.add(pred_y * 8)
        
        if self.physics_model.k is None:
            self.k_buffer.add(np.exp(self.model.pe_variables[0].detach().cpu().numpy()) / 60)
        if self.physics_model.C is None:
            self.C_buffer.add(np.exp(self.model.pe_variables[1].detach().cpu().numpy()) * 8)
        if self.physics_model.theta is None:
            self.theta_buffer.add(np.exp(self.model.pe_variables[2].detach().cpu().numpy()))

        try:
            self.sd_buffer.add(self.model.net.log_sd.exp().item() * 8)
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