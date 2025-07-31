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
# import statsmodels.api as sm
'''
Linear1D: Linear model with parameter estimation 
'''

class Linear1D(PhysicsModel):
    def __init__(self, 
                 t_start=-1.0,
                 t_end=1.0, 
                 sol_sd=1.0,
                 diff_sd=0.0,
                 n_sol_sensors=10,
                 n_sol_replicates=1,
                 n_diff_sensors=100,
                 n_diff_replicates=1,
                 is_inverse=False,
                 ):
        super().__init__(t_start=t_start, 
                         t_end=t_end, 
                         sol_sd=sol_sd, 
                         diff_sd=diff_sd, 
                         n_sol_sensors=n_sol_sensors,
                         n_sol_replicates=n_sol_replicates,
                         n_diff_sensors=n_diff_sensors,
                         n_diff_replicates=n_diff_replicates,
                         is_inverse=is_inverse
                         )
        if is_inverse:
            self.pe_dim = 1
        else:
            self.pe_dim = 0

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
        X = torch.linspace(self.t_start, self.t_end, steps=self.n_sol_sensors).repeat_interleave(self.n_sol_replicates).view(-1, 1)
        true_y = self.physics_law(X)
        y = true_y + self.sol_sd * torch.randn_like(true_y)
        return X, y, true_y
    
    def get_diff_data(self):
        X = torch.linspace(self.t_start, self.t_end, steps=self.n_diff_sensors).repeat_interleave(self.n_diff_replicates).view(-1, 1)
        true_y = self.differential_function(X)
        y = true_y + self.diff_sd * torch.randn_like(true_y)
        return X, y, true_y
    
    def physics_law(self, X):
        y = 2*X - 1
        return y
    
    def differential_function(self, X):
        y = torch.zeros_like(X)
        return y
    
    def differential_operator(self, model: torch.nn.Module, physics_X, pe_variables=None):
        u = model(physics_X)
        u_x = grad(u, physics_X)[0]
        u_xx = grad(u_x, physics_X)[0]
        pde = u_xx
        
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
        else:
            plt.show()
        plt.close()
        



class Linear1DCallback(BaseCallback):
    def __init__(self):
        super().__init__()
    
    def _init_callback(self) -> None:
        self.eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        self.eval_y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        
        self.eval_X_cpu = self.eval_X.clone().detach().cpu()
        self.eval_y_cpu = self.eval_y.clone().detach().cpu()
        
        X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'solution'], dim=0).to('cpu')
        y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'solution'], dim=0).to('cpu')
        X_ols = torch.stack([torch.ones_like(X), X], dim=1).squeeze(dim=-1)

        sd = self.dataset[0]['noise_sd']
        # Solve OLS: β = (X^T X)^(-1) X^T y
        XtX_inv = torch.linalg.inv(X_ols.T @ X_ols)  # (X^T X)^(-1)
        beta_hat = XtX_inv @ X_ols.T @ y  # OLS coefficients

        # Predicted values
        y_pred = X_ols @ beta_hat

        # Compute residual standard error
        residuals = y - y_pred
        sigma_hat = torch.sqrt((residuals.T @ residuals) / (X.shape[0] - 2))

        # Compute standard errors of predictions
        X_diag = torch.einsum('ij,jk,ik->i', X_ols, XtX_inv, X_ols).reshape(-1,1)  # Variance of predictions
        
        se_pred = sd * torch.sqrt(X_diag)

        se0 = sd * torch.sqrt(XtX_inv[0, 0]).item()
        se1 = sd * torch.sqrt(XtX_inv[1, 1]).item()

        # Compute confidence interval (95% CI)
        t_value = 1.96  # Approximate for large samples
        y_lower = y_pred - t_value * se_pred
        y_upper = y_pred + t_value * se_pred
        
        self.ols_X = X.flatten().numpy()
        self.ols_y_pred = y_pred.flatten().numpy()
        self.ols_upper = y_upper.flatten().numpy()
        self.ols_lower = y_lower.flatten().numpy()
        self.ols_b0_upper = beta_hat[0].item() + t_value * se0
        self.ols_b0_lower = beta_hat[0].item() - t_value * se0
        self.ols_b1_upper = beta_hat[1].item() + t_value * se1
        self.ols_b1_lower = beta_hat[1].item() - t_value * se1

        self.b0_buffer = ScalarBuffer(burn=self.burn)
        self.b1_buffer = ScalarBuffer(burn=self.burn)
        
        self.eigenvals_buffer = EvaluationBuffer(burn=self.burn)

        if self.physics_model.is_inverse:
            self.k_buffer = ScalarBuffer(burn=self.burn)
        if self.model.net.sd_known==False:
            self.sd_buffer = ScalarBuffer(burn=self.burn)
    
    def _on_training(self):
        
        if self.model.net.sd_known==False:
            self.sd_buffer.add(self.model.net.log_sd.exp().item())
        
        pred_y = self.model.net(self.eval_X).detach().cpu()
        self.eval_buffer.add(pred_y)

        origin = torch.tensor([0.0], device=self.device).reshape(1, 1).requires_grad_()
        b0_hat = self.model.net(origin).flatten().detach().cpu().item()
        u = self.model.net(origin)
        b1_hat = grad(u, origin)[0].detach().cpu().item()

        self.b0_buffer.add(b0_hat)
        self.b1_buffer.add(b1_hat)
        
        m = self.model.net.encoder.m.clone().detach().cpu()
        mtm = m.T @ m / m.shape[0]

        eigenvals = torch.linalg.eigvalsh(mtm).sort()[0]
        
        # if min_eigenvalue < 0:
        #     print(f"Warning: Negative eigenvalue detected: {min_eigenvalue}")
        #     print(mtm)
        #     raise ValueError("Negative eigenvalue detected in the model's encoder matrix.")

        self.eigenvals_buffer.add(eigenvals)
        
        
        
        if self.physics_model.is_inverse:
            self.k_buffer.add(self.model.pe_variables[0].item())
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
        
        if self.model.net.sd_known==False:
            sd_mean = self.sd_buffer.get_mean()
            sd_low, sd_high = self.sd_buffer.get_ci()
            sd_ci_range = sd_high - sd_low
            sd_cr = ((sd_low <= self.physics_model.sol_sd) & (self.physics_model.sol_sd <= sd_high))
            
            self.logger.record('eval/sd_ci_range', sd_ci_range)
            self.logger.record('eval/sd_coverage_rate', sd_cr)
            self.logger.record('eval/sd_mean', sd_mean)
        
        
        pred_b0_mean = self.b0_buffer.get_mean()
        pred_b1_mean = self.b1_buffer.get_mean()
        b0_low, b0_high = self.b0_buffer.get_ci()
        b1_low, b1_high = self.b1_buffer.get_ci()
        
        
        self.logger.record('ci/ols_b0', '({:.2f}, {:.2f})'.format(self.ols_b0_lower, self.ols_b0_upper))
        self.logger.record('ci/ols_b1', '({:.2f}, {:.2f})'.format(self.ols_b1_lower, self.ols_b1_upper))
        self.logger.record('ci/efi_b0', '({:.2f}, {:.2f})'.format(b0_low, b0_high))
        self.logger.record('ci/efi_b1', '({:.2f}, {:.2f})'.format(b1_low, b1_high))

        # self.logger.record('eval/mm_min_eigenval', self.eigen_mm_buffer.last()[0])

        self.save_evaluation()
        self.plot_eigenvals()
        # self.plot_latent_Z()
        try:
            self.plot_latent_Z()
        except:
            pass    
        
        if self.model.progress <= self.eval_buffer.burn:
            self.eval_buffer.reset()
            self.b0_buffer.reset()
            self.b1_buffer.reset()
            self.eigenvals_buffer.reset()
            if self.physics_model.is_inverse:
                self.k_buffer.reset()
            if self.model.net.sd_known==False:
                self.sd_buffer.reset()
    
    def _on_training_end(self) -> None:
        self.save_gif()
        
    def plot_latent_Z(self):
        true_y = self.dataset[0]['true_y'].flatten()
        sol_y = self.dataset[0]['y'].flatten()
        sd = self.dataset[0]['noise_sd']
        true_Z = (sol_y - true_y) / sd
        
        latent_Z = self.model.latent_Z[0].flatten().detach().cpu().numpy()
        
        np.save(os.path.join(self.save_path, 'true_Z.npy'), true_Z)
        np.save(os.path.join(self.save_path, 'latent_Z.npy'), latent_Z)
        
        plt.subplots(figsize=(6, 6))
        plt.scatter(true_Z, latent_Z, label='Latent Z')
        plt.xlabel('True Z')
        plt.ylabel('Latent Z')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.savefig(os.path.join(self.save_path, 'latent_Z.png'))
        plt.close()
    
    def plot_eigenvals(self):
        eigenvals = torch.cat(self.eigenvals_buffer.memory, dim=0)
        
        for i in range(eigenvals.shape[1]):
            plt.figure(figsize=(8, 6))
            plt.plot(eigenvals[:, i].cpu().numpy(), label=f'Eigenvalue {i+1}')
            plt.xlabel('Sample')
            plt.ylabel(f'Eigenvalue {i+1}')
            plt.title(f'Eigenvalue {i+1} over time')
            plt.legend()
            plt.savefig(os.path.join(self.save_path, f'eigenvalue_{i+1}.png'))
            plt.close()
        
        
    def save_evaluation(self):
        X = self.eval_X_cpu.flatten().numpy()
        y = self.eval_y_cpu.flatten().numpy()
        
        preds_mean = self.eval_buffer.get_mean()
        preds_upper, preds_lower = self.eval_buffer.get_ci()
        
        sns.set_theme()
        plt.subplots(figsize=(8, 6))
        # plt.plot(X, y, alpha=0.8, color='r', label='True')
        plt.plot(self.model.sol_X.clone().cpu().numpy() , self.model.sol_y.clone().cpu().numpy(), 'x', label='Training data', color='orange')
        plt.plot(X, preds_mean, alpha=0.8, color='g', label='EFI')
        plt.fill_between(X, preds_upper, preds_lower, alpha=0.2, color='g', label='EFI 95% CI')

        plt.plot(self.ols_X, self.ols_y_pred, label='OLS', color='blue', linestyle='--', alpha=0.8)
        plt.plot(self.ols_X, self.ols_upper, label='OLS 95% CI', color='blue', linestyle=':', alpha=0.8)
        plt.plot(self.ols_X, self.ols_lower, color='blue', linestyle=':', alpha=0.8)

        # plt.fill_between(self.ols_X, self.ols_upper, self.ols_lower, alpha=0.1, color='blue', label='OLS 95% CI')
        
        plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
        plt.ylabel('y')
        plt.xlabel('x')
        # plt.ylim(-1.5, 1.5)
        plt.savefig(os.path.join(self.save_path, 'pred_solution.png'), dpi=300)


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
    model = Linear1D()
    # dataset = model.generate_data()
    model.plot_true_solution()
    dataset = model.generate_data('cpu')
    for d in dataset:
        print(d['category'], d['X'], d['y'])