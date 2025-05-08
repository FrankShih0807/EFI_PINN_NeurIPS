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
'''
PoissonNonlinear: Poisson with parameter estimation 
'''

class TaylorGreen(PhysicsModel):
    def __init__(self, 
                 t_start=0.0,
                 t_end=1.0, 
                 sol_sd=0.1,
                 diff_sd=0.0,
                 n_sol=10,
                 n_diff=10,
                 nu = 0.01,
                 is_inverse=False,
                 ):
        super().__init__(t_start=t_start, 
                         t_end=t_end, 
                         sol_sd=sol_sd, 
                         diff_sd=diff_sd, 
                         n_sol=n_sol,
                         n_diff=n_diff,
                         nu=nu,
                         is_inverse=is_inverse
                         )
        if is_inverse:
            self.pe_dim = 1
        else:
            self.pe_dim = 0

    def generate_data(self, device):
        dataset = PINNDataset(device=device)
        if self.n_sol > 0:
            sol_X, sol_y, sol_true_y = self.get_sol_data()
            dataset.add_data(sol_X, sol_y, sol_true_y, category='solution', noise_sd=self.sol_sd)
        if self.n_diff > 0:
            diff_X, diff_y, diff_true_y = self.get_diff_data()
            dataset.add_data(diff_X, diff_y, diff_true_y, category='differential', noise_sd=self.diff_sd)
        eval_X, eval_Y = self.get_eval_data()
        dataset.add_data(eval_X, eval_Y, eval_Y, category='evaluation', noise_sd=0)
        
        return dataset
    
    def get_eval_data(self):
        t = torch.linspace(self.t_start, self.t_end, steps=3)
        x = torch.linspace(-1, 1, steps=101)
        y = torch.linspace(-1, 1, steps=101)
        
        
        
        t, x, y = torch.meshgrid(t, x, y, indexing='ij')
        t, x, y = t.reshape(-1, 1), x.reshape(-1, 1), y.reshape(-1, 1)
        eval_X = torch.cat([t, x, y], dim=1)
        eval_Y = self.taylor_green_solution(eval_X)
        
        return eval_X, eval_Y
    
    
    def get_sol_data(self):
        x_ic = 2 * torch.rand(self.n_sol, 1) - 1
        y_ic = 2 * torch.rand(self.n_sol, 1) - 1
        t_ic = torch.zeros_like(x_ic)
        txy_ic = torch.cat([t_ic, x_ic, y_ic], dim=1)
        uvp_ic = self.taylor_green_solution(txy_ic)
        uv_ic = uvp_ic[:, 0:2]

        # uvp_ic_noisy = uvp_ic + self.sol_sd * torch.randn_like(uvp_ic)
        uv_ic_noisy = uv_ic + self.sol_sd * torch.randn_like(uv_ic)
        
        return txy_ic, uv_ic_noisy, uv_ic
    
    def generate_domain(self, n):
        t = torch.rand(n, 1)
        x = 2 * torch.rand(n, 1) - 1
        y = 2 * torch.rand(n, 1) - 1
        return torch.cat([t, x, y], dim=1)

    
    def get_diff_data(self):
        X = self.generate_domain(self.n_diff)
        
        true_Y = torch.zeros(X.shape[0], 3)
        Y = true_Y + self.diff_sd * torch.randn_like(true_Y)
        return X, Y, true_Y

    def differential_operator(self, model: torch.nn.Module, physics_X, pe_variables=None):
            
        uvp = model(physics_X)
        # print(uvp.shape)
        u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
        # print(u.shape, v.shape, p.shape)

        du = grad(u, physics_X)[0]
        dudt = du[:, 0:1].view(-1, 1)
        dudx = du[:, 1:2].view(-1, 1)
        dudy = du[:, 2:3].view(-1, 1)
        
        d2udx2 = grad(dudx, physics_X)[0][:, 1:2].view(-1, 1)
        d2udy2 = grad(dudy, physics_X)[0][:, 2:3].view(-1, 1)

        dv = grad(v, physics_X)[0]
        dvdt = dv[:, 0:1].view(-1, 1)
        dvdx = dv[:, 1:2].view(-1, 1)
        dvdy = dv[:, 2:3].view(-1, 1)
        d2vdx2 = grad(dvdx, physics_X)[0][:, 1:2].view(-1, 1)
        d2vdy2 = grad(dvdy, physics_X)[0][:, 2:3].view(-1, 1)
        
        dp = grad(p, physics_X)[0]
        dpdx = dp[:, 1:2].view(-1, 1)
        dpdy = dp[:, 2:3].view(-1, 1)
        
        
        cont = dudx + dvdy
        mom_u = dudt + u * dudx + v * dudy + dpdx - self.nu * (d2udx2 + d2udy2)
        mom_v = dvdt + u * dvdx + v * dvdy + dpdy - self.nu * (d2vdx2 + d2vdy2)
        
        pde = torch.cat([mom_u, mom_v, cont], dim=1)
        
        return pde
    
    
    def taylor_green_solution(self, X):
        t = X[:, 0:1]
        x = X[:, 1:2]
        y = X[:, 2:3]
        u = -torch.cos(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-2 * np.pi**2 * self.nu * t)
        v = torch.sin(np.pi * x) * torch.cos(np.pi * y) * torch.exp(-2 * np.pi**2 * self.nu * t)
        p = -0.25 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y)) * torch.exp(-4 * np.pi**2 * self.nu * t)
        
        Y = torch.cat([u, v, p], dim=1)
        return Y

    def plot_true_solution(self, save_path=None):
        eval_X, eval_Y = self.get_eval_data()
        # Parse tensors
        t, x, y = eval_X[:, 0:1], eval_X[:, 1:2], eval_X[:, 2:3]
        u, v, p = eval_Y[:, 0:1], eval_Y[:, 1:2], eval_Y[:, 2:3]

        # Define time points and get indices
        times = [0, 0.5, 1.0]
        time_indices = [(eval_X[:, 0] == t_val).nonzero(as_tuple=True)[0] for t_val in times]

        # Collect slices
        u_slices = [u[idx] for idx in time_indices]
        v_slices = [v[idx] for idx in time_indices]
        p_slices = [p[idx] for idx in time_indices]

        # Determine global vmin/vmax per row (per field)
        u_min, u_max = torch.min(torch.cat(u_slices)).item(), torch.max(torch.cat(u_slices)).item()
        v_min, v_max = torch.min(torch.cat(v_slices)).item(), torch.max(torch.cat(v_slices)).item()
        p_min, p_max = torch.min(torch.cat(p_slices)).item(), torch.max(torch.cat(p_slices)).item()

        # Create 3x3 plot
        fig, axes = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)
        field_data = [
            ('u', u, u_min, u_max, 'PuOr'),
            ('v', v, v_min, v_max, 'BrBG'),
            ('p', p, p_min, p_max, 'coolwarm')
        ]

        for row, (field_name, field_tensor, vmin, vmax, cmap) in enumerate(field_data):
            for col, (t_val, idx) in enumerate(zip(times, time_indices)):
                x_vals = eval_X[idx, 1].reshape(101, 101)
                y_vals = eval_X[idx, 2].reshape(101, 101)
                f_vals = field_tensor[idx].reshape(101, 101)

                contour = axes[row, col].contourf(
                    x_vals, y_vals, f_vals,
                    levels=20, cmap=cmap,
                    vmin=vmin, vmax=vmax
                )
                axes[row, col].set_title(f'{field_name} at t={t_val:.2f}')
                axes[row, col].set_xlabel('x')
                axes[row, col].set_ylabel('y')

            # Shared colorbar per row
            cbar = fig.colorbar(contour, ax=axes[row, :], orientation='vertical', fraction=0.02, pad=0.05)
            cbar.set_label(f'{field_name}')

        # Save or show
        if save_path:
            plt.savefig(os.path.join(save_path, 'u_v_p_contour_grid_colormap.png'))
            plt.close()
        else:
            plt.show()


class TaylorGreenCallback(BaseCallback):
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
        
        if self.model.progress >= self.eval_buffer.burn and hasattr(self.model, 'sampler'):
            if hasattr(self, 'max_lr'):
                self.max_lr = max(self.max_lr, self.model.cur_sgld_lr)
                accept_rate = self.model.cur_sgld_lr / self.max_lr
                if random.random() > accept_rate:
                    # print(f"Rejecting rate: {1-accept_rate}")
                    return
            else:
                self.max_lr = self.model.cur_sgld_lr

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
        
        if self.model.progress <= self.eval_buffer.burn:
            self.eval_buffer.reset()
            if self.physics_model.is_inverse:
                self.k_buffer.reset()
        # self.physics_model.save_evaluation(self.model, self.save_path)
        # self.physics_model.save_temp_frames(self.model, self.n_evals, self.save_path)
    
    def _on_training_end(self) -> None:
        self.save_gif()
        pass
        
    def plot_latent_Z(self):
        true_y = self.dataset[0]['true_y'].flatten()
        sol_y = self.dataset[0]['y'].flatten()
        sd = self.dataset[0]['noise_sd']
        true_Z = (sol_y - true_y) / sd
        
        latent_Z = self.model.latent_Z[0].flatten().detach().cpu().numpy()
        min_val = min(true_Z.min(), latent_Z.min())
        max_val = max(true_Z.max(), latent_Z.max())
        np.save(os.path.join(self.save_path, 'true_Z.npy'), true_Z)
        np.save(os.path.join(self.save_path, 'latent_Z.npy'), latent_Z)
        
        plt.subplots(figsize=(6, 6))
        plt.scatter(true_Z, latent_Z, label='Latent Z')
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='x = y')
        plt.xlabel('True Z')
        plt.ylabel('Latent Z')
        # plt.xlim(-3*sd, 3*sd)
        # plt.ylim(-3*sd, 3*sd)
        plt.savefig(os.path.join(self.save_path, 'latent_Z.png'))
        plt.close()
        
        true_y = self.dataset[1]['true_y'].flatten()
        sol_y = self.dataset[1]['y'].flatten()
        sd = self.dataset[1]['noise_sd']
        true_Z = sol_y - true_y
        
        latent_Z = self.model.latent_Z[1].flatten().detach().cpu().numpy()
        min_val = min(true_Z.min(), latent_Z.min())
        max_val = max(true_Z.max(), latent_Z.max())
        
        np.save(os.path.join(self.save_path, 'true_Z_diff.npy'), true_Z)
        np.save(os.path.join(self.save_path, 'latent_Z_diff.npy'), latent_Z)
        
        plt.subplots(figsize=(6, 6))
        plt.scatter(true_Z, latent_Z, label='Latent Z')
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='x = y')
        plt.xlabel('True Z')
        plt.ylabel('Latent Z')
        # plt.xlim(-3*sd, 3*sd)
        # plt.ylim(-3*sd, 3*sd)
        plt.savefig(os.path.join(self.save_path, 'latent_Z_diff.png'))
        plt.close()
        
    def save_evaluation(self):

        
        X = self.eval_X_cpu.numpy()
        # y = self.eval_y_cpu.numpy()
        grids = 25
        
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
    model = TaylorGreen(
        t_start=0.0,
        t_end=1.0,
        sol_sd=0.1,
        diff_sd=0.0,
        n_sol=1000,
        n_diff=10000,
        nu = 0.01,
        is_inverse=False
    )


    dataset = model.generate_data(device='cpu')
    
    for d in dataset:
        print(d['X'].shape, d['y'].shape, d['true_y'].shape, d['category'])
        
        
    model.plot_true_solution()
    