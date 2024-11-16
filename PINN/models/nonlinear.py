import os
import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel
from PINN.common.utils import PINNDataset

class Nonlinear(PhysicsModel):
# class FuncApprox(PhysicsModel):
    def __init__(self, t_start=0, t_end=20, noise_sd=1.0):
        super().__init__(
            t_start=t_start, t_end=t_end, noise_sd=noise_sd
        )

    def generate_data(self, n_samples, device):
        dataset = PINNDataset(device=device)
        X, y = self.get_solu_data(n_samples)
        # diff_X, diff_y = self.get_diff_data()
        eval_X, eval_y = self.get_eval_data()
        dataset.add_data(X, y, category='solution', noise_sd=self.noise_sd)
        # dataset.add_data(diff_X, diff_y, category='differential', noise_sd=0)
        dataset.add_data(eval_X, eval_y, category='evaluation', noise_sd=0)

        return dataset
    
    def get_eval_data(self):
        t = torch.linspace(
            self.t_start, self.t_end, round((self.t_end - self.t_start) * 10)
        ).reshape(round((self.t_end - self.t_start) * 10), -1)
        y = self.physics_law(t)
        return t, y
    
    def get_solu_data(self, n_samples):
        t = torch.linspace(self.t_start, self.t_end, n_samples).reshape(n_samples, -1)
        y = self.physics_law(t)
        y += self.noise_sd * torch.randn_like(y)
        return t, y
    
    def get_diff_data(self):
        t = torch.linspace(
            self.t_start, self.t_end, round((self.t_end - self.t_start) * 10)
        ).reshape(round((self.t_end - self.t_start) * 10), -1)
        y = torch.zeros_like(t)
        return t, y

    def physics_law(self, time):
        Y = 3 * torch.sin(time)
        return Y
    
    def differential_operator(self, model: torch.nn.Module, physics_X):
        return 0

    
    def plot_true_solution(self, save_path=None):
        t = torch.linspace(self.t_start, self.t_end, round((self.t_end - self.t_start) * 10))
        Y = self.physics_law(t)

        sns.set_theme()
        plt.plot(t, Y, alpha=0.8, color='b', label='Equation')
        plt.legend()
        plt.ylabel('Y(t)')
        plt.xlabel('t')

        plt.savefig(os.path.join(save_path, 'true_solution.png'))
        plt.close()

    def save_evaluation(self, model, save_path=None):
        # preds_upper, preds_lower, preds_mean = model.summary()
        pred_dict = model.summary()


        preds_upper = pred_dict['y_preds_upper'].flatten()
        preds_lower = pred_dict['y_preds_lower'].flatten()
        preds_mean = pred_dict['y_preds_mean'].flatten()

        times = torch.linspace(self.t_start, self.t_end, round((self.t_end - self.t_start) * 10))
        Y_true = self.physics_law(times)

        # np.savez(os.path.join(save_path, "evaluation_data.npz"), preds_upper=preds_upper, preds_lower=preds_lower, preds_mean=preds_mean)
        np.savez(os.path.join(save_path, 'evaluation_data.npz') , **pred_dict)
        
        sns.set_theme()

        plt.plot(times, Y_true, alpha=0.8, color='b', label='Equation')
        plt.plot(times, preds_mean, alpha=0.8, color='g', label='PINN')
        plt.fill_between(times, preds_upper, preds_lower, color='g', alpha=0.2)
        plt.scatter(model.X.detach().cpu(), model.y.detach().cpu(), color='r', label='Data', marker='x')
        plt.legend()
        plt.ylabel('Y(t)')
        plt.xlabel('t')

        plt.savefig(os.path.join(save_path, 'pred_solution.png'))
        plt.close()
