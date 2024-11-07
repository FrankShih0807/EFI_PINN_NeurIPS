import os
import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel

class Nonlinear(PhysicsModel):
# class FuncApprox(PhysicsModel):
    def __init__(self, t_start=0, t_end=20, noise_sd=1.0):
        super().__init__(
            t_start=t_start, t_end=t_end, noise_sd=noise_sd
        )

    def _data_generation(self, n_samples=200):
        t = torch.linspace(self.t_start, self.t_end, n_samples).reshape(n_samples, -1)

        # Y1, Y2 = self.physics_law(t)
        # y = torch.cat([Y1, Y2], dim=1)
        # y += self.noise_sd * torch.randn_like(y)
        # return t, y
    
        y = self.physics_law(t)
        y += self.noise_sd * torch.randn_like(y)
        return t, y

    def _eval_data_generation(self):
        t = torch.linspace(
            self.t_start, self.t_end, round((self.t_end - self.t_start) * 10)
        ).reshape(round((self.t_end - self.t_start) * 10), -1)
        return t

    def physics_law(self, time):
        # Y1 = 3 * torch.cos(time)
        # Y2 = 3 * torch.sin(time)
        # # Y1 = time**2
        # # Y2 = (-0.01*time**7-time**4-2*time**2-4*time+1)
        # return Y1, Y2

        Y = 3 * torch.sin(time)
        # Y = time
        # Y = 3 * torch.sin(0.6 * time) ** 3
        return Y

    def physics_loss(self, model: torch.nn.Module):
        return 0
    
    def save_evaluation(self, model, save_path=None):
        preds_upper, preds_lower, preds_mean = model.summary()

        # Y1_upper = preds_upper[:,0]
        # Y1_lower = preds_lower[:,0]
        # Y1_mean = preds_mean[:,0]
    
        # Y2_upper = preds_upper[:,1]
        # Y2_lower = preds_lower[:,1]
        # Y2_mean = preds_mean[:,1]

        # times = torch.linspace(self.t_start, self.t_extend, (self.t_extend - self.t_start) * 10)
        # Y1_true, Y2_true = self.physics_law(times)

        # np.savez(os.path.join(save_path, 'evaluation_data.npz'), 
        #          Y1_upper=Y1_upper, Y1_lower=Y1_lower, Y1_mean=Y1_mean,
        #          Y2_upper=Y2_upper, Y2_lower=Y2_lower, Y2_mean=Y2_mean)
        
        # sns.set_theme()

        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # ax1.plot(times, Y1_true.flatten().numpy(), 'b', label='True Y1')
        # ax1.plot(times, Y1_mean, 'g', label='Estimated Y1')
        # ax1.fill_between(times, Y1_upper, Y1_lower, color='g', alpha=0.2)
        # ax1.scatter(model.X, model.y[:,0], color='r', label='Data', marker='x')
        # ax1.set_ylabel('Y1(t)')
        # ax1.legend()

        # ax2.plot(times, Y2_true.flatten().numpy(), 'b', label='True Y2')
        # ax2.plot(times, Y2_mean, 'g', label='Estimated Y2')
        # ax2.fill_between(times, Y2_upper, Y2_lower, color='g', alpha=0.2)
        # ax2.scatter(model.X, model.y[:,1], color='r', label='Data', marker='x')
        # ax2.set_xlabel('t')
        # ax2.set_ylabel('Y2(t)')
        # ax2.legend()

        # plt.savefig(os.path.join(save_path, 'pred_solution.png'))
        # plt.close()

        preds_upper = preds_upper.flatten()
        preds_lower = preds_lower.flatten()
        preds_mean = preds_mean.flatten()

        times = torch.linspace(self.t_start, self.t_end, round((self.t_end - self.t_start) * 10))
        Y_true = self.physics_law(times)

        np.savez(os.path.join(save_path, "evaluation_data.npz"),
            preds_upper=preds_upper, preds_lower=preds_lower, preds_mean=preds_mean)
        
        sns.set_theme()

        plt.plot(times, Y_true, alpha=0.8, color='b', label='Equation')
        plt.plot(times, preds_mean, alpha=0.8, color='g', label='PINN')
        plt.fill_between(times, preds_upper, preds_lower, color='g', alpha=0.2)
        plt.scatter(model.X, model.y, color='r', label='Data', marker='x')
        plt.legend()
        plt.ylabel('Y(t)')

        plt.savefig(os.path.join(save_path, 'pred_solution.png'))
        plt.close()
