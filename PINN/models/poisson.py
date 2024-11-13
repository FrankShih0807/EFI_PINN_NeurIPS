import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel
from PINN.common.utils import PINNDataset

class Poisson(PhysicsModel):
    def __init__(self, 
                 sigma=0.01, 
                 lam1=100.0, 
                 lam2=100.0, 
                 t_start=-0.7,
                 t_end=0.7, 
                 noise_sd=0.01, 
                #  n_samples=16
                 ):
        super().__init__(sigma=sigma, lam1=lam1, lam2=lam2, t_start=t_start, t_end=t_end, noise_sd=noise_sd)

    def generate_data(self, n_samples, device): ###################
        dataset = PINNDataset(device=device)
        X, y = self.get_solu_data()
        diff_X, diff_y = self.get_diff_data(n_samples)
        eval_X, eval_y = self.get_eval_data()
        dataset.add_data(X, y, category='solution', noise_sd=self.noise_sd)
        dataset.add_data(diff_X, diff_y, category='differential', noise_sd=self.noise_sd)
        dataset.add_data(eval_X, eval_y, category='evaluation', noise_sd=0)
        
        return dataset
    
    def get_eval_data(self):
        X = torch.linspace(self.t_start, self.t_end, steps=100).reshape(100, -1)
        y = self.physics_law(X)
        return X, y
    
    def get_solu_data(self):
        X = torch.tensor([self.t_start, self.t_end]).view(-1, 1)
        y = self.physics_law(X)
        y += self.noise_sd * torch.randn_like(y)
        return X, y
    
    def get_diff_data(self, n_samples):
        X = torch.linspace(self.t_start, self.t_end, steps=n_samples).view(-1,1)
        y = self.lam1 * (-1.08) * torch.sin(6 * X) * (torch.sin(6 * X) ** 2 - 2 * torch.cos(6 * X) ** 2)
        y += self.noise_sd * torch.randn_like(y)
        return X, y

    # def _data_generation(self, n_samples=16):
    #     X = np.linspace(self.t_start, self.t_end, n_samples)[:, None]
    #     lb, rb = np.array([[self.t_start]]), np.array([[self.t_end]])
    #     y = self.physics_law(X)
    #     # y = self.lam1 * (-1.08) * np.sin(6 * X) * (np.sin(6 * X) ** 2 - 2 * np.cos(6 * X) ** 2)
    #     X = np.concatenate([X, lb, rb], axis=0)
    #     y = np.concatenate([y, np.sin(6 * lb) ** 3 * self.lam2, np.sin(6 * rb) ** 3 * self.lam2], axis=0)
    #     y = y * (1 + self.sigma * np.random.randn(*y.shape))
        
    #     X = torch.FloatTensor(X)
    #     y = torch.FloatTensor(y)
        
    #     self.physics_X = torch.linspace(self.t_start, self.t_end, steps=100).view(-1, 1).requires_grad_(True)
    #     return X, y
    
    # def _eval_data_generation(self):
    #     X = torch.linspace(self.t_start, self.t_end, steps=100).reshape(100, -1)
    #     lb, rb = torch.tensor([[self.t_start]]), torch.tensor([[self.t_end]])
    #     X = torch.cat([X, lb, rb], dim=0)
    #     return X
    
    def physics_law(self, X):
        y = self.lam2 * torch.sin(6 * X) ** 3
        # y = self.lam1 * (-1.08) * np.sin(6 * X) * (np.sin(6 * X) ** 2 - 2 * np.cos(6 * X) ** 2)
        return y
    
    def differential_operator(self, model: torch.nn.Module, physics_X):
        u = model(physics_X)
        # u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad(u, physics_X)[0]
        u_xx = grad(u_x, physics_X)[0]
        pde = self.lam1 * 0.01 * u_xx - self.physics_law(physics_X)
        
        return pde
    
    # def physics_loss(self, model: torch.nn.Module, physics_X):
    #     x = physics_X.requires_grad_(True)
    #     u = model(x)
    #     u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    #     u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    #     pde = self.lam1 * 0.01 * u_xx - self.physics_law(x)
        
    #     return torch.mean(pde**2)
    
    def plot_true_solution(self, save_path=None):
        X = torch.linspace(self.t_start, self.t_end, steps=100)
        y = self.physics_law(X)
        
        sns.set_theme()
        plt.plot(X, y, label='Equation')
        plt.legend()
        plt.ylabel('u')
        plt.xlabel('x')
        if save_path:
            plt.savefig(os.path.join(save_path, 'true_solution.png'))
        else:
            plt.show()
        plt.close()
        
    def save_evaluation(self, model, save_path=None):
        preds_upper, preds_lower, preds_mean = model.summary()
        preds_upper = preds_upper.flatten()
        preds_lower = preds_lower.flatten()
        preds_mean = preds_mean.flatten()     

        X = torch.linspace(self.t_start, self.t_end, steps=100)
        y = self.physics_law(X)
        
        if save_path is None:
            save_path = './evaluation_results'
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        np.savez(os.path.join(save_path, 'evaluation_data.npz'), preds_upper=preds_upper, preds_lower=preds_lower, preds_mean=preds_mean)
        
        sns.set_theme()
        plt.plot(X, y, alpha=0.8, color='b', label='True')
        plt.plot(X, preds_mean, alpha=0.8, color='g', label='Mean')
        plt.fill_between(X, preds_upper, preds_lower, alpha=0.2, color='g', label='95% CI')
        plt.legend()
        plt.ylabel('u')
        plt.xlabel('x')
        plt.savefig(os.path.join(save_path, 'pred_solution.png'))
        plt.close()


# if __name__ == "__main__":
#     physics = Poisson()
#     print(physics.model_params)
    
#     X, y = physics.X, physics.y
    
#     X_plot = torch.linspace(-0.7, 0.7, 100)
#     y_plot = physics.physics_law(X_plot)
    
#     plt.plot(X_plot, y_plot, label='Equation')
#     plt.plot(X, y, 'x', label='Training data')
#     plt.legend()
#     plt.ylabel('Value')
#     plt.xlabel('X')
#     plt.show()