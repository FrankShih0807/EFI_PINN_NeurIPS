import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel
from PINN.common.utils import PINNDataset


        
class Cooling(PhysicsModel):
    def __init__(self, 
                 Tenv=25, 
                 T0=100, 
                 R=0.005, 
                 t_end=300, 
                 t_extend=1500,
                 noise_sd=1.0,
                 n_samples=200
                 ):
        super().__init__(Tenv=Tenv, T0=T0, R=R, t_end=t_end, t_extend=t_extend, noise_sd=noise_sd, n_samples=n_samples)

        
    # def _data_generation(self, n_samples=200):
    #     t = torch.linspace(0, self.t_end, n_samples).reshape(n_samples, -1)
    #     T = self.physics_law(t) +  self.noise_sd * torch.randn(n_samples).reshape(n_samples, -1)
        
    #     self.physics_X = torch.linspace(0, self.t_extend, steps=self.t_extend,).view(-1,1).requires_grad_(True)
    #     return t, T
    
    def generate_data(self, n_samples, device):
        dataset = PINNDataset(device=device)
        X, y = self.get_temp_data(n_samples)
        diff_X, diff_y = self.get_diff_data()
        eval_X, eval_y = self.get_eval_data()
        dataset.add_data(X, y, category='solution', noise_sd=self.noise_sd)
        dataset.add_data(diff_X, diff_y, category='differential', noise_sd=0)
        dataset.add_data(eval_X, eval_y, category='evaluation', noise_sd=0)
        
        return dataset
    
    def get_eval_data(self):
        t = torch.linspace(0, self.t_extend, self.t_extend).reshape(self.t_extend, -1)
        T = self.physics_law(t)
        return t, T
    
    def get_temp_data(self, n_samples):
        X = torch.linspace(0, self.t_end, n_samples).reshape(n_samples, -1)
        y = self.physics_law(X)
        y += self.noise_sd * torch.randn_like(y)
        return X, y
    
    def get_diff_data(self):
        X = torch.linspace(0, self.t_extend, steps=self.t_extend,).view(-1,1)
        y = torch.zeros(self.t_extend, 1)
        return X, y
    
    # def _eval_data_generation(self):
    #     t = torch.linspace(0, self.t_extend, self.t_extend).reshape(self.t_extend, -1)
    #     return t
    
    def physics_law(self, time):
        T = self.Tenv + (self.T0 - self.Tenv) * torch.exp(-self.R * time)
        return T
    
    # def physics_loss(self, model: torch.nn.Module, physics_X):
    #     # ts = torch.linspace(0, self.t_extend, steps=self.t_extend,).view(-1,1).requires_grad_(True)
    #     temps = model(physics_X)
    #     dT = grad(temps, physics_X)[0]
    #     pde = self.R*(self.Tenv - temps) - dT
        
    #     return torch.mean(pde**2)
    
    def differential_operator(self, model: torch.nn.Module, physics_X):
        temps = model(physics_X)
        dT = grad(temps, physics_X)[0]
        pde = self.R*(self.Tenv - temps) - dT
        return pde
    
    def plot_true_solution(self, save_path=None):
        times = torch.linspace(0, self.t_extend, self.t_extend)
        temps = self.physics_law(times)
        
        sns.set_theme()
        plt.plot(times, temps, label='Equation')
        # plt.plot(self.X, self.y, 'x', label='Training data')
        plt.legend()
        plt.ylabel('Temperature (C)')
        plt.xlabel('Time (s)')
        plt.savefig(os.path.join(save_path, 'true_solution.png'))
        plt.close()
        
    def save_evaluation(self, model, save_path=None):
        preds_upper, preds_lower, preds_mean = model.summary()
        preds_upper = preds_upper.flatten()
        preds_lower = preds_lower.flatten()
        preds_mean = preds_mean.flatten()     

        times = torch.linspace(0, self.t_extend, self.t_extend)
        temps = self.physics_law(times)
        
        
        np.savez(os.path.join(save_path, 'evaluation_data.npz') , preds_upper=preds_upper, preds_lower=preds_lower, preds_mean=preds_mean)
        
        sns.set_theme()
        plt.plot(times, temps, alpha=0.8, color='b', label='Equation')
        plt.plot(times, preds_mean, alpha=0.8, color='g', label='PINN')
        # plt.plot(times, preds, alpha=0.8, color='g', label='PINN')
        # plt.plot(self.X, self.y, 'x', label='Training data')
        plt.vlines(self.t_end, self.Tenv, self.T0, color='r', linestyles='dashed', label='no data beyond this point')
        plt.fill_between(times, preds_upper, preds_lower, alpha=0.2, color='g', label='95% CI')
        plt.legend()
        plt.ylabel('Temperature (C)')
        plt.xlabel('Time (s)')
        plt.savefig(os.path.join(save_path, 'pred_solution.png'))
        plt.close()
    


if __name__ == "__main__":
    
    physics = Cooling()
    print(physics.model_params)
    
    t, T = physics.X, physics.y
    
    times = torch.linspace(0, 1000, 1000)
    temps = physics.physics_law(times)
    
    # for key, value in physics.__dict__.items():
    #     print('{}: {}'.format(key, value))
    
    
    plt.plot(times, temps, label='Equation')
    plt.plot(t, T, 'x', label='Training data')
    plt.legend()
    plt.ylabel('Temperature (C)')
    plt.xlabel('Time (s)')
    plt.show()
