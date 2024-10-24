import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel


    
    
        
class Cooling(PhysicsModel):
    def __init__(self, 
                 Tenv=25, 
                 T0=100, 
                 R=0.005, 
                 t_end=300, 
                 t_extend=1500,
                 noise_sd=1.0
                 ):
        super().__init__(Tenv=Tenv, T0=T0, R=R, t_end=t_end, t_extend=t_extend, noise_sd=noise_sd)

        
    def _data_generation(self, n_samples=200):
        t = torch.linspace(0, self.t_end, n_samples).reshape(n_samples, -1)
        T = self.physics_law(t) +  self.noise_sd * torch.randn(n_samples).reshape(n_samples, -1)
        
        return t, T
    
    def _eval_data_generation(self):
        t = torch.linspace(0, self.t_extend, self.t_extend).reshape(self.t_extend, -1)
        return t
    
    def physics_law(self, time):
        T = self.Tenv + (self.T0 - self.Tenv) * torch.exp(-self.R * time)
        return T
    
    def physics_loss(self, model: torch.nn.Module):
        ts = torch.linspace(0, self.t_extend, steps=self.t_extend,).view(-1,1).requires_grad_(True)
        temps = model(ts)
        dT = grad(temps, ts)[0]
        pde = self.R*(self.Tenv - temps) - dT
        
        return torch.mean(pde**2)
    


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
