import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel


    
    
        
class CoolingModel(PhysicsModel):
    def __init__(self, Tenv=25, T0=100, R=0.005, T_end=300, T_extend=1000) -> None:
        super().__init__(Tenv=Tenv, T0=T0, R=R, T_end=T_end, T_extend=T_extend)

        
    def _data_generation(self, n_samples=10, noise_sd=1.0):
        t = torch.linspace(0, self.T_end, n_samples).reshape(n_samples, -1)
        T = self.physics_law(t) +  noise_sd * torch.randn(n_samples).reshape(n_samples, -1)
        
        return t, T
    
    def physics_law(self, time):
        T = self.Tenv + (self.T0 - self.Tenv) * torch.exp(-self.R * time)
        return T
    
    def physics_loss(self, model: torch.nn.Module):
        ts = torch.linspace(0, self.T_extend, steps=self.T_extend,).view(-1,1).requires_grad_(True)
        temps = model(ts)
        dT = grad(temps, ts)[0]
        pde = self.R*(self.Tenv - temps) - dT
        
        return torch.mean(pde**2)
    


if __name__ == "__main__":
    
    physics = CoolingModel()
    
    t, T = physics.X, physics.y
    
    times = torch.linspace(0, 1000, 1000)
    temps = physics.physics_law(times)
    
    
    
    plt.plot(times, temps, label='Equation')
    plt.plot(t, T, 'x', label='Training data')
    plt.legend()
    plt.ylabel('Temperature (C)')
    plt.xlabel('Time (s)')
    plt.show()
