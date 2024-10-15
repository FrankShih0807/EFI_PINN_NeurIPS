import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel


    
    
        
class FitzHugh_Nagumo(PhysicsModel):
    def __init__(self, 
                 a = 0.2,
                 b = 0.2,
                 c = 3.0,
                 V0 = -1.0,
                 R0 = 1.0
                 ):
        super().__init__(a=a, b=b, c=c, V0=V0, R0=R0)

        
    def _data_generation(self, n_samples=200, noise_sd=1.0):
        t = torch.linspace(0, self.t_end, n_samples).reshape(n_samples, -1)
        T = self.physics_law(t) +  noise_sd * torch.randn(n_samples).reshape(n_samples, -1)
        
        return t, T
    
    def physics_law(self,x):
        u = torch.sin(6*x) ** 3
        return u
    
    def physics_loss(self, model: torch.nn.Module):
        ts = torch.linspace(0, self.t_extend, steps=self.t_extend,).view(-1,1).requires_grad_(True)
        temps = model(ts)
        dT = grad(temps, ts)[0]
        pde = self.R*(self.Tenv - temps) - dT
        
        return torch.mean(pde**2)
    


if __name__ == "__main__":
    
    physics = Poisson1D()
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
