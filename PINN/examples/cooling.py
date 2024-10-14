import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from PINN.common.grad_tool import grad



class PhysicsModel(object):
    def __init__(self, **kwargs) -> None:

        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.X, self.y = self._data_generation()
        self.input_dim = self.X.shape[1]
        self.output_dim = self.y.shape[1]
        self.n_samples = self.X.shape[0]

    
    def physics_law(self):
        ''' Implement the physics law here '''
        raise NotImplementedError()
    
    def physics_loss(self):
        ''' Implement the physics loss here '''
        raise NotImplementedError()
    
    
    
        
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
    


Tenv = 25
T0 = 100
R = 0.005


if __name__ == "__main__":
    
    # physics = CoolingModel(1, 1, Tenv, T0, R)
    physics = CoolingModel()
    
    print(physics.input_dim)
    print(physics.output_dim)
    
    print(physics.X.shape)
    print(physics.y.shape)
    
    # t = torch.linspace(0, 300, 10).reshape(10, -1)
    # T = physics.physics_law(t) +  torch.randn(10).reshape(10, -1)
    
    t, T = physics.X, physics.y
    
    times = torch.linspace(0, 1000, 1000)
    temps = physics.physics_law(times)
    
    
    
    plt.plot(times, temps)
    plt.plot(t, T, 'o')
    plt.legend(['Equation', 'Training data'])
    plt.ylabel('Temperature (C)')
    plt.xlabel('Time (s)')
    plt.show()
