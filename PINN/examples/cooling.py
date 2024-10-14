import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from PINN.common.grad_tool import grad



class PhysicsModel(object):
    def __init__(self, input_dim, output_dim) -> None:
        
        self.input_dim = input_dim
        self.output_dim = output_dim

    
    def physics_law(self):
        ''' Implement the physics law here '''
        raise NotImplementedError()
    
    def physics_loss(self):
        ''' Implement the physics loss here '''
        raise NotImplementedError()
    
    def plot(self):
        ''' Implement the plot here '''
        raise NotImplementedError()
    
    
        
class CoolingModel(PhysicsModel):
    def __init__(self, input_dim, output_dim, Tenv, T0, R) -> None:
        super().__init__(input_dim, output_dim)
        self.Tenv = Tenv
        self.T0 = T0
        self.R = R
    
    def physics_law(self, time):
        T = self.Tenv + (self.T0 - self.Tenv) * np.exp(-self.R * time)
        return T
    
    def physics_loss(self, model: torch.nn.Module):
        ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True)
        temps = model(ts)
        dT = grad(temps, ts)[0]
        pde = self.R*(self.Tenv - temps) - dT
        
        return torch.mean(pde**2)
    

def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * np.exp(-R * time)
    return T


def physics_loss(model: torch.nn.Module):
    ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True)
    temps = model(ts)
    dT = grad(temps, ts)[0]
    pde = R*(Tenv - temps) - dT
    
    return torch.mean(pde**2)


Tenv = 25
T0 = 100
R = 0.005
times = np.linspace(0, 1000, 1000)
eq = functools.partial(cooling_law, Tenv=Tenv, T0=T0, R=R)
temps = eq(times)

# Make training data



if __name__ == "__main__":
    
    physics = CoolingModel(1, 1, Tenv, T0, R)
    
    t = np.linspace(0, 300, 10)
    T = eq(t) +  np.random.randn(10)
    
    
    plt.plot(times, temps)
    plt.plot(t, T, 'o')
    plt.legend(['Equation', 'Training data'])
    plt.ylabel('Temperature (C)')
    plt.xlabel('Time (s)')
    plt.show()
