import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
import seaborn as sns

from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel
from scipy.integrate import solve_ivp


    
    
        
class FitzHugh_Nagumo(PhysicsModel):
    def __init__(self, 
                 a = 0.2,
                 b = 0.2,
                 c = 3.0,
                 V0 = -1.0,
                 R0 = 1.0,
                 t_end=5,
                 t_extend=20
                 ):
        super().__init__(a=a, b=b, c=c, V0=V0, R0=R0, t_end=t_end, t_extend=t_extend)

        
    def _data_generation(self, n_samples=200, noise_sd=0.05):
        t_span = (0, 5)
        t_eval = np.linspace(*t_span, n_samples)
        sol = solve_ivp(self.fitzhugh_nagumo, t_span, [self.V0, self.R0], t_eval=t_eval)
        
        X = torch.tensor(sol.t).reshape(-1,1)
        y = torch.stack([torch.tensor(sol.y[0]), torch.tensor(sol.y[1])], dim=1)
        y += noise_sd * torch.randn_like(y)
        return X, y
    
    def physics_law(self,x):
        pass
    
    def fitzhugh_nagumo(self, t, y):
        V, R = y
        dVdt = self.c * (V - V**3 / 3 + R)
        dRdt =  -(V - self.a + self.b * R) / self.c
        return [dVdt, dRdt]
    
    def physics_loss(self, model: torch.nn.Module):
        ts = torch.linspace(0, self.t_extend, steps=self.t_extend,).view(-1,1).requires_grad_(True)
        ys = model(ts)
        print(ys)
        d = jacobian(model.forward, ts)
        print(d.shape)
        raise
        dT = grad(ys, ts)[0]
        pde = self.R*(self.Tenv - temps) - dT
        
        return torch.mean(pde**2)
    
    
    def plot(self):
        pass
    


if __name__ == "__main__":
    
    # physics = Poisson1D()
    # print(physics.model_params)
    
    # t, T = physics.X, physics.y
    
    # times = torch.linspace(0, 1000, 1000)
    # temps = physics.physics_law(times)
    
    # # for key, value in physics.__dict__.items():
    # #     print('{}: {}'.format(key, value))
    
    
    # plt.plot(times, temps, label='Equation')
    # plt.plot(t, T, 'x', label='Training data')
    # plt.legend()
    # plt.ylabel('Temperature (C)')
    # plt.xlabel('Time (s)')
    # plt.show()
    
    
    model = FitzHugh_Nagumo()
    
    net = nn.Sequential(
        nn.Linear(1, 15),
        nn.Softplus(),
        nn.Linear(15, 15),
        nn.Softplus(),
        nn.Linear(15, 2)
    )
    model.physics_loss(net)
        
    
    # plt.plot(model.X, model.y, 'x')
    # plt.show()
    
    # def fitzhugh_nagumo(t, y, a, b, c):
    #     V, R = y
    #     dVdt = c * (V - V**3 / 3 + R)
    #     dRdt =  -(V - a + b * R) / c
    #     return [dVdt, dRdt]

    # # Parameters
    # a = 0.2
    # b = 0.2
    # c = 3.0
    
    # # Initial conditions
    # y0 = [-1.0, 1.0]

    # # Time span
    # t_span = (0, 20)
    # t_eval = np.linspace(*t_span, 1000)

    # # Solve the system
    # sol = solve_ivp(fitzhugh_nagumo, t_span, y0, args=(a, b, c), t_eval=t_eval)

    
    # sns.set_theme()
    # # Plot the results
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    
    # ax1.plot(sol.t, sol.y[0], 'b', label='V (Voltage Variable)')
    # ax1.set_ylabel('v')
    # ax1.set_ylim(-2, 4)
    # ax1.legend()
    
    # ax2.plot(sol.t, sol.y[1], 'r', label='R (Recovery Variable)')
    # ax2.set_xlabel('t')
    # ax2.set_ylabel('w')
    # ax2.set_ylim(-1, 2)
    # ax2.legend()
    
    # plt.tight_layout()
    # plt.show()
    
