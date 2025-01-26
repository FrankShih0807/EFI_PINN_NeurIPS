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
                 t_extend=10,
                 noise_sd=0.1
                 ):
        super().__init__(a=a, b=b, c=c, V0=V0, R0=R0, t_end=t_end, t_extend=t_extend, noise_sd=noise_sd)

        
    def _data_generation(self, n_samples=300):
        t_eval = np.linspace(0, self.t_end, n_samples)
        V, R = self.physics_law(t_eval, self.V0, self.R0)
        
        X = torch.tensor(t_eval).to(torch.float32).reshape(-1,1)
        y = torch.stack([V, R], dim=1)
        y += self.noise_sd * torch.randn_like(y)
        return X, y
    
    def physics_law(self, ts, V_init, R_init)->torch.Tensor:
        sol = solve_ivp(self.fitzhugh_nagumo, (ts[0], ts[-1]), [V_init, R_init], t_eval=ts)
        V = torch.tensor(sol.y[0]).to(torch.float32)
        R = torch.tensor(sol.y[1]).to(torch.float32)
        return V, R
    
    def fitzhugh_nagumo(self, t, y):
        V, R = y
        dVdt = self.c * (V - V**3 / 3 + R)
        dRdt =  -(V - self.a + self.b * R) / self.c
        return [dVdt, dRdt]
    
    def physics_loss(self, model: torch.nn.Module):
        ts = torch.linspace(0, self.t_extend, steps=10*self.t_extend,).view(-1,1).requires_grad_(True)
        
        Vs = model(ts)[...,0]
        dV = grad(Vs, ts)[0]
        
        Rs = model(ts)[...,1]
        dR = grad(Rs, ts)[0]
        
        ode1 = self.c * (Vs - Vs**3 / 3 + Rs) - dV
        ode2 = -(Vs - self.a + self.b * Rs) / self.c - dR
        
        return torch.mean(ode1**2 + ode2**2)

    
    
    def plot(self, ts=None, V_init=None, R_init=None):
        if V_init is None:
            V_init = self.V0
        if R_init is None:
            R_init = self.R0
        if ts is None:
            ts = torch.linspace(0, self.t_extend, steps=1000)
        V, R = self.physics_law(ts, V_init, R_init)
        
        sns.set_theme()
        # Plot the results
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        
        ax1.plot(ts, V.flatten().numpy(), 'b', label='V (Voltage Variable)')
        ax1.scatter(self.X, self.y[:,0], color='r', label='Data', marker='x')
        ax1.set_ylabel('V(t)')
        ax1.set_ylim(-2, 4)
        ax1.legend()
        
        ax2.plot(ts, R.flatten().numpy(), 'g', label='R (Recovery Variable)')
        ax2.scatter(self.X, self.y[:,1], color='r', label='Data', marker='x')
        ax2.set_xlabel('t')
        ax2.set_ylabel('R(t)')
        ax2.set_ylim(-1, 2)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    
    
    model = FitzHugh_Nagumo()
    print(model.X.shape, model.y.shape)
    print(model.input_dim, model.output_dim)
    
    net = nn.Sequential(
        nn.Linear(1, 15),
        nn.Softplus(),
        nn.Linear(15, 15),
        nn.Softplus(),
        nn.Linear(15, 2)
    )
    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # optimizer.zero_grad()
    # loss = model.physics_loss(net)
    # loss.backward()
    
    # for p in net.parameters():
        # print(p.grad)
    
    # model.plot()
    
    # print(model.physics_loss(net))
    
        
    
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

    # Time span
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
    
