import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel


    
    
        
class SinCos(PhysicsModel):
    def __init__(self, 
                 t_end=20, 
                 t_extend=25,
                 noise_sd=1.0
                 ):
        super().__init__(t_end=t_end, t_extend=t_extend, noise_sd=noise_sd)

        
    def _data_generation(self, n_samples=200):
        t = torch.linspace(0, self.t_end, n_samples).reshape(n_samples, -1)
        Y1, Y2 = self.physics_law(t)
        
        y = torch.cat([Y1, Y2], dim=1)
        y += self.noise_sd * torch.randn_like(y)
        
        return t, y
    
    def _eval_data_generation(self):
        t = torch.linspace(0, self.t_extend, self.t_extend * 10).reshape(self.t_extend * 10, -1)
        return t
    
    def physics_law(self, time):
        Y1 = 3 * np.cos(time)
        Y2 = 3 * np.sin(time)
        return Y1, Y2
    
    def physics_loss(self, model: torch.nn.Module):
        return torch.zeros(1)
