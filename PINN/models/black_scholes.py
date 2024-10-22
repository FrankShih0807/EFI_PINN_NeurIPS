import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel

class BlackScholes(PhysicsModel):
    def __init__(self, 
                 S0=100,        # Initial stock price
                 K=100,         # Strike price
                 r=0.05,        # Risk-free interest rate
                 sigma=0.2,     # Volatility
                 T=1,           # Time to maturity (in years)
                 t_extend=1     # Extended time for physics loss
                 ):
        super().__init__(S0=S0, K=K, r=r, sigma=sigma, T=T, t_extend=t_extend)

    def _data_generation(self, n_samples=200, noise_sd=1.0):
        t = torch.linspace(0, self.T, n_samples).reshape(n_samples, -1)
        S = self.physics_law(t) + noise_sd * torch.randn(n_samples).reshape(n_samples, -1)

        return t, S

    def physics_law(self, time):
        """
        Black-Scholes formula for the price of a European call option.
        """
        d1 = (torch.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * time) / (self.sigma * torch.sqrt(time))
        d2 = d1 - self.sigma * torch.sqrt(time)
        
        call_price = (self.S0 * self.norm_cdf(d1)) - (self.K * torch.exp(-self.r * time) * self.norm_cdf(d2))
        
        return call_price

    def norm_cdf(self, x):
        """
        Cumulative distribution function for the standard normal distribution.
        """
        return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

    def physics_loss(self, model: torch.nn.Module):
        ts = torch.linspace(0, self.t_extend, steps=self.t_extend).view(-1, 1).requires_grad_(True)
        stock_prices = model(ts)
        
        # Use the Black-Scholes PDE: dV/dt + 0.5 * sigma^2 * S^2 * d²V/dS² + r * S * dV/dS - r * V = 0
        dV_dt = grad(stock_prices, ts)[0]
        dV_dS = grad(stock_prices, stock_prices, create_graph=True)[0]
        d2V_dS2 = grad(dV_dS, stock_prices, create_graph=True)[0]
        
        pde = dV_dt + 0.5 * self.sigma ** 2 * stock_prices ** 2 * d2V_dS2 + self.r * stock_prices * dV_dS - self.r * stock_prices
        
        return torch.mean(pde ** 2)
