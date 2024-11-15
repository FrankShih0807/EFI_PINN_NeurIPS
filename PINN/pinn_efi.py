import torch
import torch.nn as nn
import torch.optim as optim
from PINN.common import SGLD
from PINN.common.torch_layers import EFI_Net
from PINN.common.base_pinn import BasePINN
from copy import deepcopy


class PINN_EFI(BasePINN):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        encoder_kwargs=dict(),
        lr=1e-3,
        physics_loss_weight=10,
        sgld_lr=1e-3,
        lambda_y=1,
        lambda_theta=1,
        save_path=None,
        device='cpu'
    ) -> None:
        super().__init__(physics_model, dataset, hidden_layers, activation_fn, lr, physics_loss_weight, save_path, device)
        
        # EFI configs
        self.encoder_kwargs = encoder_kwargs
        self.sgld_lr = sgld_lr
        self.lambda_y = lambda_y
        self.lambda_theta = lambda_theta
        
        self.noise_sd = physics_model.noise_sd
        
        self.n_samples = self.X.shape[0]
        
        # self._pinn_init()
    
    def _pinn_init(self):
        # init EFI net and optimiser
        self.net = EFI_Net(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn, device=self.device, **self.encoder_kwargs)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        
        self.latent_Z = []
        for d in self.dataset:
            if d['noise_sd'] > 0:
                self.latent_Z.append((d['noise_sd'] * torch.randn_like(d['y'])).requires_grad_().to(self.device))
            else:
                self.latent_Z.append(None)

        self.sampler = SGLD([p for p in self.latent_Z if p is not None], self.sgld_lr)
    
    def solution_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'solution' and d['noise_sd'] > 0:
                loss += self.mse_loss(d['y'], self.net(d['X']) + self.latent_Z[i])
            elif d['category'] == 'solution':
                loss += self.mse_loss(d['y'], self.net(d['X']))
        return loss
    
    def theta_loss(self):
        noise_X = torch.cat([d['X'] for d in self.dataset if d['noise_sd'] > 0], dim=0)
        noise_y = torch.cat([d['y'] for d in self.dataset if d['noise_sd'] > 0], dim=0)
        noise_Z = torch.cat([ Z for Z in self.latent_Z if Z is not None], dim=0)
        
        theta_loss = self.net.theta_encode(noise_X, noise_y, noise_Z)
        return theta_loss
    
    def z_prior_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['noise_sd'] > 0:
                loss += torch.mean(self.latent_Z[i]**2)/2/d['noise_sd']**2
        return loss
    
    def pde_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'differential':
                if d['noise_sd'] > 0:
                    diff_o = self.differential_operator(self.net, d['X']) + self.latent_Z[i]
                    loss += self.mse_loss(diff_o, d['y'])
                else:
                    diff_o = self.differential_operator(self.net, d['X'])
                    loss += self.mse_loss(diff_o, d['y'])
        return loss

    def update(self):
        ## 1. Latent variable sampling (Sample Z)
        self.net.eval()

        theta_loss = self.theta_loss()
        y_loss = self.solution_loss()
        z_prior_loss = self.z_prior_loss()
        Z_loss = self.lambda_y * y_loss + self.lambda_theta * theta_loss + z_prior_loss + self.physics_loss_weight * self.pde_loss()
        

        self.sampler.zero_grad()
        Z_loss.backward()
        self.sampler.step()

        ## 2. DNN weights update (Optimize W)
        
        self.net.train()

        theta_loss = self.theta_loss()
        y_loss = self.solution_loss()
        w_prior_loss = - self.net.gmm_prior_loss() / self.n_samples
        
        pde_loss = self.pde_loss()
        w_loss = self.lambda_y * (y_loss + w_prior_loss) + self.lambda_theta * theta_loss + self.physics_loss_weight * pde_loss
        
        

        self.optimiser.zero_grad()
        w_loss.backward()
        self.optimiser.step()
        
        return y_loss.item(), pde_loss.item()

