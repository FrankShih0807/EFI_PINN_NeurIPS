import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import seaborn as sns
from torch.nn.utils import parameters_to_vector
from PINN.common import SGLD, SGHMC
from PINN.common.torch_layers import EFI_Net
from PINN.common.base_pinn import BasePINN
from PINN.common.torch_layers import BaseDNN
from PINN.common.scheduler import get_schedule


class PINN_EFI(BasePINN):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        encoder_kwargs=dict(),
        annealing_period=0.3,
        grad_norm_max=-1,
        lr=1e-3,
        sgld_lr=1e-3,
        lam=1,
        lambda_pde=1,
        lambda_theta=1,
        pretrain_epochs=0,
        save_path=None,
        device="cpu",
    ) -> None:
        # EFI configs
        self.encoder_kwargs = encoder_kwargs
        self.sgld_lr = sgld_lr
        self.lam = lam
        self.lambda_theta = lambda_theta
        self.pretrain_epochs = pretrain_epochs
        
        super().__init__(
            physics_model=physics_model,
            dataset=dataset,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            lr=lr,
            lambda_pde=lambda_pde,
            save_path=save_path,
            device=device,
        )

        self.annealing_period = annealing_period
        self.grad_norm_max = grad_norm_max
        # # EFI configs
        self.n_samples = self.sol_X.shape[0]
        self.mse_loss = nn.MSELoss(reduction="sum")

    def _pinn_init(self):


        # init latent noise and sampler
        # self.Z = (self.noise_sd * torch.randn_like(self.y)).requires_grad_().to(self.device)
        self.latent_Z = []
        self.noise_sd = []
        for d in self.dataset:
            if d['noise_sd'] > 0:
                self.latent_Z.append((d['noise_sd'] * torch.randn_like(d['y'])).requires_grad_().to(self.device))
                self.noise_sd.append(d['noise_sd'])
            else:
                self.latent_Z.append(None)
                self.noise_sd.append(0)
        self.latent_Z_dim = len([ Z for Z in self.latent_Z if Z is not None])
        # init EFI net and optimiser
        self.net = EFI_Net(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            latent_Z_dim=self.latent_Z_dim,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
            device=self.device,
            **self.encoder_kwargs
        )
        self.optimiser = optim.SGD(self.net.parameters(), lr=self.lr(0))
        self.sampler = SGLD([ Z for Z in self.latent_Z if Z is not None], self.sgld_lr(0))
        # self.sampler = SGHMC([ Z for Z in self.latent_Z if Z is not None], self.sgld_lr(0), alpha=0.1)

    def _get_scheduler(self):
        self.lr = get_schedule(self.lr)
        self.sgld_lr = get_schedule(self.sgld_lr)
        self.lambda_pde = get_schedule(self.lambda_pde)
        self.lam = get_schedule(self.lam)
        self.lambda_theta = get_schedule(self.lambda_theta)
        # self.sparse_threshold = get_schedule(self.encoder_kwargs.get('sparse_threshold', 0.01))
        # self.encoder_kwargs['sparse_threshold'] = self.sparse_threshold(0)
        
    def _update_lr(self, optimiser, lr):
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr
    
    def solution_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'solution' and d['noise_sd'] > 0:
                loss += self.mse_loss(d['y'], self.net(d['X']) + self.latent_Z[i])
            elif d['category'] == 'solution':
                loss += self.mse_loss(d['y'], self.net(d['X']))
        return loss
    
    def theta_loss(self):
        # noise_X = torch.cat([d['X'] for d in self.dataset if d['noise_sd'] > 0], dim=0)
        # noise_y = torch.cat([d['y'] for d in self.dataset if d['noise_sd'] > 0], dim=0)
        # # noise_Z = torch.cat([ Z for Z in self.latent_Z if Z is not None], dim=0)
        # noise_Z = torch.cat([ Z/sd for Z, sd in zip(self.latent_Z, self.noise_sd) if Z is not None], dim=0)
        
        # theta_loss = self.net.theta_encode(noise_X, noise_y, noise_Z)
        
        noise_X = []
        noise_y = []
        noise_Z = []
        i = 0
        for d, Z in zip(self.dataset, self.latent_Z):
            if d['noise_sd'] > 0:
                noise_X.append(d['X'])
                noise_y.append(d['y'])
                padded_Z = torch.zeros(d['X'].shape[0], self.latent_Z_dim, device=self.device, requires_grad=True)
                padded_Z = padded_Z.clone()
                padded_Z[:, i:i+1] = Z / d['noise_sd']
                noise_Z.append(padded_Z)
                i += 1
        theta_loss = self.net.theta_encode(torch.cat(noise_X, dim=0), torch.cat(noise_y, dim=0), torch.cat(noise_Z, dim=0))
        return theta_loss
    
    def z_prior_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['noise_sd'] > 0:
                loss += torch.sum(self.latent_Z[i]**2)/2/d['noise_sd']**2
                # loss += torch.mean(self.latent_Z[i]**2)/2/d['noise_sd']**2
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
    
    def pretrain_pde_loss(self, net):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'differential':
                loss += F.mse_loss(self.differential_operator(net, d['X']), d['y'])
        return loss

    def train_base_dnn(self):
        print('Pretraining PINN...')
        base_net = BaseDNN(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
        ).to(self.device)
        optimiser = optim.Adam(base_net.parameters(), lr=3e-4)
        base_net.train()
        for ep in range(self.pretrain_epochs):
            optimiser.zero_grad()
            output = base_net(self.sol_X)
            sol_loss = self.mse_loss(output, self.sol_y)
            pde_loss = self.pretrain_pde_loss(base_net)
            l2_loss = 0
            for param in base_net.parameters():
                l2_loss += torch.sum(param**2)
            loss = sol_loss + pde_loss + l2_loss * 1e-5
            loss.backward()
            optimiser.step()
            if (ep+1) % 1000 == 0:
                print(f"Epoch {ep+1}/{self.pretrain_epochs}, sol_loss: {sol_loss.item():.2f}, pde_loss: {pde_loss.item():.2f}")
                # print('haha')
        print('PINN pretraining done.')
        return base_net

    def optimize_encoder(self, param_vector, steps=1000):
        # optimiser = optim.Adam(self.net.parameters(), lr=3e-4)
        optimiser = optim.SGD(self.net.parameters(), lr=1e-3)
        print('Pretraining EFI...')
        for _ in range(steps):
            self.net.train()
            # batch_size = self.n_samples
            noise_X = torch.cat([d['X'] for d in self.dataset if d['noise_sd'] > 0], dim=0)
            noise_y = torch.cat([d['y'] for d in self.dataset if d['noise_sd'] > 0], dim=0)
            # noise_Z = torch.cat([ torch.randn_like(Z) * sd for Z, sd in zip(self.latent_Z, self.noise_sd) if sd > 0], dim=0)
            # noise_Z = torch.cat([ Z for Z, sd in zip(self.latent_Z, self.noise_sd) if sd > 0], dim=0)
            noise_Z = []
            i = 0
            for d, Z in zip(self.dataset, self.latent_Z):
                if d['noise_sd'] > 0:
                    padded_Z = torch.zeros(d['X'].shape[0], self.latent_Z_dim, device=self.device, requires_grad=True)
                    padded_Z = padded_Z.clone()
                    padded_Z[:, i:i+1] = Z / d['noise_sd']
                    noise_Z.append(padded_Z)
                    i += 1
            noise_Z = torch.cat(noise_Z, dim=0)
            batch_size = noise_X.shape[0]

            encoder_output = self.net.encoder(torch.cat([noise_X, noise_y, noise_Z], dim=1))
            # loss = F.mse_loss(encoder_output, param_vector.repeat(batch_size, 1), reduction="sum")
            loss = F.mse_loss(encoder_output, param_vector.repeat(batch_size, 1), reduction="sum") / batch_size
            w_prior_loss = self.net.gmm_prior_loss() /batch_size
            loss += w_prior_loss

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 100)
            optimiser.step()
        print('EFI pretraining done.')


    def update(self):
        # update training parameters
        annealing_progress = self.progress / self.annealing_period
        lambda_pde = self.lambda_pde(annealing_progress)
        lam = self.lam(annealing_progress)
        lambda_theta = self.lambda_theta(annealing_progress)
        # self.net.sparse_threshold = self.sparse_threshold(self.progress * 3 - 1)
        lr = self.lr(annealing_progress)
        sgld_lr = self.sgld_lr(annealing_progress)
        self._update_lr(self.optimiser, lr)
        self._update_lr(self.sampler, sgld_lr)
        
        
        ## 1. Latent variable sampling (Sample Z)
        self.net.eval()
        theta_loss = self.theta_loss()
        y_loss = self.solution_loss()
        z_prior_loss = self.z_prior_loss()
        pde_loss = self.pde_loss()
        Z_loss = lam * (y_loss + lambda_theta * theta_loss + lambda_pde * pde_loss) + z_prior_loss

        self.sampler.zero_grad()
        Z_loss.backward()
        if self.grad_norm_max > 0:
            nn.utils.clip_grad_norm_([ Z for Z in self.latent_Z if Z is not None], self.grad_norm_max)
        self.sampler.step()

        ## 2. DNN weights update (Optimize W)
        self.net.train()
        theta_loss = self.theta_loss()
        y_loss = self.solution_loss()
        w_prior_loss = self.net.gmm_prior_loss()
        pde_loss = self.pde_loss()

        w_loss = lam * (y_loss + lambda_theta * theta_loss + lambda_pde * pde_loss) + w_prior_loss

        self.optimiser.zero_grad()
        w_loss.backward()
        if self.grad_norm_max > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm_max)
        self.optimiser.step()
        
        # record training parameters
        self.logger.record('train_param/lr', self.optimiser.param_groups[0]['lr'])
        self.logger.record('train_param/sgld_lr', self.sampler.param_groups[0]['lr'])
        self.logger.record('train_param/lambda_pde', lambda_pde)
        # self.logger.record('train_param/sparse_threshold', self.net.sparse_threshold)
        self.logger.record('train/theta_loss', theta_loss.item())
        # print(y_loss.item(), pde_loss.item(), theta_loss.item(), w_prior_loss.item(), z_prior_loss.item())
        # raise
        return y_loss.item(), pde_loss.item()

    def train(self, epochs=10000, eval_freq=-1, burn=0.5, callback=None):
        # Train BaseDNN
        base_net = self.train_base_dnn()

        # Plot pretraining result
        base_net.eval()

        # Convert BaseDNN parameters to vector
        param_vector = parameters_to_vector(base_net.parameters()).to(self.device)

        # Optimize encoder network
        self.optimize_encoder(param_vector)
        
        super().train(epochs, eval_freq, burn, callback)
        
        
