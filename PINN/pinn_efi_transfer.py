import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PINN.common import SGHMC
from PINN.common.torch_layers import TransferEFI
from PINN.common.base_pinn import BasePINN
from PINN.common.scheduler import get_schedule
import time

class PINN_EFI_Transfer(BasePINN):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[15, 15],
        feature_extractor_layers=None,
        feature_dim=None,
        efi_hidden_layers=[50, 50],
        activation_fn=nn.Softplus(beta=10),
        encoder_kwargs=dict(),
        annealing_period=0.3,
        grad_norm_max=-1,
        lr=1e-3,
        sgd_momentum=0.0,
        sgld_lr=1e-3,
        sgld_alpha=1.0,
        lam=1,
        lambda_pde=1,
        lambda_theta=1,
        positive_output=False,
        save_path=None,
        device="cpu",
    ) -> None:
        # EFI configs
        self.feature_extractor_layers = feature_extractor_layers
        self.feature_dim = feature_dim
        self.efi_hidden_layers = efi_hidden_layers
        
        self.encoder_kwargs = encoder_kwargs
        self.sgd_momentum = sgd_momentum
        self.sgld_lr = sgld_lr
        self.sgld_alpha = sgld_alpha
        self.lam = lam
        self.lambda_theta = lambda_theta
        self.pe_dim = physics_model.pe_dim
        self.positive_output = positive_output
        
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
        # parameter estimation configs
        self.annealing_period = annealing_period
        self.grad_norm_max = grad_norm_max
        # # EFI configs
        self.n_samples = self.sol_X.shape[0]
        self.mse_loss = nn.MSELoss(reduction="sum")

    def _pinn_init(self):
        self.latent_Z = []
        self.noise_sd = []
        for d in self.dataset:
            if d['noise_sd'] > 0:
                # self.latent_Z.append((d['noise_sd'] * torch.randn_like(d['y'])).requires_grad_().to(self.device))
                self.latent_Z.append((torch.randn_like(d['y'])).requires_grad_().to(self.device))
                self.noise_sd.append(d['noise_sd'])
            else:
                self.latent_Z.append(None)
                self.noise_sd.append(0)
        # self.latent_Z_dim = len([ Z for Z in self.latent_Z if Z is not None])
        self.latent_Z_dim = sum([ Z.shape[1] for Z in self.latent_Z if Z is not None])
        print(f"latent_Z_dim: {self.latent_Z_dim}")
        # init EFI net and optimiser
        self.net = TransferEFI(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            latent_Z_dim=self.latent_Z_dim,
            feature_extractor_layers=self.feature_extractor_layers,
            feature_dim=self.feature_dim,
            efi_hidden_layers=self.efi_hidden_layers,
            activation_fn=self.activation_fn,
            pe_dim=self.pe_dim,
            positive_output=self.positive_output,
            device=self.device,
            **self.encoder_kwargs
        )
            
            
        
        self.pretrain_optimiser = optim.Adam( list(self.net.feature_extractor.parameters()) + list(self.net.efi_layers.parameters()) , lr=0.3e-3)
        self.optimiser = optim.SGD(self.net.hyper.parameters(), lr=self.lr(0), momentum=self.sgd_momentum(0))
        # self.optimiser = optim.SGD(list(self.net.feature_extractor.parameters()) + list(self.net.hyper.parameters()) , lr=self.lr(0), momentum=self.sgd_momentum(0))
        self.sampler = SGHMC([ Z for Z in self.latent_Z if Z is not None], self.sgld_lr(0), alpha=self.sgld_alpha(0))

    def _get_scheduler(self):
        self.lr = get_schedule(self.lr)
        self.sgd_momentum = get_schedule(self.sgd_momentum)
        self.sgld_lr = get_schedule(self.sgld_lr)
        self.sgld_alpha = get_schedule(self.sgld_alpha)
        self.lambda_pde = get_schedule(self.lambda_pde)
        self.lam = get_schedule(self.lam)
        self.lambda_theta = get_schedule(self.lambda_theta)

        
    def _update_optimiser_kwargs(self, optimiser, kwargs):
        for param_group in optimiser.param_groups:
            for key, value in kwargs.items():
                param_group[key] = value

    def solution_loss(self, reduction='sum'):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'solution' and d['noise_sd'] > 0:
                y_dim = d['y'].shape[1]
                # print(f"y_dim: {y_dim}")
                # print(self.net(d['X']).shape)
                loss += F.mse_loss(d['y'], self.net(d['X'])[:, 0:y_dim] + self.latent_Z[i] * d['noise_sd'], reduction=reduction)
            elif d['category'] == 'solution':
                loss += F.mse_loss(d['y'], self.net(d['X'])[:, 0:y_dim], reduction=reduction)
        return loss
    
    def theta_loss(self):
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
                padded_Z[:, i:i+1] = Z
                # padded_Z[:, i:i+1] = Z 
                noise_Z.append(padded_Z)
                i += 1
        theta_loss = self.net.encode_efi_params(torch.cat(noise_X, dim=0), torch.cat(noise_y, dim=0), torch.cat(noise_Z, dim=0))
        return theta_loss
    
    def z_prior_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['noise_sd'] > 0:
                loss += torch.sum(self.latent_Z[i]**2)/2
        return loss
    
    def pde_loss(self, reduction='sum'):
        if self.pe_dim == 0:
            pe_variables = None
        else:
            pe_variables = self.net.pe_variables
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'differential':
                if d['noise_sd'] > 0:
                    diff_o = self.differential_operator(self.net, d['X'], pe_variables) + self.latent_Z[i] * d['noise_sd']
                    loss += F.mse_loss(diff_o, d['y'], reduction=reduction)
                else:
                    diff_o = self.differential_operator(self.net, d['X'], pe_variables)
                    loss += F.mse_loss(diff_o, d['y'], reduction=reduction)
        return loss

    def optimize_encoder(self):
        print('Pretraining EFI...')
        
        noise_X = torch.cat([d['X'] for d in self.dataset if d['noise_sd'] > 0], dim=0)
        noise_y = torch.cat([d['y'] for d in self.dataset if d['noise_sd'] > 0], dim=0)     
        noise_Z = []
        i = 0
        for d, Z in zip(self.dataset, self.latent_Z):
            if d['noise_sd'] > 0:
                y_dim = d['y'].shape[1]
                padded_Z = torch.zeros(d['X'].shape[0], self.latent_Z_dim, device=self.device, requires_grad=True)
                padded_Z = padded_Z.clone()
                padded_Z[:, i:i+y_dim] = Z
                noise_Z.append(padded_Z)
                i += y_dim
        noise_Z = torch.cat(noise_Z, dim=0).detach().clone()
        self.net.pretrain(noise_X, noise_y, noise_Z)
        


    def update(self):
        # update training parameters
        annealing_progress = self.progress / self.annealing_period
        lambda_pde = self.lambda_pde(annealing_progress)
        lam = self.lam(annealing_progress)
        lambda_theta = self.lambda_theta(annealing_progress)

        lr = self.lr(self.progress)
        sgd_momentum = self.sgd_momentum(annealing_progress)
        sgld_lr = self.sgld_lr(self.progress)
        sgld_alpha = self.sgld_alpha(annealing_progress)
        self.cur_lr = lr
        self.cur_sgld_lr = sgld_lr
        self._update_optimiser_kwargs(self.optimiser, dict(lr=lr, momentum=sgd_momentum))
        self._update_optimiser_kwargs(self.sampler, dict(lr=sgld_lr, alpha=sgld_alpha))
        
        
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
        pde_loss = self.pde_loss()

        w_loss = lam * (y_loss + lambda_theta * theta_loss + lambda_pde * pde_loss) 

        
        self.optimiser.zero_grad()
        w_loss.backward()
        
        grad_norm = torch.sqrt(sum(p.grad.norm(2)**2 for p in self.net.parameters() if p.grad is not None)).item()
            
        # if self.grad_norm_max > 0 and self.progress < self.annealing_period:
        if self.grad_norm_max > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm_max)
        self.optimiser.step()
        
        # record training parameters
        self.logger.record('train_param/lr', lr, exclude='csv')
        self.logger.record('train_param/sgd_momentum', sgd_momentum, exclude='csv')
        self.logger.record('train_param/sgld_lr', sgld_lr, exclude='csv')
        self.logger.record('train_param/sgld_alpha', sgld_alpha, exclude='csv')
        self.logger.record('train/grad_norm', grad_norm, exclude='csv')
        self.logger.record('train_param/lambda', lam, exclude='csv')

        self.logger.record('train/theta_loss', theta_loss.item())
        
        self.pe_variables = self.net.pe_variables
        return y_loss.item(), pde_loss.item()

    def train(self, epochs=10000, eval_freq=-1, burn=0.1, callback=None):
        # Pretrain DNN
        tic = time.time()
        for epoch in range(1000):
            
            y_loss = self.solution_loss(reduction='mean')
            pde_loss = self.pde_loss(reduction='mean')
            loss = y_loss + pde_loss
            self.pretrain_optimiser.zero_grad()
            loss.backward()            
            self.pretrain_optimiser.step()
            
            if (epoch+1) % 1000 == 0:
                toc = time.time()
                print(f"Pretraining step {epoch+1}, Loss: {loss.item():.5f}, Time: {toc-tic:.2f}s")
                tic = time.time()
                # print(f"Pretraining step {epoch}, Loss: {loss.item():.5f}")
        
        self.optimize_encoder()
        super().train(epochs, eval_freq, burn, callback)
