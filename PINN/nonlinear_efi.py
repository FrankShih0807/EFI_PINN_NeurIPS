import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import seaborn as sns
from torch.nn.utils import parameters_to_vector
from PINN.common import SGLD
from PINN.common.torch_layers import EFI_Net
from PINN.common.base_pinn import BasePINN
from PINN.common.torch_layers import BaseDNN


class NONLINEAR_EFI(BasePINN):
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
        device="cpu",
    ) -> None:
        super().__init__(
            physics_model,
            dataset,
            hidden_layers,
            activation_fn,
            lr,
            physics_loss_weight,
            save_path,
            device,
        )

        # EFI configs
        self.encoder_kwargs = encoder_kwargs
        self.sgld_lr = sgld_lr
        self.initial_lambda_y = lambda_y
        self.initial_lambda_theta = lambda_theta
        self.lambda_y = lambda_y
        self.lambda_theta = lambda_theta

        self.noise_sd = physics_model.noise_sd

        self.n_samples = self.X.shape[0]

    def _pinn_init(self):
        # init EFI net and optimiser
        self.net = EFI_Net(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
            device=self.device,
            **self.encoder_kwargs
        )
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)

        # init latent noise and sampler
        # self.Z = (self.noise_sd * torch.randn_like(self.y)).requires_grad_().to(self.device)
        self.latent_Z = []
        for d in self.dataset:
            if d['noise_sd'] > 0:
                self.latent_Z.append((d['noise_sd'] * torch.randn_like(d['y'])).requires_grad_().to(self.device))
            else:
                self.latent_Z.append(None)
        
        self.sampler = SGLD([ Z for Z in self.latent_Z if Z is not None], self.sgld_lr)

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

    def train_base_dnn(self, epochs=10000):
        base_net = BaseDNN(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
        ).to(self.device)
        optimiser = optim.Adam(base_net.parameters(), lr=1e-3)
        base_net.train()
        for ep in range(epochs):
            optimiser.zero_grad()
            output = base_net(self.X)
            loss = self.mse_loss(output, self.y)
            loss.backward()
            optimiser.step()
        return base_net

    def optimize_encoder(self, param_vector, steps=5000):

        for _ in range(steps):
            self.net.train()
            # batch_size = self.n_samples
            noise_X = torch.cat([d['X'] for d in self.dataset if d['noise_sd'] > 0], dim=0)
            noise_y = torch.cat([d['y'] for d in self.dataset if d['noise_sd'] > 0], dim=0)
            noise_Z = torch.cat([ Z for Z in self.latent_Z if Z is not None], dim=0)
            batch_size = noise_X.shape[0]

            encoder_output = self.net.encoder(torch.cat([noise_X, noise_y, noise_Z], dim=1))
            loss = F.mse_loss(
                encoder_output, param_vector.repeat(batch_size, 1), reduction="sum"
            )
            # loss = self.mse_loss(encoder_output, param_vector)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    def scheduler(self, epoch, total_epochs):
        # Decay lambda_y and lambda_theta using a power function
        if epoch > 5000:
            decay_factor = 10 ** (epoch / total_epochs)
            self.lambda_y = self.initial_lambda_y * decay_factor
            # self.lambda_theta = self.initial_lambda_theta * decay_factor

    def update(self, epoch, total_epochs):
        # Adjust lambda_y and lambda_theta
        self.scheduler(epoch, total_epochs)

        ## 1. Latent variable sampling (Sample Z)
        self.net.eval()
        # theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        theta_loss = self.theta_loss()
        # y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        y_loss = self.solution_loss()
        z_prior_loss = self.z_prior_loss()
        pde_loss = self.pde_loss()
        Z_loss = (
            self.lambda_y * y_loss
            + self.lambda_theta * theta_loss
            + z_prior_loss + self.physics_loss_weight * pde_loss
        )

        self.sampler.zero_grad()
        Z_loss.backward()
        self.sampler.step()

        ## 2. DNN weights update (Optimize W)
        self.net.train()
        # theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        # y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        # prior_loss = -self.net.gmm_prior_loss() / self.n_samples
        theta_loss = self.theta_loss()
        y_loss = self.solution_loss()
        w_prior_loss = -self.net.gmm_prior_loss() / self.n_samples
        pde_loss = self.pde_loss()

        w_loss = (
            self.lambda_y * (y_loss + w_prior_loss)
            + self.lambda_theta * theta_loss
            + self.physics_loss_weight * pde_loss
        )

        self.optimiser.zero_grad()
        w_loss.backward()
        self.optimiser.step()

    def train(self, epochs=10000, eval_freq=1000):
        self._pinn_init()
        self.collection = []

        # Train BaseDNN
        base_net = self.train_base_dnn()

        # Convert BaseDNN parameters to vector
        param_vector = parameters_to_vector(base_net.parameters())

        # Optimize encoder network
        self.optimize_encoder(param_vector)

        losses = []

        tic = time.time()
        for ep in range(epochs):
            self.update(ep, epochs)
            
            ## 3. Loss calculation
            if (ep+1) % eval_freq == 0:
                toc = time.time()
                loss = self.mse_loss(self.y, self.net(self.X))
                losses.append(loss.item())
                print(f"Epoch {ep+1}/{epochs}, loss: {losses[-1]:.2f}, time: {toc-tic:.2f}s")
                tic = time.time()
                
            if ep > epochs - 1000:
                y_pred = self.evaluate().detach().cpu()
                self.collection.append(y_pred)

        self.physics_model.save_evaluation(self, self.save_path)
        return losses
