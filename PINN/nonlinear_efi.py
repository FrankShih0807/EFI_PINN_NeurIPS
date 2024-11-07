import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        # activation_fn=nn.Tanh(),
        lr=1e-3,
        physics_loss_weight=10,
        sgld_lr=1e-3,
        lambda_y=1,
        lambda_theta=1,
        save_path=None,
    ) -> None:
        super().__init__(physics_model, hidden_layers, activation_fn, lr, physics_loss_weight, save_path)
        
        # EFI configs
        self.sgld_lr = sgld_lr
        self.initial_lambda_y = lambda_y
        self.initial_lambda_theta = lambda_theta
        self.lambda_y = lambda_y
        self.lambda_theta = lambda_theta
        
        self.noise_sd = physics_model.noise_sd

    
    def _pinn_init(self):
        # init EFI net and optimiser
        # self.net = EFI_Net(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
        self.net = EFI_Net(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn, prior_sd=0.5)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        
        # init latent noise and sampler
        self.Z = (self.noise_sd * torch.randn_like(self.y)).requires_grad_()
        self.sampler = SGLD([self.Z], self.sgld_lr)
        
        # init encoder optimiser
        # self.encoder_optimiser = optim.Adam(self.net.encoder.parameters(), lr=self.lr)

    def train_base_dnn(self, epochs=10000):
        base_net = BaseDNN(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
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
            batch_size = self.X.shape[0]
            encoder_output = self.net.encoder(torch.cat([self.X, self.y, self.Z], dim=1))
            loss = F.mse_loss(encoder_output, param_vector.repeat(batch_size, 1), reduction='sum')
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
        theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        Z_loss = self.lambda_y * y_loss + self.lambda_theta * theta_loss + torch.mean(self.Z**2)/2/self.noise_sd**2

        self.sampler.zero_grad()
        Z_loss.backward()
        self.sampler.step()

        ## 2. DNN weights update (Optimize W)
        self.net.train()
        theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        prior_loss = - self.net.gmm_prior_loss() / self.n_samples

        w_loss = self.lambda_y * (y_loss + prior_loss) + self.physics_loss_weight * self.physics_loss(self.net) + self.lambda_theta * theta_loss 

        self.optimiser.zero_grad()
        w_loss.backward()
        self.optimiser.step()

    def train(self, epochs=10000):
        self._pinn_init()
        self.collection = []
        
        # Train BaseDNN
        base_net = self.train_base_dnn()
        
        # Convert BaseDNN parameters to vector
        param_vector = parameters_to_vector(base_net.parameters())
        
        # Optimize encoder network
        self.optimize_encoder(param_vector)
        
        losses = []
        for ep in range(epochs):
            self.update(ep, epochs)
            
            ## 3. Loss calculation
            if ep % int(epochs / 10) == 0:
                loss = self.mse_loss(self.y, self.net(self.X))
                losses.append(loss.item())
                print(f"Epoch {ep}/{epochs}, loss: {losses[-1]:.2f}")
                
            if ep > epochs - 1000:
                y_pred = self.evaluate()
                self.collection.append(y_pred)
        
        self.physics_model.save_evaluation(self, self.save_path)
        return losses