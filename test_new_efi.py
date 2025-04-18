import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PINN.common.utils import get_activation_fn

from torch.func import functional_call
from collections import OrderedDict

import torch.optim as optim
from PINN.common.sgld import SGHMC
import os



class BaseDNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=None, activation_fn='relu'):
        super(BaseDNN, self).__init__()
        # Define the initial layers
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation_fn = get_activation_fn(activation_fn)
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, self.hidden_layers[0]))
        # Add hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        # Add the output layer
        if self.output_dim is not None:
            self.layers.append(nn.Linear(self.hidden_layers[-1], self.output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        return x
    
class HyperEncoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_fn='relu', neck_layer_activation='identity'):
        super().__init__()
        # Define the initial layers
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation_fn = get_activation_fn(activation_fn)
        self.neck_layer_activation = get_activation_fn(neck_layer_activation)
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, self.hidden_layers[0]))
        # Add hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        # Add the output layer
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.output_dim)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        x = self.neck_layer_activation(x)
        theta_all = self.output_layer(x)
        theta_mean = theta_all.mean(dim=0)
        # print("theta_mean", theta_mean.repeat(100, 1).shape)
        # print("theta_all", theta_all.shape)
        # raise
        theta_loss = ((theta_all - theta_mean.repeat(100, 1))**2).sum()
        
        return theta_mean, theta_loss
    
    
    
class PartialEFINet(nn.Module):
    def __init__(self,
                 input_dim=1,
                 output_dim=1,
                 latent_Z_dim=1,
                 fixed_hidden_layers=[15, 15],
                 efi_hidden_layers=[15],
                 activation_fn='relu',
                 pe_dim=0,
                 encoder_hidden_layers=[32, 32, 16],
                 encoder_activation='relu',
                 neck_layer_activation='identity',
                 positive_output=False,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.pe_dim = pe_dim
        self.positive_output = positive_output

        self.fixed_layers = BaseDNN(
            input_dim=input_dim,
            hidden_layers=fixed_hidden_layers,
            activation_fn=activation_fn
        )

        self.efi_layers = BaseDNN(
            input_dim=fixed_hidden_layers[-1],
            hidden_layers=efi_hidden_layers,
            output_dim=output_dim,
            activation_fn=activation_fn
        )

        self.param_shapes = {
            name: param.shape for name, param in self.efi_layers.named_parameters()
        }
        self.total_params = sum(p.numel() for p in self.efi_layers.parameters())

        self.hyper = HyperEncoder(
            input_dim=fixed_hidden_layers[-1] + output_dim + latent_Z_dim,
            hidden_layers=encoder_hidden_layers,
            output_dim=self.total_params + pe_dim + 1,
            activation_fn=encoder_activation,
            neck_layer_activation=neck_layer_activation
        )

        self._cached_param_dict = None
        self._cached_pe_variables = None
        self._cached_log_sd = None

        print("Fixed layer parameters:", sum(p.numel() for p in self.fixed_layers.parameters()))
        print("EFI layer parameters:", sum(p.numel() for p in self.efi_layers.parameters()))
        print("Hypernet parameters:", sum(p.numel() for p in self.hyper.parameters()))

    def encode_efi_params(self, X, Y, Z):
        X, Y, Z = X.to(self.device), Y.to(self.device), Z.to(self.device)       
        X = self.fixed_layers(X)
        xyz = torch.cat([X, Y, Z], dim=1)  # shape: [B, D]
        
        theta_bar, theta_loss = self.hyper(xyz)  # shape: [total_params + pe_dim]
        self._cached_param_dict, self._cached_log_sd, self._cached_pe_variables = self.split_encoder_output(theta_bar)
        return theta_loss

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        h = self.fixed_layers(x)

        if self._cached_param_dict is None:
            raise RuntimeError("Must call encode_efi_params before forward")

        out = functional_call(self.efi_layers, self._cached_param_dict, (h,))
        if self.positive_output:
            out = F.softplus(out)
        return out, self._cached_log_sd, self._cached_pe_variables

    def split_encoder_output(self, flat_vector):
        param_dict = OrderedDict()
        offset = 0
        for name, shape in self.param_shapes.items():
            numel = torch.tensor(shape).prod().item()
            param_dict[name] = flat_vector[offset: offset + numel].view(shape)
            offset += numel
        log_sd = flat_vector[offset]
        pe_variables = flat_vector[offset+1:] if self.pe_dim > 0 else None
        return param_dict, log_sd, pe_variables
    

if __name__ == '__main__':

    torch.manual_seed(42)
    x = torch.linspace(-0.7, 0.7, 100).unsqueeze(1)
    y_true = torch.sin(6 * x) ** 3
    z = torch.randn(100, 1)  # Random latent variable
    y = y_true + 0.1 * torch.randn_like(y_true)  # Add noise
    
    
    # plt.scatter(x.numpy(), y.numpy(), label='Noisy Data', color='blue')
    # plt.plot(torch.linspace(-1, 1, 1000) , torch.sin(2 * torch.pi * torch.linspace(-1, 1, 1000)), label='True Function', color='orange')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()
    
    
    net = PartialEFINet(
        input_dim=1,
        output_dim=1,
        latent_Z_dim=1,
        fixed_hidden_layers=[50, 50],
        efi_hidden_layers=[50],
        activation_fn='tanh',
        encoder_hidden_layers=[32, 32, 16],
        encoder_activation='leaky_relu',
        neck_layer_activation='identity',
        positive_output=False,
        device='cpu'
    )
    lr = 1e-4
    lr1 = 2e-6
    lr2 = 0.5e-6
    sgld_lr = 0.5e-4
    lam = 500

    latent_Z = torch.randn(100, 1).requires_grad_()

    
    # optimizer = optim.SGD(list(net.fixed_layers.parameters()) + list(net.hyper.parameters()), lr=lr, momentum=0.9)
    optimizer1 = optim.SGD(net.fixed_layers.parameters(), lr=lr1, momentum=0.9)
    optimizer2 = optim.SGD(net.hyper.parameters(), lr=lr2, momentum=0.9)
    
    sampler = SGHMC([latent_Z], lr=sgld_lr)

    
    burnin = 10000
    total_steps = 200000
    eval_buffer = []
    save_dir = "./plots"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(total_steps):
        # Sample latent variable
        theta_loss = net.encode_efi_params(x, y, latent_Z)
        pred_y, log_sd, pe_variables = net(x)
        y_loss = F.mse_loss(y, pred_y + latent_Z * log_sd.exp(), reduction='sum')
        z_prior_loss = 0.5 * torch.sum(latent_Z**2)
        Z_loss = lam * (y_loss + theta_loss) + z_prior_loss 
        
        sampler.zero_grad()
        Z_loss.backward()
        sampler.step()
        
        # Update the parameters
        theta_loss = net.encode_efi_params(x, y, latent_Z)
        pred_y, log_sd, pe_variables = net(x)
        y_loss = F.mse_loss(y, pred_y + latent_Z * log_sd.exp(), reduction='sum')
        
        w_loss = lam * (y_loss + theta_loss)
        
        # optimizer.zero_grad()
        # w_loss.backward()
        # optimizer.step()
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        w_loss.backward()
        optimizer1.step()
        optimizer2.step()
        
        if epoch >= burnin:
            with torch.no_grad():
                pred_y, _, _ = net(x)
                eval_buffer.append(pred_y.detach().clone())

        if epoch % 2000 == 0 and epoch >= burnin:
            with torch.no_grad():
                preds = torch.stack(eval_buffer).squeeze(-1)  # [N, B]
                pred_mean = preds.mean(dim=0)
                lower = preds.quantile(0.025, dim=0)
                upper = preds.quantile(0.975, dim=0)

                # Plotting
                plt.figure(figsize=(8, 4))
                x_plot = x.squeeze().cpu().numpy()
                y_plot = y_true.squeeze().cpu().numpy()

                plt.plo(x_plot, y_plot, 'k-', linewidth=1, label="Ground Truth", color='red')
                plt.scatter(x_plot, y.cpu().numpy(), label='Noisy Data', color='orange', alpha=0.3)
                plt.plot(x_plot, pred_mean.cpu().numpy(), 'b-', label="Mean Prediction")
                plt.fill_between(x_plot, lower.cpu().numpy(), upper.cpu().numpy(),
                                color='blue', alpha=0.3, label="95% CI")

                plt.title(f"Step {epoch} - Prediction with 95% Confidence Band")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.grid(True)

                plot_path = os.path.join(save_dir, f"steps.png")
                plt.savefig(plot_path)
                plt.close()
                
        if epoch % 2000 == 0:
            print(f"Epoch {epoch} | y_loss: {y_loss.item():.4f} | theta_loss: {theta_loss.item():.4f} | z_prior_loss: {z_prior_loss.item():.4f}")