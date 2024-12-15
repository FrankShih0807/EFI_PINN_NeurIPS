import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchbnn as bnn
import numpy as np
import torch.distributions as dist
from collections import defaultdict
from PINN.common.gmm import GaussianMixtureModel
from PINN.common.utils import get_activation_fn
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from PINN.common.losses import gmm_loss
    
    
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)  # Or use xavier_normal_
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


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
    

class DropoutDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, activation_fn=F.relu, dropout_rate=0.01):
        super(DropoutDNN, self).__init__()
        # Define input, output dimensions, hidden layers, and activation function
        self.input_dim = input_dim
        self.output_dim = output_dim
        # if hidden_layers is None:
        #     self.hidden_layers = [self.output_dim]
        # else:
        self.hidden_layers = hidden_layers
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout_rate = dropout_rate

        # Initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, self.hidden_layers[0]))  # Input layer

        # Add hidden layers with dropout
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
            self.layers.append(nn.Dropout(p=self.dropout_rate))  # Dropout after each hidden layer

        # Output layer without dropout
        self.layers.append(nn.Linear(self.hidden_layers[-1], self.output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Linear):  # Apply activation after linear layers
                x = self.activation_fn(x)
        x = self.layers[-1](x)  # Output layer without activation
        return x
    


class EFI_Net(nn.Module):
    def __init__(self, 
                 input_dim=1, 
                 output_dim=1, 
                 latent_Z_dim=1,
                 hidden_layers=[15, 15], 
                 activation_fn='relu', 
                 encoder_hidden_layers=None,
                 encoder_activation='relu',
                 prior_sd=0.1, 
                 sparse_sd=0.01,
                 sparsity=1.0,
                 device='cpu'
                 ):
        super(EFI_Net, self).__init__()
        
        self.device = device
        # EFI Net Info
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_Z_dim = latent_Z_dim
        self.hidden_layers = hidden_layers
        self.activation_fn = get_activation_fn(activation_fn)

        # Encoder Net Info
        self.encoder_input_dim = self.input_dim + self.output_dim + self.latent_Z_dim
        self.encoder_activation = get_activation_fn(encoder_activation)
        
        # sparse prior settings
        self.sparsity = sparsity
        self.prior_sd = prior_sd
        self.sparse_sd = sparse_sd
        
        sample_net = nn.ModuleList()
        sample_net.append(nn.Linear(input_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            sample_net.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        sample_net.append(nn.Linear(hidden_layers[-1], output_dim))
        
        
        self.n_layers = len(sample_net)
        self.n_parameters = sum([p.numel() for p in sample_net.parameters()])
        self.nn_shape = defaultdict(list)
        self.nn_numel = defaultdict(list)
        self.nn_tensor = defaultdict(list)

        
        for key, value in sample_net.named_parameters():
            if 'weight' in key:
                self.nn_shape['weight'].append(value.shape)
                self.nn_numel['weight'].append(value.numel())
            if 'bias' in key:
                self.nn_shape['bias'].append(value.shape)
                self.nn_numel['bias'].append(value.numel())
        
        self.encoder = BaseDNN(input_dim=self.encoder_input_dim, hidden_layers=encoder_hidden_layers, output_dim=self.n_parameters, activation_fn=self.encoder_activation).to(self.device)

    def split_encoder_output(self, theta):
        '''Split encoder output into network layer shapes

        Args:
            theta (tensor): encoder ouput mean

        Returns:
            weight_tensors, bias_tensors: tensors
        '''
        theta_weight, theta_bias = torch.split(theta, [sum(self.nn_numel['weight']), sum(self.nn_numel['bias'])] , dim=-1)
        theta_weight_split = torch.split(theta_weight, self.nn_numel['weight'], dim=-1)
        theta_bias_split = torch.split(theta_bias, self.nn_numel['bias'], dim=-1)
        
        weight_tensors = [theta_weight_split[i].view(*shape).to(self.device) for i, shape in enumerate(self.nn_shape['weight'])]
        bias_tensors = [theta_bias_split[i].view(*shape).to(self.device) for i, shape in enumerate(self.nn_shape['bias'])]
        
        return weight_tensors, bias_tensors
            
    
    def forward(self, x):
        x = x.to(self.device)
        for i in range(self.n_layers-1):
            x = self.activation_fn(F.linear(x, self.weight_tensors[i], self.bias_tensors[i]))
        x = F.linear(x, self.weight_tensors[-1], self.bias_tensors[-1])
        return x

    
    def theta_encode(self, X, y, Z):
        '''Encode X, y, and Z into theta
        Args:
            X (tensor): explanatory variable
            y (tensor): response variable
            Z (tensor): noise variable

        Returns:
            tensor: flattend theta
        '''
        X, y, Z = X.to(self.device), y.to(self.device), Z.to(self.device)
        
        batch_size = X.shape[0]
        xyz = torch.cat([X, y, Z], dim=1).to(self.device)
        batch_theta = self.encoder(xyz)
        theta_bar = batch_theta.mean(dim=0)
        theta_loss = F.mse_loss(batch_theta, theta_bar.repeat(batch_size, 1), reduction='sum')
        self.weight_tensors, self.bias_tensors = self.split_encoder_output(theta_bar)
        return theta_loss
        
    def gmm_prior_loss(self):
        loss = 0
        for p in self.parameters():
            loss += gmm_loss(p, self.prior_sd, self.sparse_sd, self.sparsity).sum()
        return loss
    
        
class EFI_Discovery_Net(nn.Module):
    def __init__(self, 
                 input_dim=1, 
                 output_dim=1, 
                 variable_dim=0,
                 hidden_layers=[15, 15], 
                 activation_fn=F.softplus, 
                 prior_sd=0.1, 
                 sparse_sd=0.01, 
                 sparsity=1):
        super(EFI_Discovery_Net, self).__init__()
        
        # EFI Net Info
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.variable_dim = variable_dim
        self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
        
        
        
        # Encoder Net Info
        self.encoder_input_dim = self.input_dim + 2 * self.output_dim
        self.sparsity = sparsity
        self.prior_sd = prior_sd
        self.sparse_sd = sparse_sd
        
        sample_net = nn.ModuleList()
        sample_net.append(nn.Linear(input_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            sample_net.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        sample_net.append(nn.Linear(hidden_layers[-1], output_dim))
        
        self.n_layers = len(sample_net)
        self.n_parameters = sum([p.numel() for p in sample_net.parameters()])
        self.nn_shape = defaultdict(list)
        self.nn_numel = defaultdict(list)
        self.nn_tensor = defaultdict(list)

        
        for key, value in sample_net.named_parameters():
            if 'weight' in key:
                self.nn_shape['weight'].append(value.shape)
                self.nn_numel['weight'].append(value.numel())
            if 'bias' in key:
                self.nn_shape['bias'].append(value.shape)
                self.nn_numel['bias'].append(value.numel())
        

        self.gmm = GaussianMixtureModel(prior_sd, sparse_sd)
        
        self.encoder = BaseDNN(input_dim=self.encoder_input_dim, output_dim=self.n_parameters+self.variable_dim, activation_fn=activation_fn)
        for p in self.parameters():
            p.data = torch.randn_like(p.data) * 0.001
            

    def split_encoder_output(self, theta):
        '''Split encoder output into network layer shapes

        Args:
            theta (tensor): encoder ouput mean

        Returns:
            weight_tensors, bias_tensors: tensors
        '''
        theta_weight, theta_bias, theta_variable = torch.split(theta, [sum(self.nn_numel['weight']), sum(self.nn_numel['bias']), self.variable_dim] , dim=-1)
        theta_weight_split = torch.split(theta_weight, self.nn_numel['weight'], dim=-1)
        theta_bias_split = torch.split(theta_bias, self.nn_numel['bias'], dim=-1)
        
        weight_tensors = []
        bias_tensors = []
        for i, shape in enumerate(self.nn_shape['weight']):
            weight_tensors.append(theta_weight_split[i].view(*shape))
        for i, shape in enumerate(self.nn_shape['bias']):
            bias_tensors.append(theta_bias_split[i].view(*shape))
        
        return weight_tensors, bias_tensors, theta_variable
    
    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.activation_fn(x @ self.weight_tensors[i].T + self.bias_tensors[i])
        x = x @ self.weight_tensors[-1].T + self.bias_tensors[-1]
        
        return x

    
    def theta_encode(self, X, y, Z):
        '''Encode X, y, and Z into theta
        Args:
            X (tensor): explanatory variable
            y (tensor): response variable
            Z (tensor): noise variable

        Returns:
            tensor: flattend theta
        '''
        batch_size = X.shape[0]
        xyz = torch.cat([X, y, Z], dim=1)
        batch_theta = self.encoder(xyz)
        theta_bar = batch_theta.mean(dim=0)
        theta_loss = F.mse_loss(batch_theta, theta_bar.repeat(batch_size, 1), reduction='sum')
        theta_loss += self.sparsity_loss(theta_bar[:self.n_parameters-self.variable_dim])
        
        self.weight_tensors, self.bias_tensors, self.variable_tensor = self.split_encoder_output(theta_bar)
        
        return theta_loss
        
    def gmm_prior_loss(self, sparsity=None):
        if sparsity is None:
            sparsity = self.sparsity
        log_prior = 0
        for p in self.parameters():
            log_prior += self.gmm.log_prob(p.flatten(), sparsity).sum()
        return log_prior

# class MLP(nn.Module):
#     def __init__(self,in_features : int, out_features: int, hidden_features: int,num_hidden_layers: int) -> None:
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
        
#         self.linear_in = nn.Linear(in_features,hidden_features)
#         self.linear_out = nn.Linear(hidden_features,out_features)
        
#         self.activation = torch.tanh
#         self.layers = nn.ModuleList([self.linear_in] + [nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers)  ])
        
         
#     def forward(self,x):
#         for layer in self.layers:
#             x = self.activation(layer(x))
    
#         return self.linear_out(x)


# class DeepONet(nn.Module):
#     def __init__(self,out_features,branch,trunk) -> None:
#         super().__init__()
#         if branch.out_features != trunk.out_features:
#             raise ValueError('Branch and trunk networks must have the same output dimension')
#         latent_features = branch.out_features
#         self.branch = branch
#         self.trunk = trunk
#         self.fc = nn.Linear(latent_features,out_features,bias = False)
        

#     def forward(self,y,u):
#         return self.fc(self.trunk(y)*self.branch(u))

    
class BayesianPINNNet(nn.Module):
    def __init__(self, sigma_diff, sigma_sol, physics_model, num_bd):
        super(BayesianPINNNet, self).__init__()

        self.fnn = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1), 
        )

        self.sigma_diff = sigma_diff
        self.sigma_sol = sigma_sol
        self.num_bd = num_bd
        
        self.differential_operator = physics_model.differential_operator
        
    def forward(self, X):
        x = X[:-self.num_bd].requires_grad_(True)
        pde = self.differential_operator(self.fnn, x)
        u_bd = self.fnn(X[-self.num_bd:])

        return torch.cat([pde / (self.sigma_diff * 2 ** 0.5), u_bd / (self.sigma_sol * 2 ** 0.5)], dim=0)

class HyperLinear(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim, activation_fn=nn.Identity()):
        """
        Custom linear layer with weights and biases generated from a latent vector z.
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            feature_dim (int): Dimensionality of the latent vector z.
        """
        super(HyperLinear, self).__init__()
        self.feature_dim = feature_dim
        self.activation_fn = get_activation_fn(activation_fn)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Weight generator: maps z to a weight matrix of shape (output_dim, input_dim)
        self.weight_gen = nn.Sequential(
            self.activation_fn,
            nn.Linear(feature_dim, output_dim * input_dim)
        )
        
        # Bias generator: maps z to a bias vector of shape (output_dim)
        self.bias_gen = nn.Sequential(
            self.activation_fn,
            nn.Linear(feature_dim, output_dim)
        )
        
        
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    
    def encode_weight(self, latent_features):
        weight_i = self.weight_gen(latent_features)
        self.weight = weight_i.mean(dim=0).view(self.output_dim, self.input_dim)
        # self.weight = self.weight_gen(latent_features).mean(dim=0).view(self.output_dim, self.input_dim)
        bias_i = self.bias_gen(latent_features)
        self.bias = bias_i.mean(dim=0)
        weight_loss = F.mse_loss(weight_i, self.weight.view(1, -1).expand(weight_i.shape), reduction='sum')
        bias_loss = F.mse_loss(bias_i, self.bias.view(1, -1).expand(bias_i.shape), reduction='sum')
        
        return weight_loss + bias_loss



class EFI_Net_v2(nn.Module):
    def __init__(self, 
                 input_dim=1, 
                 output_dim=1, 
                 latent_Z_dim=1,
                 hidden_layers=[30, 30], 
                 activation_fn='relu', 
                 encoder_hidden_layers=[64, 64],
                 encoder_activation='relu',
                 prior_sd=0.1, 
                 sparse_sd=0.01,
                 sparsity=1.0,
                 device='cpu'
                 ):
        super(EFI_Net_v2, self).__init__()
        
        self.device = device
        # EFI Net Info
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_Z_dim = latent_Z_dim
        self.hidden_layers = hidden_layers
        self.activation_fn = get_activation_fn(activation_fn)

        # Encoder Net Info
        self.encoder_input_dim = self.input_dim + self.output_dim + self.latent_Z_dim
        self.encoder_activation = get_activation_fn(encoder_activation)
        self.feature_dim = encoder_hidden_layers[-1]
        self.feature_encoder = BaseDNN(input_dim=self.encoder_input_dim, hidden_layers=encoder_hidden_layers, activation_fn=self.encoder_activation).to(self.device)
        
        # sparse prior settings
        self.sparsity = sparsity
        self.prior_sd = prior_sd
        self.sparse_sd = sparse_sd
        
        self.layers = nn.ModuleList()
        self.layers.append(HyperLinear(input_dim, hidden_layers[0], self.feature_dim))
        for i in range(1, len(hidden_layers)):
            self.layers.append(HyperLinear(hidden_layers[i-1], hidden_layers[i], self.feature_dim))
        self.layers.append(HyperLinear(hidden_layers[-1], output_dim, self.feature_dim))
        
        
            
    def forward(self, x):
        x = x.to(self.device)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        return x

    
    def encode_weights(self, X, y, Z):
        '''Encode X, y, and Z into theta
        Args:
            X (tensor): explanatory variable
            y (tensor): response variable
            Z (tensor): latent noise variable
        '''
        X, y, Z = X.to(self.device), y.to(self.device), Z.to(self.device)
        
        feature = self.feature_encoder(torch.cat([X, y, Z], dim=1))
        theta_loss = 0
        for layer in self.layers:
            theta_loss += layer.encode_weight(feature)
        return theta_loss

        
    def gmm_prior_loss(self):
        loss = 0
        for p in self.parameters():
            loss += gmm_loss(p, self.prior_sd, self.sparse_sd, self.sparsity).sum()
        return loss
    


if __name__ == '__main__':


    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    z = torch.randn(10, 1)
    
    hypernet = nn.Sequential(
        nn.Linear(3, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
    )
    
    feature = hypernet(torch.cat([x, y, z], dim=1))
    print(feature.shape)
    
    layer = HyperLinear(30, 40, 16)
    
    theta_loss = layer.encode_weight(feature)
    print(theta_loss)
    
    
    # for name, p in layer.named_parameters():
    #     print(name, p.shape)
    
    efi_net = EFI_Net_v2(input_dim=1, 
                         output_dim=1, 
                         latent_Z_dim=1, 
                         hidden_layers=[30, 30], 
                         activation_fn='relu', 
                         encoder_hidden_layers=[16, 16, 16], 
                         encoder_activation='relu', 
                         prior_sd=0.1, 
                         sparse_sd=0.01, 
                         sparsity=1.0, 
                         device='cpu'
                         )
    
    theta_loss = efi_net.encode_weights(x, y, z)
    print(theta_loss)
    
    y_pred = efi_net(x)
    print(y_pred)
    
    gmm_loss = efi_net.gmm_prior_loss()
    print(gmm_loss)
    # for name, p in efi_net.named_parameters():
    #     print(name, p.shape)
    