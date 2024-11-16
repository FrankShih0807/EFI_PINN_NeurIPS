import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn
import numpy as np
import torch.distributions as dist
from collections import defaultdict
from PINN.common.gmm import GaussianMixtureModel
from PINN.common.utils import get_activation_fn
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

    
class BaseDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation_fn):
        super(BaseDNN, self).__init__()
        # Define the initial layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        # if hidden_layers is None:
        #     self.hidden_layers = [self.output_dim]
        # else:
        self.hidden_layers = hidden_layers
        self.activation_fn = get_activation_fn(activation_fn)
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, self.hidden_layers[0]))
        # Add hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        # Add the output layer
        self.layers.append(nn.Linear(self.hidden_layers[-1], self.output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        return x
    

class EncoderDNN(BaseDNN):
    def __init__(self, input_dim, output_dim, hidden_layers=None, activation_fn=F.relu):
        if hidden_layers is None:
            last_hidden = 2 ** int(np.log2(output_dim))
            # hidden_layers = [last_hidden//4, last_hidden//2 , last_hidden]
            hidden_layers = [last_hidden//2 , last_hidden]
            # hidden_layers = [input_dim * 2, last_hidden // 4, last_hidden // 2, last_hidden]
        super(EncoderDNN, self).__init__(input_dim, output_dim, hidden_layers, activation_fn)

    

class DropoutDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, activation_fn=F.relu, dropout_rate=0.01):
        super(DropoutDNN, self).__init__()
        # Define input, output dimensions, hidden layers, and activation function
        self.input_dim = input_dim
        self.output_dim = output_dim
        if hidden_layers is None:
            self.hidden_layers = [self.output_dim]
        else:
            self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
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
    
               
class BayesianNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, activation_fn=F.relu):
        super(BayesianNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if hidden_layers is None:
            self.hidden_layers = [self.output_dim]
        else:
            self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
        # Define layers with Bayesian Linear (weights are distributions)
        self.layers = nn.ModuleList()
        self.layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_dim, out_features=self.hidden_layers[0]))
        # Add hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.hidden_layers[i-1], out_features=self.hidden_layers[i]))
        # Add the output layer
        self.layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.hidden_layers[-1], out_features=self.output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        return x


class EFI_Net(nn.Module):
    def __init__(self, 
                 input_dim=1, 
                 output_dim=1, 
                 hidden_layers=[15, 15], 
                 activation_fn='relu', 
                 sparse_threshold=0.01,
                 encoder_hidden_layers=None,
                 encoder_activation='relu',
                 prior_sd=0.1, 
                #  sparse_sd=0.01, 
                #  sparsity=1.0,
                 device='cpu'
                 ):
        super(EFI_Net, self).__init__()
        
        self.device = device
        # EFI Net Info
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation_fn = get_activation_fn(activation_fn)
        self.sparse_threshold = sparse_threshold

        # Encoder Net Info
        self.encoder_input_dim = self.input_dim + 2 * self.output_dim
        self.encoder_activation = get_activation_fn(encoder_activation)
        # self.sparsity = sparsity
        self.prior_sd = prior_sd
        # self.sparse_sd = sparse_sd
        
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
        

        # self.gmm = GaussianMixtureModel(prior_sd, sparse_sd)
        self.prior_dist = dist.Normal(0, prior_sd)
        
        # self.encoder = BaseDNN(input_dim=self.encoder_input_dim, output_dim=self.n_parameters, activation_fn=activation_fn)
        self.encoder = EncoderDNN(input_dim=self.encoder_input_dim, output_dim=self.n_parameters, activation_fn=self.encoder_activation, hidden_layers=encoder_hidden_layers).to(self.device)
        for p in self.parameters():
            p.data = torch.randn_like(p.data) * 0.001
            p.data = p.data.to(self.device)


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
        
        # weight_tensors = []
        # bias_tensors = []
        # for i, shape in enumerate(self.nn_shape['weight']):
        #     weight_tensors.append(theta_weight_split[i].view(*shape))
        # for i, shape in enumerate(self.nn_shape['bias']):
        #     bias_tensors.append(theta_bias_split[i].view(*shape))
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
        theta_loss = F.mse_loss(batch_theta, theta_bar.repeat(batch_size, 1), reduction='mean')
        theta_loss += self.sparsity_loss(theta_bar)
        
        self.weight_tensors, self.bias_tensors = self.split_encoder_output(theta_bar)
        
        return theta_loss
        
    def gmm_prior_loss(self):
        # if sparsity is None:
        #     sparsity = self.sparsity
        log_prior = 0
        for p in self.parameters():
            # log_prior += self.gmm.log_prob(p.flatten(), sparsity).sum()
            log_prior += self.prior_dist.log_prob(p).sum()
        return log_prior
    
    def sparsity_loss(self, theta):
        # xi = 0.01
        # a = 1
        return torch.where(theta.abs() > self.sparse_threshold, torch.zeros_like(theta.abs()).to(self.device), theta.abs()).sum()
    
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
        
        self.encoder = EncoderDNN(input_dim=self.encoder_input_dim, output_dim=self.n_parameters+self.variable_dim, activation_fn=activation_fn)
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
    
    def sparsity_loss(self, theta):
        xi = 0.01
        a = 1
        return torch.where(theta.abs() > a * xi, torch.zeros_like(theta.abs()), theta.abs()).sum()

class EFI_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, activation_fn=F.relu):
        super(EFI_Encoder, self).__init__()
        # Define the initial layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        if hidden_layers is None:
            self.hidden_layers = [self.output_dim]
        else:
            self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, self.hidden_layers[0]))
        # Add hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        # Add the output layer
        self.layers.append(nn.Linear(self.hidden_layers[-1], output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        return x    

    
    
class MLP(nn.Module):
    def __init__(self,in_features : int, out_features: int, hidden_features: int,num_hidden_layers: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear_in = nn.Linear(in_features,hidden_features)
        self.linear_out = nn.Linear(hidden_features,out_features)
        
        self.activation = torch.tanh
        self.layers = nn.ModuleList([self.linear_in] + [nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers)  ])
        
         
    def forward(self,x):
        for layer in self.layers:
            x = self.activation(layer(x))
    
        return self.linear_out(x)


class DeepONet(nn.Module):
    def __init__(self,out_features,branch,trunk) -> None:
        super().__init__()
        if branch.out_features != trunk.out_features:
            raise ValueError('Branch and trunk networks must have the same output dimension')
        latent_features = branch.out_features
        self.branch = branch
        self.trunk = trunk
        self.fc = nn.Linear(latent_features,out_features,bias = False)
        

    def forward(self,y,u):
        return self.fc(self.trunk(y)*self.branch(u))
# class NetworkA(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim):
#         super(NetworkA, self).__init__()
#         self.layers = nn.ModuleList()
#         dims = [input_dim] + hidden_dims + [output_dim]
        
#         # Store the structure for assigning weights later
#         self.shapes = []
        
#         for i in range(len(dims) - 1):
#             layer = nn.Linear(dims[i], dims[i + 1], bias=False)
#             self.layers.append(layer)
#             self.shapes.append((dims[i], dims[i + 1]))  # Store layer shapes for reshaping weights

#     def forward(self, x, weights):
#         # Set weights dynamically
#         index = 0
#         for layer, shape in zip(self.layers, self.shapes):
#             # Extract the weight slice for the current layer and reshape
#             layer_weight = weights[index:index + shape[0] * shape[1]].view(shape)
#             layer.weight = nn.Parameter(layer_weight)  # Dynamically assign weights
#             x = F.relu(layer(x))  # Apply layer with ReLU activation
#             index += shape[0] * shape[1]
#         return x

# # Define Network B to approximate weights of Network A
# class NetworkB(nn.Module):
#     def __init__(self, input_dim, num_weights):
#         super(NetworkB, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, num_weights)  # Output size matches total parameters of Network A

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         weights = self.fc3(x)  # Output shape: [batch_size, num_weights]
#         return weights
    
if __name__ == '__main__':

    # branch1 = MLP(100,100,75,2)
    # branch2 = BaseDNN(100,100,[75,75,75],F.tanh)
    # print('parameters1:', sum(p.numel() for p in branch1.parameters()))
    # print('parameters2:', sum(p.numel() for p in branch2.parameters()))
    
    # for name, p in branch1.named_parameters():
    #     print(name, p.shape)
        
    # for name, p in branch2.named_parameters():
    #     print(name, p.shape)
    

    # branch = BaseDNN(input_dim=100, output_dim=75, hidden_layers=[75,75,75,75,75], activation_fn=F.tanh)
    # trunk = BaseDNN(input_dim=1, output_dim=75, hidden_layers=[75,75,75,75,75], activation_fn=F.tanh)
    
    branch = MLP(100,75,75,4)
    trunk = MLP(1,75,75,4)
    
    Onet = DeepONet(1, branch, trunk)
    # Onet = DeepONet(1, branch = MLP(100,75,75,4), trunk = MLP(1,75,75,4))
    
    print('parameters:', sum(p.numel() for p in Onet.parameters()))
    
    ys = torch.randn(500, 100,1)
    us = torch.randn(500, 1, 100)
    Guys = torch.randn(500, 100, 1)
    
    outputs = Onet(ys, us)
    print(branch(us).shape)
    print(trunk(ys).shape)
    print(outputs.shape)
        
    # mlp = MLP(layers=[64, 128, 64, 10], activation=F.relu)
    # m=10
    # branch_layers = [m, 50, 50]
    # trunk_layers =  [2, 50, 50]
    
    # deeponet = DeepONet(branch_layers, trunk_layers)

    # # Run a forward pass with some sample data
    # u_samples = torch.randn(100, m)
    # y_points = torch.randn(100, 2)
    
    # outputs = deeponet(u_samples, y_points)
    # print(outputs.shape)    
    
    # params_count = sum(p.numel() for p in deeponet.parameters())
    # print(f"Total number of parameters: {params_count}")
    # for name, p in deeponet.named_parameters():
    #     print(name, p.shape)
    

    # # Example Usage
    # input_dim_A = 10  # Input dimension for Network A
    # hidden_dims_A = [20, 30]  # Hidden dimensions for Network A
    # output_dim_A = 5  # Output dimension for Network A

    # # Calculate total number of weights needed for Network A
    # dummy_A = NetworkA(input_dim_A, hidden_dims_A, output_dim_A)
    # num_weights_A = sum(p.numel() for p in dummy_A.parameters())

    # # Instantiate Network A and Network B
    # network_a = NetworkA(input_dim_A, hidden_dims_A, output_dim_A)
    # network_b = NetworkB(input_dim=15, num_weights=num_weights_A)  # Input to B can be any context vector

    # # Forward pass example
    # context_vector = torch.randn(1, 15)  # Example input for Network B
    # weights_for_A = network_b(context_vector).squeeze()  # Output weights for Network A
    # input_to_A = torch.randn(1, input_dim_A)  # Example input for Network A

    # # Forward pass through Network A using the weights from Network B
    # output_from_A = network_a(input_to_A, weights_for_A)

    # # Compute gradients with respect to parameters of Network B
    # output_from_A.backward(torch.ones_like(output_from_A))