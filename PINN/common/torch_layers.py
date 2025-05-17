import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from collections import defaultdict
from PINN.common.utils import get_activation_fn
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from PINN.common.losses import gmm_loss
from torch.func import functional_call
from collections import OrderedDict 
    
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)  # Or use xavier_normal_
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class BaseDNN(nn.Module):
    def __init__(self, input_dim, hidden_layers=None, output_dim=None, activation_fn='relu'):
        super(BaseDNN, self).__init__()
        # Define the initial layers
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation_fn = get_activation_fn(activation_fn)
        
        self.layers = nn.ModuleList()
        
        # Initialize layers
        if hidden_layers is not None:
            
            if len(hidden_layers) > 0:
                self.layers.append(nn.Linear(input_dim, self.hidden_layers[0]))
                # Add hidden layers
                for i in range(1, len(self.hidden_layers)):
                    self.layers.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
                # Add the output layer
                if self.output_dim is not None:
                    self.layers.append(nn.Linear(self.hidden_layers[-1], self.output_dim))
            else:
                if self.output_dim is not None:
                    self.layers.append(nn.Linear(self.input_dim, self.output_dim))
                else:
                    raise ValueError("Either hidden_layers or output_dim must be specified.")

    def forward(self, x):
        if self.hidden_layers is not None:
            for layer in self.layers[:-1]:
                x = layer(x)
                x = self.activation_fn(x)
            x = self.layers[-1](x)
        return x
    

class HyperEncoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_layers, 
                 output_dim, 
                 pe_dim,
                 param_shapes,
                 activation_fn='relu', 
                 neck_layer_activation='identity',
                 device='cpu'
                 ):
        super().__init__()
        # Define the initial layers
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.pe_dim = pe_dim
        self.param_shapes = param_shapes
        self.activation_fn = get_activation_fn(activation_fn)
        self.neck_layer_activation = get_activation_fn(neck_layer_activation)
        self.device = device
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, self.hidden_layers[0]))
        # Add hidden layers
        for i in range(1, len(self.hidden_layers)):
            self.layers.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        # Add the output layer
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.output_dim)

    def forward(self, X, Y, Z):
        x = torch.cat([X, Y, Z], dim=1)  # shape: [B, D]
        x = x.to(self.device)
        batch_size = x.shape[0]
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        x = self.neck_layer_activation(x)
        theta_all = self.output_layer(x)
        theta_mean = theta_all.mean(dim=0)
        

        theta_loss = ((theta_all - theta_mean.repeat(batch_size, 1))**2).sum()
        param_dict, log_sd, pe_variables = self.split_encoder_output(theta_mean)
        
        return theta_loss, theta_mean, param_dict, log_sd, pe_variables
    
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
    


class TransferEFI(nn.Module):
    def __init__(self,
                 input_dim=1,
                 output_dim=1,
                 latent_Z_dim=1,
                 feature_extractor_layers=None,
                 feature_dim=None,
                 efi_hidden_layers=[50, 50, 50],
                 activation_fn='relu',
                 pe_dim=0,
                 encoder_hidden_layers=[32, 32, 16],
                 encoder_activation='relu',
                 neck_layer_activation='identity',
                 positive_output=False,
                 sd_known=True,
                 device='cpu'):
        super().__init__()
        
        self.device = device
        self.pe_dim = pe_dim
        self.positive_output = positive_output
        self.sd_known = sd_known
        if feature_dim is not None:
            self.feature_dim = feature_dim
        else:
            self.feature_dim = input_dim
            
        if feature_extractor_layers is not None and isinstance(feature_extractor_layers, list):
            self.feature_extractor = BaseDNN(
            input_dim=input_dim,
            hidden_layers=feature_extractor_layers,
            output_dim=self.feature_dim,
            activation_fn=activation_fn
            )
        else:
            self.feature_extractor = nn.Identity()
        
        self.efi_layers = BaseDNN(
            input_dim=self.feature_dim,
            hidden_layers=efi_hidden_layers,
            output_dim=output_dim,
            activation_fn=activation_fn
        )
        self.param_shapes = {
            name: param.shape for name, param in self.efi_layers.named_parameters()
        }
        self.total_params = sum(p.numel() for p in self.efi_layers.parameters())
        
        self.hyper = HyperEncoder(
            input_dim=self.feature_dim + output_dim + latent_Z_dim,
            hidden_layers=encoder_hidden_layers,
            output_dim=self.total_params + pe_dim + 1,
            pe_dim=pe_dim,
            param_shapes=self.param_shapes,
            activation_fn=encoder_activation,
            neck_layer_activation=neck_layer_activation,
            device=device
        )
        self.feature_extractor.to(self.device)
        self.efi_layers.to(self.device)
        self.hyper.to(self.device)

        self.param_dict = None
        self.pe_variables = None
        self.log_sd = None
        
        self.efi_on = False
        
    def encode_efi_params(self, X, Y, Z):
        X = self.feature_extractor(X)
        theta_loss, theta_mean, param_dict, log_sd, pe_variables = self.hyper(X, Y, Z)  # shape: [total_params + pe_dim]
        self.param_dict = param_dict
        self.log_sd = log_sd
        self.pe_variables = pe_variables
        self.theta_mean = theta_mean
        return theta_loss

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)

        if self.efi_on == False:
            x = self.feature_extractor(x)
            out = self.efi_layers(x)
            # print('efi off')
        else:
            if self.param_dict is None:
                raise RuntimeError("Must call encode_efi_params before forward")
            x = self.feature_extractor(x)
            out = functional_call(self.efi_layers, self.param_dict, (x,))
            # print('efi on')
        # out = functional_call(self.efi_layers, self.param_dict, (x,))
        if self.positive_output:
            out = F.softplus(out)
            # out = torch.exp(out)
        return out
    
    def pretrain(self, X, Y, Z):
        optimizer = torch.optim.Adam(self.hyper.parameters(), lr=3e-4)
        theta = utils.parameters_to_vector(self.efi_layers.parameters())
        for _ in range(1000):
            optimizer.zero_grad()
            theta_loss = self.encode_efi_params(X, Y, Z)

            loss = F.mse_loss(self.theta_mean[:(-1-self.pe_dim)], theta) + theta_loss
            loss.backward()
            optimizer.step()
        
        self.efi_on = True
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.efi_layers.parameters():
            p.requires_grad = False
            
        print('pretrain done and efi on')
            
            
class DropoutDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, activation_fn=F.relu, dropout_rate=0.01, positive_output=False, sd_known=True):
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
        self.positive_output = positive_output
        # Initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, self.hidden_layers[0]))  # Input layer
        self.sd_known = sd_known

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
        if self.positive_output:
            x = torch.exp(x)
        return x

class BottleneckHypernet(nn.Module):
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
        x = self.output_layer(x)
        return x

    
        
class EFI_Net_PE(nn.Module):
    def __init__(self, 
                 input_dim=1, 
                 output_dim=1, 
                 latent_Z_dim=1,
                 hidden_layers=[15, 15], 
                 activation_fn='relu', 
                 sd_known=True,
                 pe_dim=0,
                 encoder_hidden_layers=None,
                 encoder_activation='relu',
                 neck_layer_activation='identity',
                 prior_sd=0.1, 
                 sparse_sd=0.01,
                 sparsity=1.0,
                 positive_output=False,
                 device='cpu'
                 ):
        super(EFI_Net_PE, self).__init__()
        
        self.device = device
        # EFI Net Info
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_Z_dim = latent_Z_dim
        self.hidden_layers = hidden_layers
        self.activation_fn = get_activation_fn(activation_fn)
        self.sd_known = sd_known

        # Encoder Net Info
        self.encoder_input_dim = self.input_dim + self.output_dim + self.latent_Z_dim
        self.encoder_activation = get_activation_fn(encoder_activation)
        self.neck_layer_activation = get_activation_fn(neck_layer_activation)
        
        # sparse prior settings
        self.sparsity = sparsity
        self.prior_sd = prior_sd
        self.sparse_sd = sparse_sd
        
        self.positive_output = positive_output
        
        # parameter estimation settings
        if self.sd_known:
            self.pe_dim = pe_dim
        else:
            self.pe_dim = pe_dim + 1
        # if self.pe_dim > 0:
        #     self.pe_variables = nn.Parameter(torch.randn(self.pe_dim), requires_grad=True, device=self.device)
        
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
        
        # self.encoder = BaseDNN(input_dim=self.encoder_input_dim, hidden_layers=encoder_hidden_layers, output_dim=self.n_parameters+self.pe_dim , activation_fn=self.encoder_activation).to(self.device)
        self.encoder = BottleneckHypernet(input_dim=self.encoder_input_dim, hidden_layers=encoder_hidden_layers, output_dim=self.n_parameters+self.pe_dim , activation_fn=self.encoder_activation, neck_layer_activation=self.neck_layer_activation).to(self.device)
        # print(self.encoder_input_dim, self.n_parameters, self.pe_dim, )

    def split_encoder_output(self, theta):
        '''Split encoder output into network layer shapes

        Args:
            theta (tensor): encoder ouput mean

        Returns:
            weight_tensors, bias_tensors: tensors
        '''
        if self.sd_known:
            theta_weight, theta_bias, theta_pe = torch.split(theta, [sum(self.nn_numel['weight']), sum(self.nn_numel['bias']), self.pe_dim] , dim=-1)
        else:
            theta_weight, theta_bias, theta_pe, log_sd = torch.split(theta, [sum(self.nn_numel['weight']), sum(self.nn_numel['bias']), self.pe_dim-1, 1], dim=-1)
            self.log_sd = log_sd.view(-1)
            
        # theta_weight, theta_bias, theta_pe = torch.split(theta, [sum(self.nn_numel['weight']), sum(self.nn_numel['bias']), self.pe_dim ] , dim=-1)
        theta_weight_split = torch.split(theta_weight, self.nn_numel['weight'], dim=-1)
        theta_bias_split = torch.split(theta_bias, self.nn_numel['bias'], dim=-1)
        
        weight_tensors = [theta_weight_split[i].view(*shape).to(self.device) for i, shape in enumerate(self.nn_shape['weight'])]
        bias_tensors = [theta_bias_split[i].view(*shape).to(self.device) for i, shape in enumerate(self.nn_shape['bias'])]
        
        return weight_tensors, bias_tensors, theta_pe
            
    
    def forward(self, x):
        x = x.to(self.device)
        for i in range(self.n_layers-1):
            x = self.activation_fn(F.linear(x, self.weight_tensors[i], self.bias_tensors[i]))
        x = F.linear(x, self.weight_tensors[-1], self.bias_tensors[-1])
        if self.positive_output:
            # x = torch.exp(x)
            x = F.softplus(x)
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
        self.weight_tensors, self.bias_tensors, self.pe_variables = self.split_encoder_output(theta_bar)
        return theta_loss
        
    def gmm_prior_loss(self):
        loss = 0
        for p in self.parameters():
            loss += gmm_loss(p, self.prior_sd, self.sparse_sd, self.sparsity).sum()
        return loss



    
class BayesianPINNNet(nn.Module):
    def __init__(self, sigma_diff, sigma_sol, physics_model, num_bd, input_dim, output_dim, hidden_layers):
        super(BayesianPINNNet, self).__init__()

        self.fnn = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.Tanh(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.Tanh(),
            nn.Linear(hidden_layers[1], output_dim), 
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
    
class BayesianNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_layers = [50, 50], activation_fn=torch.tanh, sd_known=True, positive_output=False):
        super(BayesianNet, self).__init__()
        self.hidden_layers = hidden_layers
        # self.layer_list = []
        self.activation_fn = activation_fn
        # self.pe_variables = nn.Parameter(torch.randn(1), requires_grad=True)
        self.sd_known = sd_known
        self.positive_output = positive_output

        self.l1 = nn.Linear(input_dim, hidden_layers[0])
        self.l2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.l3 = nn.Linear(hidden_layers[1], output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.activation_fn(x)
        x = self.l2(x)
        x = self.activation_fn(x)
        x = self.l3(x)
        if self.positive_output:
            x = torch.exp(x)
        return x

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
    
class MixedActivationNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MixedActivationNet, self).__init__()
        self.relu_branch = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU()
        )
        self.softplus_branch = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.Softplus(beta=5),
            nn.Linear(50, 25),
            nn.Softplus(beta=5)
        )
        self.output_layer = nn.Linear(50, output_dim)

    def forward(self, x):
        relu_out = self.relu_branch(x)
        softplus_out = self.softplus_branch(x)
        combined = torch.cat((relu_out, softplus_out), dim=1)
        return self.output_layer(combined)

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
    