import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as dist


class BaseNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[32, 32], activation_fn=nn.ReLU):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        
        self.activation_fn= activation_fn
        
        self.layers = nn.ModuleList()

        # Add the first layer (input layer)
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.layers.append(self.activation_fn())

        # Add hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(self.activation_fn())

        # Add the output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))

        self.parameter_size = sum([p.numel() for p in self.parameters()])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

    
class EncoderNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, activation_fn=F.softplus):
        super(EncoderNetwork, self).__init__()
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
        self.layers.append(nn.Linear(self.hidden_layers[-1], self.output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        return x
    

    
    

if __name__ == '__main__':
    
    pass
    
    x = torch.randn(3, 4, 5)
    print(x.shape)
    print(x.numel())