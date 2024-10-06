import torch
import torch.nn as nn
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
    

class SparseDNN(BaseNetwork):
    def __init__(self, input_size, output_size, hidden_layers=[32, 32], activation_fn=nn.ReLU, prior_sd=0.01, sparse_sd=0.001 , sparsity=0.5):
        super().__init__(input_size, output_size, hidden_layers, activation_fn)
        self.sparsity = sparsity
        self.prior_sd = prior_sd
        self.sparse_sd = sparse_sd
        
        self.gmm = GaussianMixtureModel(prior_sd, sparse_sd)

    
    def mixture_gaussian_prior(self, sparsity=None):
        if sparsity is None:
            sparsity = self.sparsity
        log_prior = 0
        for p in self.parameters():
            print(p.shape)
            # log_prior += self.gmm.log_prob(p, sparsity).sum()
        # return log_prior

    
class GaussianMixtureModel:
    def __init__(self, prior_sd, sparse_sd):
        """
        Initialize the GMM with means, standard deviations, and mixture weights for each component.
        :param means: A tensor of shape (2,) containing the means of the two Gaussians.
        :param stds: A tensor of shape (2,) containing the standard deviations of the two Gaussians.
        :param mixture_weights: A tensor of shape (2,) containing the mixture weights of the two Gaussians.
        """
        self.means = torch.tensor([0.0, 0.0])  # Means of the two Gaussians
        self.stds = torch.tensor([prior_sd, sparse_sd])   # Standard deviations of the two Gaussians
        self.components = dist.Normal(self.means, self.stds)


    def log_prob(self, x, sparsity):
        """
        Calculate the log probability of data points x under the GMM.
        :param x: A tensor of data points.
        :return: A tensor of log probabilities.
        """
        # Calculate log probabilities from each component for each data point
        log_probs = self.components.log_prob(x.unsqueeze(1))  # Shape will be [N, num_components]
        # Weight log probabilities by the log of mixture weights
        self.mixture_weights = torch.tensor([sparsity, 1-sparsity])  # Ensure weights sum to 1
        log_weighted_probs = log_probs + torch.log(self.mixture_weights)
        # Log-sum-exp trick for numerical stability: log(sum(exp(log_probs)))
        log_sum_exp = torch.logsumexp(log_weighted_probs, dim=1)
        return log_sum_exp

if __name__ == '__main__':
    
    model = SparseDNN(10, 20, hidden_layers=[32, 32], activation_fn=nn.Tanh, prior_sd=0.01, sparse_sd=0.001 , sparsity=0.5)
    print(model.parameter_size)
    

    model.mixture_gaussian_prior()