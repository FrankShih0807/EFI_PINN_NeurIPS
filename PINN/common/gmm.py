import torch
import torch.distributions as dist

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
    device = 'mps'
    
    mu = 0
    sigma = 1
    normal_dist = torch.distributions.Normal(mu, sigma)
    log_prob = torch.distributions.Normal(mu, sigma).log_prob
    
    x = torch.randn(3,4).to(device)
    print(log_prob(x))