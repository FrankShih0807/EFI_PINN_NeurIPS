import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the HyperNetwork, which outputs weights for another network
class HyperNetwork(nn.Module):
    def __init__(self, input_dim, target_layer_dim):
        super(HyperNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, target_layer_dim * target_layer_dim)  # Output flattened weights
        self.fc3 = nn.Linear(128, target_layer_dim)  # Output biases

    def forward(self, x):
        x = F.relu(self.fc1(x))
        weights = self.fc2(x)  # Output weights for target layer, shape: (target_layer_dim * target_layer_dim)
        biases = self.fc3(x)   # Output biases for target layer, shape: (target_layer_dim,)
        return weights, biases

# Define the TargetNetwork with an arbitrary structure
class TargetNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hyper_dim):
        super(TargetNetwork, self).__init__()
        self.hypernetwork = HyperNetwork(hyper_dim, output_dim)  # Hypernetwork to generate weights
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x, hyper_input):
        # Generate weights and biases using the hypernetwork
        weights, biases = self.hypernetwork(hyper_input)
        
        # Reshape weights to match input and output dimensions of the target layer
        weights = weights.view(self.output_dim, self.input_dim)
        
        # Apply the fully connected layer using the generated weights and biases
        x = F.linear(x, weights, biases)
        return x

# # Example usage
# # Define input dimensions for target and hyper networks
# input_dim = 64         # Dimension of input to TargetNetwork
# output_dim = 32        # Output dimension of the dynamically generated layer
# hyper_dim = 10         # Input dimension for HyperNetwork (context for generating weights)

# # Initialize target network
# target_network = TargetNetwork(input_dim=input_dim, output_dim=output_dim, hyper_dim=hyper_dim)

# # Example inputs
# target_input = torch.randn(1, input_dim)        # Input to TargetNetwork
# hyper_input = torch.randn(1, hyper_dim)         # Input to HyperNetwork

# # Forward pass
# output = target_network(target_input, hyper_input)
# print("Output shape:", output.shape)  # Should be [1, output_dim]

# # Backward pass to verify autograd works
# output.mean().backward()

if __name__ == '__main__':
    
    net1 = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU()
    )
    
    print(net1[0].weight.shape, net1[0].bias.shape)