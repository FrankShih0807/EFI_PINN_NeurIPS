import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def get_activation_fn(activation):
    # Define a dictionary mapping activation names to PyTorch classes
    activation_dict = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'softplus': nn.Softplus,
        'softmax': nn.Softmax,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
    }
    
    # If the activation is already an instance of a PyTorch activation, return it directly
    if isinstance(activation, nn.Module):
        return activation
    
    # If activation is a string, parse it
    elif isinstance(activation, str):
        # Extract function name and optional parameters
        if "(" in activation and ")" in activation:
            # Example: 'softplus(beta=2)' -> 'softplus', {'beta': 2}
            func_name, params_str = activation.split("(", 1)
            func_name = func_name.strip().lower()
            params_str = params_str.rstrip(")")
            
            # Parse parameters into a dictionary
            params = {}
            for param in params_str.split(","):
                key, value = param.split("=")
                params[key.strip()] = eval(value.strip())
            
            # Check if the function name is in the dictionary
            if func_name in activation_dict:
                return activation_dict[func_name](**params)
            else:
                raise ValueError(f"Unsupported activation function: '{func_name}'")
        
        else:
            # If no parameters are provided, simply use the activation name
            func_name = activation.lower()
            if func_name in activation_dict:
                return activation_dict[func_name]()
            else:
                raise ValueError(f"Unsupported activation function: '{func_name}'")
    
    else:
        raise TypeError("Activation must be either a string or a torch.nn activation class.")



class PINNDataset(Dataset):
    def __init__(self, device='cpu'):
        """
        Initializes an empty dataset for PINN with lists for dynamic data addition.
        """
        # Initialize lists to store data for each category
        self.X_data = []
        self.y_data = []
        self.true_y = []
        self.categories = []
        self.noise_sd = []
        self.device = device

    def add_data(self, X, y, true_y, category, noise_sd):
        """
        Adds new data to the dataset.
        
        Args:
            X (torch.Tensor): The input tensor for the new data points.
            y (torch.Tensor): The output tensor for the new data points.
            category (int): An integer indicating the category (0: solution, 1: differential, 2: boundary).
            noise_sd (torch.Tensor): A binary tensor indicating noise in y.
        """
        # Append new data to the respective lists
        self.X_data.append(X.to(self.device))
        self.y_data.append(y.to(self.device))
        self.true_y.append(true_y.to(self.device)) 
        self.categories.append(category)
        self.noise_sd.append(noise_sd)
        
    def _prepare_data(self):
        """Sets up data, ensuring X.requires_grad=True for differential category."""
        for d in self:
            # Move X and y to the specified device
            d['X'] = d['X'].to(self.device)
            d['y'] = d['y'].to(self.device)
            d['true_y'] = d['true_y'].to(self.device)

            # Set requires_grad=True for differential category (e.g., category == 'differential')
            if d['category'] == 'differential':
                d['X'].requires_grad_()
                
    def __len__(self):
        return sum(len(x) for x in self.X_data)

    def __getitem__(self, idx):
        return {
            "X": self.X_data[idx],
            "y": self.y_data[idx],
            "true_y": self.true_y[idx],
            "category": self.categories[idx],
            "noise_sd": self.noise_sd[idx]
        }
    def copy(self):
        
        """Creates a copy of the dataset with the same data and device."""
        # Create a new instance of DynamicPINNDataset on the same device
        dataset_copy = PINNDataset(device=self.device)

        # Copy data from the current dataset to the new instance
        dataset_copy.X_data = [x.clone() for x in self.X_data] 
        dataset_copy.y_data = [y.clone() for y in self.y_data] 
        dataset_copy.true_y = [y.clone() for y in self.true_y]
        dataset_copy.categories = [c for c in self.categories] 
        dataset_copy.noise_sd = [n for n in self.noise_sd] 
        dataset_copy._prepare_data()
        
        return dataset_copy

if __name__ == "__main__":
    # # Example usage
    # dataset = PINNDataset(device='mps')

    # # Add solution points
    # solution_X = torch.rand(10, 2)
    # solution_y = torch.sin(solution_X[:, 0]) * torch.cos(solution_X[:, 1]).unsqueeze(1)
    # solution_noise = 0.1
    # dataset.add_data(solution_X, solution_y, category='solution', noise_indicator=solution_noise)

    # # Add differential points
    # differential_X = torch.rand(20, 2)
    # differential_y = torch.cos(differential_X[:, 0]) * -torch.sin(differential_X[:, 1]).unsqueeze(1)
    # differential_noise = 0.5
    # dataset.add_data(differential_X, differential_y, category='boundary', noise_indicator=differential_noise)

    # # Add boundary points
    # boundary_X = torch.rand(50, 2)
    # boundary_y = torch.zeros(50, 1)
    # boundary_noise = 0.3
    # dataset.add_data(boundary_X, boundary_y, category='differential', noise_indicator=boundary_noise)

    # # Access data
    # print("Total dataset size:", len(dataset))
    # print("Sample data point:", dataset[0]['X'])
    # # print(dataset.y_data[0])
    
    # dataset_copy = dataset.copy()
    # dataset_copy.add_data(torch.rand(10, 2), torch.rand(10, 1), category='solution', noise_indicator=0.1)
    # print(len(dataset_copy))
    
    # # print(dataset[2]['X'].requires_grad)
    
    # for d in dataset_copy:
    #     print(d['X'].requires_grad)
    
    activation_fn = nn.Identity()
    activation = get_activation_fn(activation_fn)
    print(activation(torch.tensor([-1.0, 0.0, 1.0]))) # tensor([0., 0., 1.])