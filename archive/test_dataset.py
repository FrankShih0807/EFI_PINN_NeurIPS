import torch
from torch.utils.data import Dataset

class DynamicPINNDataset(Dataset):
    def __init__(self):
        """
        Initializes an empty dataset for PINN with lists for dynamic data addition.
        """
        # Initialize lists to store data for each category
        self.X_data = []
        self.y_data = []
        self.categories = []
        self.noise_indicators = []

    def add_data(self, X, y, category, noise_indicator):
        """
        Adds new data to the dataset.
        
        Args:
            X (torch.Tensor): The input tensor for the new data points.
            y (torch.Tensor): The output tensor for the new data points.
            category (int): An integer indicating the category (0: solution, 1: differential, 2: boundary).
            noise_indicator (torch.Tensor): A binary tensor indicating noise in y.
        """
        # Append new data to the respective lists
        self.X_data.append(X)
        self.y_data.append(y)
        self.categories.append(category)
        self.noise_indicators.append(noise_indicator)

    def __len__(self):
        return sum(len(x) for x in self.X_data)

    def __getitem__(self, idx):
        return {
            "X": self.X_data[idx],
            "y": self.y_data[idx],
            "category": self.categories[idx],
            "noise_indicator": self.noise_indicators[idx]
        }

# Example usage
dataset = DynamicPINNDataset()

# Add solution points
solution_X = torch.rand(10, 2)
solution_y = torch.sin(solution_X[:, 0]) * torch.cos(solution_X[:, 1]).unsqueeze(1)
solution_noise = 0.1
dataset.add_data(solution_X, solution_y, category='solution', noise_indicator=solution_noise)

# Add differential points
differential_X = torch.rand(20, 2)
differential_y = torch.cos(differential_X[:, 0]) * -torch.sin(differential_X[:, 1]).unsqueeze(1)
differential_noise = 0.5
dataset.add_data(differential_X, differential_y, category='boundary', noise_indicator=differential_noise)

# Add boundary points
boundary_X = torch.rand(50, 2)
boundary_y = torch.zeros(50, 1)
boundary_noise = 0.3
dataset.add_data(boundary_X, boundary_y, category='differential', noise_indicator=boundary_noise)

# Access data
print("Total dataset size:", len(dataset))
print("Sample data point:", dataset[0]['X'])
# print(dataset.y_data[0])