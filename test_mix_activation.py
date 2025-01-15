import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the model
class MixedActivationNet(nn.Module):
    def __init__(self, input_dim):
        super(MixedActivationNet, self).__init__()
        self.relu_branch = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.softplus_branch = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Softplus(),
            nn.Linear(16, 8),
            nn.Softplus()
        )
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x):
        relu_out = self.relu_branch(x)
        softplus_out = self.softplus_branch(x)
        combined = torch.cat((relu_out, softplus_out), dim=1)
        return self.output_layer(combined)

# Generate data
x = torch.linspace(-10, 10, 100).unsqueeze(1)
y = torch.where(x < 0, -5*x, x**2)

# Initialize model, criterion, and optimizer
model = MixedActivationNet(input_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(2000):
    optimizer.zero_grad()
    predictions = model(x)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

# Get predictions
predicted_y = model(x).detach()

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(x.numpy(), y.numpy(), label='Target Function', color='blue', linewidth=2)
plt.plot(x.numpy(), predicted_y.numpy(), label='DNN Approximation', color='orange', linestyle='--')
plt.title('Function Approximation with Mixed Activation DNN')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()