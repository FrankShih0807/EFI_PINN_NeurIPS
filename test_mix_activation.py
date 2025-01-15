import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the model
class MixedActivationNet(nn.Module):
    def __init__(self, input_dim):
        super(MixedActivationNet, self).__init__()
        self.relu_branch = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU()
        )
        self.softplus_branch = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.Softplus(),
            nn.Linear(50, 25),
            nn.Softplus()
        )
        self.output_layer = nn.Linear(50, 1)

    def forward(self, x):
        relu_out = self.relu_branch(x)
        softplus_out = self.softplus_branch(x)
        combined = torch.cat((relu_out, softplus_out), dim=1)
        return self.output_layer(combined)


def physics_law(s, t2m)->torch.Tensor:
    r = 0.05
    sigma = 0.5
    K = 0.5
    norm_dist = torch.distributions.Normal(0, 1)
    
    s = torch.as_tensor(s)
    t2m = torch.as_tensor(t2m)
    d1 = (torch.log(s/K) + (r + sigma**2/2) * (t2m)) / (sigma * torch.sqrt(t2m))
    d2 = d1 - sigma * torch.sqrt(t2m)
    Nd1 = norm_dist.cdf(d1)
    Nd2 = norm_dist.cdf(d2)
    V = s * Nd1 - K * torch.exp(-r * (t2m)) * Nd2
    return V
# Generate data
s = torch.linspace(0, 1, 100)
t = torch.linspace(0, 1, 100)
S, T = torch.meshgrid(s, t, indexing='ij')
X = torch.stack([S.reshape(-1), T.reshape(-1)], dim=1)

C = physics_law(S, T)

# fig = plt.figure()

# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(S.numpy(), T.numpy(), C.numpy(), cmap='plasma')
# # ax.contourf(S.numpy(), T.numpy(), C.numpy(), zdir='z', offset=-20, cmap='plasma')

# # fig.colorbar(im, shrink=0.5, aspect=5, pad=0.07)
# ax.set_xlabel('Stock Price')
# ax.set_ylabel('Time to Maturity')
# ax.set_zlabel('Option Price')
# ax.view_init(elev=15, azim=-125)
# plt.tight_layout()
# plt.show()

# Initialize model, criterion, and optimizer
model = MixedActivationNet(input_dim=2)
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5000):
    optimizer.zero_grad()
    predictions = model(X)
    # print(predictions.shape, C.shape)   
    loss = criterion(predictions.flatten(), C.flatten())
    loss.backward()
    optimizer.step()

# Get predictions
predicted_y = model(X).detach().reshape(100,100)

# Plot results
# print(S.shape, T.shape, predicted_y.shape)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(S.numpy(), T.numpy(), predicted_y.numpy(), cmap='plasma')

ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Option Price')
ax.view_init(elev=15, azim=-125)
plt.tight_layout()
plt.show()