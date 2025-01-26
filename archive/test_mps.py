import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the network for a few epochs and measure time
def train_model(device, epochs=5000):
    model = SimpleNet().to(device)
    print('parameters:', sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Random data to simulate a training dataset
    inputs = torch.randn(2048, 784).to(device)  # Batch size 2048, input size 784
    labels = torch.randint(0, 10, (2048,)).to(device)  # 10 classes

    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    
    return end_time - start_time

# Run the training on CPU and MPS
if torch.backends.mps.is_available():
    print("Testing on CPU:")
    cpu_time = train_model("cpu")
    print(f"CPU training time: {cpu_time:.2f} seconds")

    print("\nTesting on MPS:")
    mps_time = train_model("mps")
    print(f"MPS training time: {mps_time:.2f} seconds")

    speedup = cpu_time / mps_time
    print(f"\nMPS speedup over CPU: {speedup:.2f}x")
else:
    print("MPS is not available on this device.")