import torch

# Example: p vectors of size (n, 1)
n = 4  # Number of rows
p = 3  # Number of vectors
vectors = [torch.randn(n, 1, requires_grad=True) for _ in range(p)]  # Create p autograd-enabled vectors


optimizer = torch.optim.SGD(vectors, lr=0.1)  # Example optimizer
# Generate n x p tensors for each vector
tensors = []
for i, vec in enumerate(vectors):
    # Create a zero tensor
    padded_tensor = torch.zeros(n, p, requires_grad=True)
    
    # Use torch.cat to construct the tensor without in-place operations
    padded_tensor = padded_tensor.clone()  # Avoid modifying a view
    padded_tensor[:, i:i+1] = vec  # Assign the vector to the i-th column (no in-place error)

    tensors.append(padded_tensor)

# Result: List of n x p tensors, each with the i-th column containing the original vector
for v in vectors:
    print(v)

optimizer.zero_grad()
# Example backward pass
loss = sum(tensor.sum() for tensor in tensors)  # Dummy loss function
loss.backward()
optimizer.step()

for v in vectors:
    print(v)    
# Check gradients for the original vectors
# for i, vec in enumerate(vectors):
#     print(f"Gradient for vector {i}:\n{vec.grad}")