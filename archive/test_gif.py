import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Simulate training loss
epochs = 100
losses = [np.exp(-0.05 * epoch) + 0.1 * np.random.rand() for epoch in range(epochs)]

# Create a directory to store temporary images (optional cleanup step)
temp_dir = "temp_frames"
os.makedirs(temp_dir, exist_ok=True)

# Create individual frames for the GIF
frames = []
for epoch in range(1, epochs + 1):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epoch + 1), losses[:epoch], label="Training Loss", color="blue")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1.2)  # Adjust y-axis limit for consistent scaling
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the frame as an image in memory
    frame_path = os.path.join(temp_dir, f"frame_{epoch}.png")
    plt.savefig(frame_path)
    frames.append(Image.open(frame_path))
    plt.close()

# Save all frames as a GIF
frames[0].save(
    "training_loss.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,  # Duration in milliseconds per frame
    loop=0  # Loop forever
)

# Optional: Cleanup temporary files
for frame_path in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, frame_path))
os.rmdir(temp_dir)

print("GIF saved as 'training_loss.gif'")