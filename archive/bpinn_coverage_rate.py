import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Data
lam_diff = np.array([1.0, 3.16, 10.0, 31.6, 100.0])
lam_sol = np.array([1.0, 3.16, 10.0, 31.6, 100.0])
cr = np.array([
    [1.0, 0.99, 0.9394, 0.4128, 0.1383], 
    [1.0, 0.95, 0.9496, 0.6306, 0.1267],
    [0.93, 0.96, 0.95, 0.8546, 0.1784],
    [0.8295, 0.8108, 0.8, 0.8849, 0.7152],
    [0.1083, 0.6952, 0.8543, 0.8002, 0.7127]
])
cr = cr.transpose()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    cr, 
    annot=True, 
    fmt=".4f", 
    xticklabels=lam_diff, # Set the x-axis labels to lam_diff
    yticklabels=lam_sol, # Set the y-axis labels to lam_sol
    cmap="coolwarm"
)

plt.gca().invert_yaxis()  # Reverse the y-axis to show lam_sol increasing upward
plt.title("Coverage Rate", fontsize=15)
plt.xlabel(r"$\lambda_{\text{diff}}$", fontsize=15)
plt.ylabel(r"$\lambda_{\text{sol}}$", fontsize=15)
# plt.savefig("Coverage_rate_map.png")
plt.show()