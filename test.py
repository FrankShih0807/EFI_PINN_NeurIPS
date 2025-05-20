import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Example data
x_data = np.linspace(0, 2, 20)
x_pred = np.linspace(0, 2, 200)

def gaussian(x, amp, sigma):
    return 1500 - amp * np.exp(-(x - 1)**2 / (2 * sigma**2))

# List of datasets
data_sets = [
    (gaussian(x_data, 1000, 0.1), gaussian(x_pred, 1000, 0.1), 'x', 'tab:blue', '0.0 days'),
    (gaussian(x_data, 800, 0.12), gaussian(x_pred, 800, 0.12), 'o', 'tab:orange', '0.5 days'),
    (gaussian(x_data, 600, 0.14), gaussian(x_pred, 600, 0.14), 's', 'tab:green', '1.0 days'),
]

# Plot lines and markers separately
for y_data, y_pred, marker, color, label in data_sets:
    plt.plot(x_pred, y_pred, linestyle='-', color=color)                     # Line from model
    plt.plot(x_data, y_data, linestyle='None', marker=marker, color=color)  # Markers from data

# Construct custom legend handles with both line and marker
legend_elements = [
    Line2D([0], [0], linestyle='-', marker=marker, color=color, label=label)
    for _, _, marker, color, label in data_sets
]
for _, _, marker, color, label in data_sets:
    print(f"Marker: {marker}, Color: {color}, Label: {label}")
print(legend_elements)

plt.xlabel('Position (mm)')
plt.ylabel('Cell density (cells/mm$^2$)')
plt.title('Initial cell density: 20,000 cells per well')
plt.legend(handles=legend_elements)
plt.grid(True)
plt.tight_layout()
plt.show()