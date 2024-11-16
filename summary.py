import numpy as np
import os

model = "poisson"
exp = "efi"
output_folder = f"output/{model}/{exp}"

n_runs = 100

coverage = np.zeros(n_runs)

for i in range(n_runs):
    run_path = f"{output_folder}/exp_{i}" 
    data_path = os.path.join(run_path, 'evaluation_data.npz')   
    if os.path.exists(run_path):
        if os.path.exists(data_path):
            data = np.load(data_path)
            # print(data['y_preds_mean'].shape)
            # print(data['y_preds_upper'].shape)
            # print(data['y_preds_lower'].shape)
            coverage[i] = data['y_covered'].mean()
    
print(coverage.mean(), coverage.std())

        
