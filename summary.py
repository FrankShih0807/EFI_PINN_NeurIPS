import numpy as np
import os

model = "poisson"
# exp = "efi_sgd"
exp = "efi_noise01"
# exp = "efi_sgd_plw20"
# exp = "efi_adam_plw20"
# exp = "efi_adam_plw20"

output_folder = f"output/{model}/{exp}"

n_runs = 100


def cr(dir):
    
    coverage = np.zeros(n_runs)
    for i in range(n_runs):
        run_path = f"{dir}/exp_{i}" 
        data_path = os.path.join(run_path, 'evaluation_data.npz')   
        if os.path.exists(run_path):
            if os.path.exists(data_path):
                data = np.load(data_path)
                # print(data['y_preds_mean'].shape)
                # print(data['y_preds_upper'].shape)
                # print(data['y_preds_lower'].shape)
                coverage[i] = data['y_covered'].mean()
                # coverage[i] = (data['y_covered'].sum()==100)
    print(dir)
    print(coverage)
    print(coverage.mean(), coverage.std())

        
cr(output_folder)