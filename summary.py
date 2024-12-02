import numpy as np
import os
import pandas as pd

def remove_outliers_iqr(data, multiplier=1.5):
    if len(data) == 0:
        return data  # Return empty array if input is empty
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # Compute the IQR
    iqr = q3 - q1
    # Define the lower and upper bounds for non-outliers
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    # Filter the data
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return filtered_data

model = "poisson"
# exp = "efi_sgd"
exp = "efi_new_loss2"
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
                coverage[i] = data['y_covered'].sum()
                # coverage[i] = (data['y_covered'].sum()==100)
    print(dir)
    print(np.hstack([np.arange(n_runs).reshape(-1,1), coverage.reshape(-1,1)]))
    print(coverage.mean(), coverage.std(), np.quantile(coverage, 0.5))
    
    filtered_coverage = remove_outliers_iqr(coverage)
    print('mean without outliers:', filtered_coverage.mean())

        
cr(output_folder)

def collect_progress_data(output_dir):
    """
    Collects all progress data from progress.csv files in the directory structure
    and combines them into a single DataFrame with additional columns for 'model', 'algo', and 'exp'.

    Parameters:
    - output_dir (str): Root directory of the structure 'output/{model}/{algo}/exp_{i}/progress.csv'
    
    Returns:
    - pd.DataFrame: Combined DataFrame containing progress data with 'model', 'algo', and 'exp' information.
    """
    data_frames = []

    # Walk through the directory structure
    for model in os.listdir(output_dir):
        model_path = os.path.join(output_dir, model)
        if not os.path.isdir(model_path):
            continue
        
        for algo in os.listdir(model_path):
            algo_path = os.path.join(model_path, algo)
            if not os.path.isdir(algo_path):
                continue
            
            for exp in os.listdir(algo_path):
                exp_path = os.path.join(algo_path, exp)
                if not os.path.isdir(exp_path):
                    continue
                
                csv_file = os.path.join(exp_path, 'progress.csv')
                if os.path.exists(csv_file):
                    try:
                        # Read the CSV file
                        df = pd.read_csv(csv_file)
                        # Add columns for 'model', 'algo', and 'exp'
                        df['model'] = model
                        df['algo'] = algo
                        df['exp'] = exp
                        # Append to the list of DataFrames
                        data_frames.append(df)
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")
    
    # Combine all DataFrames into one
    combined_df = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    return combined_df

# Example usage
output_dir = "output"
progress_df = collect_progress_data(output_dir)
# print(df_cr.groupby(['model', 'algo'])['cr'].mean())

# print(df_cr[df_cr['cr']<0.3])
print(progress_df)
print(progress_df.groupby(['model', 'algo'])['eval/coverage_rate'].mean())
