import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab






def collect_progress_data(output_dir, model, algo):
    """
    Collects all progress data from progress.csv files in the directory structure
    and combines them into a single DataFrame with additional columns for 'model', 'algo', and 'exp'.

    Parameters:
    - output_dir (str): Root directory of the structure 'output/{model}/{algo}/exp_{i}/progress.csv'
    
    Returns:
    - pd.DataFrame: Combined DataFrame containing progress data with 'model', 'algo', and 'exp' information.
    """
    data_frames = []
    initial_density = [10_000, 12_000, 14_000, 16_000, 18_000, 20_000]

    # Walk through the directory structure
    model_path = os.path.join(output_dir, model)
    if not os.path.isdir(model_path):
        print(model_path, 'does not exist.')
        return 0

    algo_path = os.path.join(model_path, algo)
    
    for exp in os.listdir(algo_path):
        exp_path = os.path.join(algo_path, exp)
        if not os.path.isdir(exp_path):
            continue
        
        csv_file = os.path.join(exp_path, 'progress.csv')
        # if not os.path.exists(os.path.join(exp_path, 'training_loss.gif')):
        #     continue
        if os.path.exists(csv_file):
            # try: 
                # Read the CSV file
            df = pd.read_csv(csv_file).tail(1)
            # Add columns for 'model', 'algo', and 'exp'
            df['model'] = model
            df['algo'] = algo
            # print(exp.split('_'))
            df['initial_density'] = initial_density[int(exp.split('_')[-1])]
            df['done'] = (df['train/progress'] == 1.0)
            # df = df.loc[:, ~df.columns.str.startswith('train')]
            # df = df.loc[:, ~df.columns.str.startswith('train')]
            
            data_frames.append(df)
            # except Exception as e:
                # print(f"Error reading {csv_file}: {e}")
    
    # Combine all DataFrames into one
    combined_df = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    combined_df.rename(columns=lambda x: x.split('/')[-1], inplace=True)
    
    drop_columns = ['time', 'callback_time', 'progress', 'epoch', 'done', 
                    'R_mean', 'R_high', 'R_low', 
                    'D_mean', 'D_high', 'D_low', 
                    'M_mean', 'M_high', 'M_low']
    for col in drop_columns:
        if col in combined_df.columns:
            combined_df.drop(columns=[col], inplace=True)
            
    combined_df.sort_values(by='initial_density', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    # Reorder columns: put 'model', 'algo', 'initial_density' in the front
    front_cols = ['model', 'algo', 'initial_density', 'gls_loss', 'sd_mean', 'ci_range']
    other_cols = [col for col in combined_df.columns if col not in front_cols]
    combined_df = combined_df[front_cols + other_cols]
    
    return combined_df



if __name__ == '__main__':
    # Example usage
    output_dir = "output"

    models = ['fkpp', 'porous_fkpp']
    

    

    for model in models:
        model_path = os.path.join(output_dir, model)
        for algo in os.listdir(model_path):
            print('\n')
            print(model, algo)
            df = collect_progress_data(output_dir, model, algo)
            print(df)
            # df = df[df['done']==1.0]

            # df = df.drop(columns=['exp'])
