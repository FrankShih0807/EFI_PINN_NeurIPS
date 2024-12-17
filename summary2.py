import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# def remove_outliers_iqr(data, multiplier=1.5):
#     if len(data) == 0:
#         return data  # Return empty array if input is empty
#     # Calculate Q1 (25th percentile) and Q3 (75th percentile)
#     q1 = np.percentile(data, 25)
#     q3 = np.percentile(data, 75)
#     # Compute the IQR
#     iqr = q3 - q1
#     # Define the lower and upper bounds for non-outliers
#     lower_bound = q1 - multiplier * iqr
#     upper_bound = q3 + multiplier * iqr
#     # Filter the data
#     filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
#     return filtered_data

# model = "poisson"
# # exp = "efi_sgd"
# exp = "efi_new_loss2"
# # exp = "efi_sgd_plw20"
# # exp = "efi_adam_plw20"
# # exp = "efi_adam_plw20"

# output_folder = f"output/{model}/{exp}"


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
                if not os.path.exists(os.path.join(exp_path, 'training_loss.gif')):
                    continue
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


def plot_latent_Z(output_dir):
    """
    Collects latent_Z.npy and true_Z.npy files from the directory structure,
    and creates scatter plots grouped by 'model' and 'algo'.

    Parameters:
    - output_dir (str): Root directory of the structure 'output/{model}/{algo}/exp_{i}/'

    Returns:
    - None: Generates scatter plots and saves them to files.
    """
    # Initialize a list to collect data for scatter plots
    all_data = []

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

                # Skip if latent_Z.npy or true_Z.npy does not exist
                latent_Z_path = os.path.join(exp_path, 'latent_Z.npy')
                true_Z_path = os.path.join(exp_path, 'true_Z.npy')

                if not (os.path.exists(latent_Z_path) and os.path.exists(true_Z_path)):
                    continue

                # Load the latent_Z and true_Z data
                latent_Z = np.load(latent_Z_path)
                true_Z = np.load(true_Z_path)

                # Combine data into a DataFrame with metadata
                df = pd.DataFrame({
                    'latent_Z': latent_Z,
                    'true_Z': true_Z,
                    'model': model,
                    'algo': algo,
                    'exp': exp
                })
                all_data.append(df)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    # Create scatter plots grouped by 'model' and 'algo'
    for (model, algo), group in combined_data.groupby(['model', 'algo']):
    #     plt.figure(figsize=(8, 6))
    #     sns.scatterplot(data=group, x='true_Z', y='latent_Z', alpha=0.6)
    #     # Add the x=y line
    #     min_val = min(group['true_Z'].min(), group['latent_Z'].min())
    #     max_val = max(group['true_Z'].max(), group['latent_Z'].max())
    #     plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='x = y')
        
    #     plt.title(f'Model: {model} | Algo: {algo}')
    #     plt.xlabel('True Z')
    #     plt.ylabel('Latent Z')
    #     plt.grid(True)

    #     # Save the plot
    #     output_path = os.path.join('figures/latent_Z', f'{model}_{algo}.png')
    #     plt.savefig(output_path)
    #     plt.close()

    # print(f"Scatter plots saved in figures")
                # Create a figure with two subplots: scatter plot and ordered comparison plot
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))

        # Scatter plot with x=y line
        sns.scatterplot(data=group, x='true_Z', y='latent_Z', alpha=0.6, ax=axes)
        min_val = min(group['true_Z'].min(), group['latent_Z'].min())
        max_val = max(group['true_Z'].max(), group['latent_Z'].max())
        axes.plot([min_val, max_val], [min_val, max_val], 'r--', label='x = y')
        axes.set_title(r'$Z_i$ vs $\hat{Z}_i$')
        axes.set_xlabel('')
        axes.set_ylabel('')
        
        axes.legend()
        axes.grid(True)
        output_path = os.path.join('figures/latent_Z', f'{model}_{algo}_1.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        # Ordered comparison plot
        sorted_latent_Z = np.sort(group['latent_Z'])
        sorted_true_Z = np.sort(group['true_Z'])
        axes.scatter(sorted_true_Z, sorted_latent_Z, alpha=0.6)
        axes.plot([min_val, max_val], [min_val, max_val], 'r--', label='x = y')
        axes.set_title(r'$Z_{(i)}$ vs $\hat{Z}_{(i)}$')
        # axes[1].set_xlabel('Ordered True Z')
        # axes[1].set_ylabel('Ordered Latent Z')
        axes.legend()
        axes.grid(True)

        # Save the combined plot
        output_path = os.path.join('figures/latent_Z', f'{model}_{algo}_2.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # print(f"Scatter and Ordered plots saved in {output_dir}")
    # Combine all DataFrames into one


def plot_cr_boxplot(df):
    plt.figure()
    ax = sns.boxplot(data=df, x='coverage_rate', y='algo', orient = 'h',
                     order=df['algo'],
                            meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"8"}
                    , boxprops={ "alpha": 0.3}
                    , showfliers=True
                    , showmeans=False
                    )
    plt.xlabel('Coverage rate', fontsize=14)
    plt.ylabel('Algorithm', fontsize=14)   
    plt.tight_layout()
    x_min, x_max = 0.3, 1.0
    x_ticks = np.arange(x_min, x_max, 0.05)
    x_ticks_major = np.arange(x_min, x_max+0.05, 0.1)
    
    ax.set_xticks(x_ticks, minor=True)
    ax.set_xticks(x_ticks_major, minor=False)
    ax.axvline(x=0.95, color='red', linestyle='--', label='95%')
    ax.xaxis.grid(True, which='major', linestyle='-', linewidth=1)
    ax.xaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)
    # legend = plt.legend(fontsize=11, loc='lower right')
    legend = plt.legend(fontsize=11, loc='upper left')
    legend.get_title().set_text('') 
    plt.xticks(fontsize=11)  # Adjust the fontsize as needed for the x-axis
    plt.yticks(fontsize=11)
    # sns.set(font_scale=1.3)
    plt.savefig(os.path.join('figures','boxplot','cr_boxplot.png'))
    print('cr boxplot saved')
    plt.close()
    

def clear_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            # Check if it is a file or a directory
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path):
                # shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
# Example usage
output_dir = "output"
# clear_dir('figures/latent_Z')
progress_df = collect_progress_data(output_dir)



df = progress_df[progress_df['train/progress']==1.0]
df = df.loc[:, ~df.columns.str.startswith('train')]
df.rename(columns=lambda x: x.split('/')[-1], inplace=True)
df = df.dropna()
# plot_cr_boxplot(df)
# df["cover_all"] = (df["coverage_rate"]==1)
# print(df_cr.groupby(['model', 'algo'])['cr'].mean())

# print(df_cr[df_cr['cr']<0.3])
# print(progress_df[progress_df['train/progress']==1.0])
# print("coverage rate")


# Filter out rows with abnormal large mse
abnormal_mse_indices = df[df['mse'] > 1].index
filtered_df = df[df['mse'] <= 1]

# Find the group and in-group index for abnormal mse rows
abnormal_indices_info = []
for idx in abnormal_mse_indices:
    row = df.loc[idx]
    group_key = (row['model'], row['algo'])
    group_df = df[(df['model'] == row['model']) & (df['algo'] == row['algo'])]
    in_group_index = group_df.index.get_loc(idx)
    abnormal_indices_info.append((idx, group_key, in_group_index))

# Print the indices of rows with abnormal large mse and their in-group index
print("Indices of rows with abnormal large mse and their in-group index:")
for info in abnormal_indices_info:
    print(f"Global index: {info[0]}, Group: {info[1]}, In-group index: {info[2]}")

# Calculate and print the mean and std of the filtered dataframe
print(filtered_df.groupby(['model', 'algo'])[['coverage_rate', 'mse', 'ci_range']].mean())
print(filtered_df.groupby(['model', 'algo'])[['coverage_rate', 'mse', 'ci_range']].std()/10)

# print(df.groupby(['model', 'algo'])[['coverage_rate', 'mse', 'ci_range']].mean())
# print(df.groupby(['model', 'algo'])[['coverage_rate', 'mse', 'ci_range']].std()/10)

# df2 = df[df['algo']=='pinn_efi_lam1k_2']
rows_with_nan = df[df.isna().any(axis=1)]
print(rows_with_nan)
# low_cr = df2[df2['coverage_rate']<0.5]
# print(low_cr)
# print("mse")
# print(df.groupby(['model', 'algo'])['eval/mse'].mean())
# print("ci range")
# print(df.groupby(['model', 'algo'])['eval/ci_range'].mean())

# print(df[df['eval/coverage_rate']<0.2][['model', 'algo', 'exp']])
# plot_latent_Z(output_dir)

