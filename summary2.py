import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



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
                # if not os.path.exists(os.path.join(exp_path, 'training_loss.gif')):
                #     continue
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
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))



# Example usage
output_dir = "output"
# clear_dir('figures/latent_Z')
progress_df = collect_progress_data(output_dir)


df = progress_df[progress_df['train/progress']==1.0]
df = df.loc[:, ~df.columns.str.startswith('train')]
df.rename(columns=lambda x: x.split('/')[-1], inplace=True)
# df = df.dropna()
# plot_cr_boxplot(df)
# df["cover_all"] = (df["coverage_rate"]==1)
# print(df_cr.groupby(['model', 'algo'])['cr'].mean())

# print(df_cr[df_cr['cr']<0.3])
# print(progress_df[progress_df['train/progress']==1.0])
# print("coverage rate")


# Filter out rows with abnormal large mse or nan mse
abnormal_mse_indices = df[(df['mse'] > .1) | (df['mse'].isna())].index
filtered_df = df[df['mse'] <= .1]

# Find the group and in-group index for abnormal mse rows
abnormal_indices_info = []
for idx in abnormal_mse_indices:
    row = df.loc[idx]
    group_key = (row['model'], row['algo'], row['exp'])
    if pd.isna(row['mse']):
        filtered_reason = "nan"
    else:
        filtered_reason = "large"
    abnormal_indices_info.append((group_key, filtered_reason))

# Print the indices of rows with abnormal large mse and their in-group index
print("Indices of rows with abnormal large mse and their in-group index:")
for info in abnormal_indices_info:
    print(f"Group: {info[0]}, Reason: {info[1]}")

# Calculate and print the mean and std of the filtered dataframe
pd.set_option('display.max_rows', 200)
# print(filtered_df.groupby(['model', 'algo'])[['coverage_rate', 'mse', 'ci_range']].mean())
# print(filtered_df.groupby(['model', 'algo'])[['coverage_rate', 'mse', 'ci_range']].std()/10)

print(filtered_df.groupby(['model', 'algo'])[['mse', 'coverage_rate', 'ci_range', 'k_mean', 'k_coverage_rate', 'k_ci_range']].mean())
print(filtered_df.groupby(['model', 'algo'])[['mse', 'coverage_rate', 'ci_range', 'k_mean', 'k_coverage_rate', 'k_ci_range']].std()/10)

# print(filtered_df.groupby(['model', 'algo'])[['mse_idx0', 'cr_idx0', 'ci_range_idx0', 'mse_idx15', 'cr_idx15', 'ci_range_idx15']].mean())
# print(filtered_df.groupby(['model', 'algo'])[['mse_idx0', 'cr_idx0', 'ci_range_idx0', 'mse_idx15', 'cr_idx15', 'ci_range_idx15']].std()/10)

# print(df.groupby(['model', 'algo'])[['coverage_rate', 'mse', 'ci_range']].mean())
# print(df.groupby(['model', 'algo'])[['coverage_rate', 'mse', 'ci_range']].std()/10)

# df2 = df[df['algo']=='pinn_efi_lam1k_2']

# rows_with_nan = df[df.isna().any(axis=1)]
# print(rows_with_nan)

# low_cr = df2[df2['coverage_rate']<0.5]
# print(low_cr)
# print("mse")
# print(df.groupby(['model', 'algo'])['eval/mse'].mean())
# print("ci range")
# print(df.groupby(['model', 'algo'])['eval/ci_range'].mean())

# print(df[df['eval/coverage_rate']<0.2][['model', 'algo', 'exp']])
# plot_latent_Z(output_dir)

