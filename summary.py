import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab


def collect_progress_data(output_dir, model):
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
    model_path = os.path.join(output_dir, model)
    if not os.path.isdir(model_path):
        print(model_path, 'does not exist.')
        return 0
    
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
                    df = pd.read_csv(csv_file).tail(1)
                    # Add columns for 'model', 'algo', and 'exp'
                    df['model'] = model
                    df['algo'] = algo
                    df['exp'] = exp
                    df['done'] = (df['train/progress'] == 1.0)
                    df = df.loc[:, ~df.columns.str.startswith('train')]
                    
                    # if df['train/progress'] >= (1-1e-6):
                    #     df['done'] = 1
                    # else:
                    #     df['done'] = 0
                    # Append to the list of DataFrames
                    data_frames.append(df)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
    
    # Combine all DataFrames into one
    combined_df = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    combined_df.rename(columns=lambda x: x.split('/')[-1], inplace=True)
    
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
                # if not os.path.exists(os.path.join(exp_path, 'training_loss.gif')):
                #     continue
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
        output_path = os.path.join('figures/latent_Z', f'{model}_{algo}_scatter.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        

        res = stats.probplot(group['latent_Z'], dist="norm", plot=pylab)
        ax = pylab.gca()
        line0 = ax.get_lines()[0]
        line0.set_alpha(0.3)

        # Add titles, labels, and grid
        ax.set_title("Q-Q Plot of Z")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.grid(True, alpha=0.5)  # Adjust grid transparency

        # Save the figure
        output_path = os.path.join('figures/latent_Z', f'{model}_{algo}_qq.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # print(f"Scatter and Ordered plots saved in {output_dir}")
    # Combine all DataFrames into one

def plot_latent_Z_diff(output_dir):
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
                if not os.path.exists(os.path.join(exp_path, 'training_loss.gif')):
                    continue
                # Skip if latent_Z.npy or true_Z.npy does not exist
                latent_Z_path = os.path.join(exp_path, 'latent_Z_diff.npy')
                true_Z_path = os.path.join(exp_path, 'true_Z_diff.npy')

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
        output_path = os.path.join('figures/latent_Z_diff', f'{model}_{algo}_scatter.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        

        res = stats.probplot(group['latent_Z'], dist="norm", plot=pylab)
        ax = pylab.gca()
        line0 = ax.get_lines()[0]
        line0.set_alpha(0.3)

        # Add titles, labels, and grid
        ax.set_title("Q-Q Plot of Z")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.grid(True, alpha=0.5)  # Adjust grid transparency

        # Save the figure
        output_path = os.path.join('figures/latent_Z_diff', f'{model}_{algo}_qq.png')
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
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            


if __name__ == '__main__':
    # Example usage
    output_dir = "output"
    clear_dir('figures/latent_Z')
    clear_dir('figures/latent_Z_diff')

    models = os.listdir(output_dir)
    
    file = open('metric.txt', 'w')
    pd.set_option('display.max_rows', None)   
    for model in models:
        print(model)
        df = collect_progress_data(output_dir, model)
        df = df[df['done']==1.0]
        # print(df)
        print(df[df['mse']>0.001])
        # print(df[df['done']==0.0])
        print(df[df['coverage_rate']<0.5])
        df = df.dropna()
        df = df.drop(columns=['exp'])
        print(df.groupby(['model', 'algo']).mean())
        
        file.write('Mean\n')
        file.write(df.groupby(['model', 'algo']).mean().to_string())
        file.write('\n')
        file.write('Standard error\n')
        file.write((df.groupby(['model', 'algo']).std()/10).to_string())
        file.write('\n')
        file.write('='*100 + '\n')
        
    # print(progress_df[progress_df['train/progress']<1.0])
    # df = progress_df[progress_df['done']==True]
    # df = progress_df

    # df = df.loc[:, ~df.columns.str.startswith('train')]
    # df.rename(columns=lambda x: x.split('/')[-1], inplace=True)
    # print(df[(df['mse']>0.001) & (df['model']=='poisson-v2') ])
    # # print(df[(df['k_coverage_rate']<1.0) &(df['model']=='poisson-inverse')  ])

    # # df = df[['model', 'algo', 'mse', 'coverage_rate', 'ci_range', 'k_mean', 'k_coverage_rate', 'k_ci_range', 'mse_idx0', 'mse_idx15', 'mse_idx29', 'cr_idx0', 'cr_idx15', 'cr_idx29', 'ci_range_idx0', 'ci_range_idx15', 'ci_range_idx29']]
    # df = df[['model', 'algo', 'mse', 'coverage_rate', 'ci_range']]


    # print(df.groupby(['model', 'algo']).mean())
    # print(df.groupby(['model', 'algo']).std()/10)

    # with open('metric.txt', 'w') as file:
    #     file.write('Mean\n')
    #     file.write(df.groupby(['model', 'algo']).mean().to_string())
    #     file.write('\n')
    #     file.write('Standard error\n')
    #     file.write((df.groupby(['model', 'algo']).std()/10).to_string())
        
    # print(df.groupby(['model', 'algo']).size())


    # rows_with_nan = df[df.isna().any(axis=1)]

    try:
        plot_latent_Z(output_dir)
    except:
        pass
    try:
        plot_latent_Z_diff(output_dir)
    except:
        pass
