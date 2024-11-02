from PINN.utils import *
import os 
from pathlib import Path
from PINN.utils import ALGOS, MODELS, load_yaml, save_yaml, create_output_dir, update_hyperparams, create_parser

if __name__ == "__main__":
    args = create_parser()
    
    
    home_dir = os.getcwd()
    yaml_dir = os.path.join(home_dir, 'hyperparams')
    algo_yaml_path = os.path.join(yaml_dir, 'pinn.yml')

    # Load hyperparameters from YAML
    hyperparams = load_yaml(algo_yaml_path)[args['model']]
    print(hyperparams)
    
    print(args)
    # Update hyperparameters with command-line arguments
    update_hyperparams(hyperparams, args['hyperparams'])