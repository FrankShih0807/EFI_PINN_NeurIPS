import os 
# import numpy as np
# import argparse
# from ruamel.yaml import YAML
from pathlib import Path


from PINN.utils import load_yaml, save_yaml, create_output_dir, update_hyperparams, create_parser

def train():
    args = create_parser()
    output_dir = create_output_dir(args)
    
    yaml_dir = os.path.join(Path(__file__).parent.parent, 'hyperparams')
    algo_yaml_path = os.path.join(yaml_dir, args['algo']+'.yml')

    # Load hyperparameters from YAML
    hyperparams = load_yaml(algo_yaml_path)
    # Update hyperparameters with command-line arguments
    update_hyperparams(hyperparams, args['hyperparams'])

    # Save the updated hyperparameters to a new YAML file
    new_yaml_file_path = os.path.join(output_dir, 'hyperparameters.yml')
    save_yaml(hyperparams, new_yaml_file_path)
    
    total_timesteps = hyperparams.pop('total_timesteps')
    callback_kwargs = hyperparams.pop('callback_kwargs')

    for key, value in hyperparams.items():
        print("{}: {}".format(key, value))
    


    

if __name__ == "__main__":
    train()