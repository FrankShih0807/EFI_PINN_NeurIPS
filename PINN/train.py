import os 
import torch
from pathlib import Path
import random
from PINN.utils import ALGOS, MODELS, load_yaml, save_yaml, create_output_dir, update_hyperparams, create_parser, set_random_seed

def train():
    args = create_parser()
    output_dir = create_output_dir(args)
    print(output_dir)
    
    yaml_dir = os.path.join(Path(__file__).parent.parent, 'hyperparams')
    algo_yaml_path = os.path.join(yaml_dir, args['algo']+'.yml')
    model_yaml_path = os.path.join(yaml_dir, 'models.yml')

    # Load hyperparameters from YAML
    hyperparams = load_yaml(algo_yaml_path)[args['model']]
    model_settings = load_yaml(model_yaml_path)[args['model']]

    # Update hyperparameters with command-line arguments
    update_hyperparams(hyperparams, args['hyperparams'])
    # Set random seed
    if args['seed'] is None:
        seed = random.randint(0, 10000)
    else:
        seed = args['seed']
    hyperparams['seed'] = seed
    # Set device
    if args['device'] == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args['device']
    print(f"Device: {device}")
    
    print(hyperparams)
    
    # Save the updated hyperparameters to a new YAML file
    new_yaml_file_path = os.path.join(output_dir, 'hyperparameters.yml')
    new_model_yaml_file_path = os.path.join(output_dir, 'model_settings.yml')
    save_yaml(hyperparams, new_yaml_file_path)
    save_yaml(model_settings, new_model_yaml_file_path)
    
    
    set_random_seed(hyperparams['seed'])
    physics_model = MODELS[args['model']](**model_settings)
    dataset = physics_model.generate_data(n_samples=hyperparams['n_samples'], device=device)

    pinn_class = ALGOS[args['algo']]
    pinn = pinn_class(physics_model = physics_model,
                      dataset = dataset,
                      **hyperparams['pinn'],
                      save_path=output_dir,
                      device=device
                      )
    pinn.train(epochs=hyperparams['epochs'])

    

if __name__ == "__main__":
    train()