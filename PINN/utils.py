import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from ruamel.yaml import YAML
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from PINN.common.base_pinn import BasePINN
from PINN.common.base_physics import PhysicsModel
from PINN.common.callbacks import BaseCallback
from PINN import PINN, PINN_EFI, Pretrain_EFI, PINN_EFI_Discovery, BayesianPINN
from PINN.models import Cooling, EuropeanCall, Nonlinear, EuropeanCallDiscovery, Poisson, Poisson_v2, PoissonNonlinear
from PINN.models import PoissonCallback, Poisson_v2Callback, PoissonNonlinearCallback

from torch.utils.data import DataLoader,Dataset

yaml = YAML()
yaml.preserve_quotes = True


ALGOS: Dict[str, Type[BasePINN]] = {
    "pinn": PINN,
    "pinn_efi": PINN_EFI,
    "pretrain_efi": Pretrain_EFI,
    "pinn_efi_discovery": PINN_EFI_Discovery,
    "bpinn": BayesianPINN,
}

MODELS: Dict[str, Type[PhysicsModel]] = {
    "cooling": Cooling,
    "european_call": EuropeanCall,
    "nonlinear": Nonlinear,
    "european_call_discovery": EuropeanCallDiscovery,
    "poisson": Poisson,
    "poisson-v2": Poisson_v2,
    "poisson-nonlinear": PoissonNonlinear,
}

CALLBACKS: Dict[str, Callable] = {
    "poisson": PoissonCallback,
    "poisson-v2": Poisson_v2Callback,
    "poisson-nonlinear": PoissonNonlinearCallback,
}

def get_callback(key: str) -> Callable:
    return CALLBACKS.get(key, BaseCallback())

# ACTIVATIONS: Dict[str, Callable] = {
#     "relu": nn.ReLU(),
#     "tanh": nn.Tanh(),
#     "sigmoid": nn.Sigmoid(),
#     "leaky_relu": nn.LeakyReLU(),
#     "softplus": nn.Softplus(beta=10),
# }

def create_log_folder(path):
    os.makedirs(path, exist_ok=True)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.load(file)

def save_yaml(config, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(config, file)
        
def create_parser():
    parser = argparse.ArgumentParser(description='Initial Argument Parser')
    parser.add_argument('--algo', help="PINN algorithms", type=str, default="pinn", required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--model', help="Physics model", type=str, default="cooling", required=False, choices=list(MODELS.keys()))
    parser.add_argument('--task_id', type=int, default=-1)

    parser.add_argument('--exp_name', type=str, default=None)
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,  # Default is random seed
        help="Random seed for reproducibility (default: random seed)"
    )
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the code')
    
    parser.add_argument(
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
    )
    
    parser.add_argument(
        "--model_settings",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite physics model setting",
    )
    
    return vars(parser.parse_args())

def create_output_dir(inital_args):
    default_output_path = os.path.join(Path(__file__).parent.parent, 'output')
    if inital_args['exp_name'] is None:
        # inital_args['exp_name'] = '{}-test'.format(inital_args['algo'])
        model = inital_args['model']
        algo = inital_args['algo']
        inital_args['exp_name'] = f'{model}/{algo}'
    else:
        model = inital_args['model']
        algo = inital_args['algo']
        inital_args['exp_name'] = f'{model}/{inital_args["exp_name"]}'
    path = os.path.join(default_output_path, inital_args['exp_name'])
    os.makedirs(path, exist_ok=True)
    
    if inital_args['task_id'] >= 0:
        exp_name = 'exp_{}'.format(inital_args['task_id'])
    else:
        task_id = 0
        while 'exp_{}'.format(task_id) in os.listdir(path):
            task_id += 1
        exp_name = 'exp_{}'.format(task_id)       
    exp_path = os.path.join(path, exp_name)    
    os.makedirs(exp_path, exist_ok=True)
    return exp_path

def update_hyperparams(original_params, new_params):
    if new_params is not None:
        for key, value in new_params.items():
            
            if find_key_in_dict(original_params, key, value):
                print('update {}: {}'.format(key, value))
            else:
                raise KeyError(f"Hyperparameter '{key}' not found.")
            
        
class StoreDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        result = {}
        # print(values)
        for item in values:
            key, value = item.split(":")
            # print(key, value)   
            # Check if the value contains commas, indicating a list
            if "," in value:
                split_values = value.split(",")
                # print(split_values)
                result[key] = []
                for v in split_values:
                    if '.' in v:
                        result[key].append(float(v))
                    else:
                        result[key].append(int(v))
            elif 'activation' in key:
                result[key] = value
            else:
                # Handle single values
                result[key] = eval(value)
        setattr(namespace, self.dest, result)
     
def find_key_in_dict(d, key_to_find, new_value):
    """
    Check if key_to_find is in the dictionary d. This function
    searches recursively in all nested dictionaries.
    
    :param d: Dictionary in which to search for the key.
    :param key_to_find: Key to search for.
    :return: True if the key is found, False otherwise.
    """
    if key_to_find in d:
        d[key_to_find] = new_value
        return True
    for key, value in d.items():
        if isinstance(value, dict):
            if find_key_in_dict(value, key_to_find, new_value):
                return True
    return False

def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class Onet_dataset(Dataset):
    def __init__(self,y,u,Guy):
        self.y = y
        self.u = u
        self.Guy = Guy
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return self.y[idx],self.u[idx],self.Guy[idx]

