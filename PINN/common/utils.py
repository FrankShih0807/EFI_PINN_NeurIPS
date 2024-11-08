import torch
import torch.nn as nn

def get_activation_fn(activation):
    # Define a dictionary mapping activation names to PyTorch classes
    activation_dict = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'softplus': nn.Softplus,
        'softmax': nn.Softmax,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
    }
    
    # If the activation is already an instance of a PyTorch activation, return it directly
    if isinstance(activation, nn.Module):
        return activation
    
    # If activation is a string, parse it
    elif isinstance(activation, str):
        # Extract function name and optional parameters
        if "(" in activation and ")" in activation:
            # Example: 'softplus(beta=2)' -> 'softplus', {'beta': 2}
            func_name, params_str = activation.split("(", 1)
            func_name = func_name.strip().lower()
            params_str = params_str.rstrip(")")
            
            # Parse parameters into a dictionary
            params = {}
            for param in params_str.split(","):
                key, value = param.split("=")
                params[key.strip()] = eval(value.strip())
            
            # Check if the function name is in the dictionary
            if func_name in activation_dict:
                return activation_dict[func_name](**params)
            else:
                raise ValueError(f"Unsupported activation function: '{func_name}'")
        
        else:
            # If no parameters are provided, simply use the activation name
            func_name = activation.lower()
            if func_name in activation_dict:
                return activation_dict[func_name]()
            else:
                raise ValueError(f"Unsupported activation function: '{func_name}'")
    
    else:
        raise TypeError("Activation must be either a string or a torch.nn activation class.")


# def get_activation_fn(activation):
#     # If the activation is already an instance of a PyTorch activation, return it directly
#     if isinstance(activation, nn.Module):
#         return activation

#     # If activation is a string, match it to a PyTorch activation
#     elif isinstance(activation, str):
#         activation = activation.lower()
        
#         # Define a dictionary mapping activation names to PyTorch classes
#         activation_dict = {
#             'relu': nn.ReLU,
#             'leaky_relu': nn.LeakyReLU,
#             'sigmoid': nn.Sigmoid,
#             'tanh': nn.Tanh,
#             'softplus': nn.Softplus,
#             'softmax': nn.Softmax,
#             'elu': nn.ELU,
#             'selu': nn.SELU,
#             'gelu': nn.GELU,
#         }
        
#         # Check if the activation string is in the dictionary and return an instance
#         if activation in activation_dict:
#             return activation_dict[activation]()
#         else:
#             raise ValueError(f"Unsupported activation function: '{activation}'")
    
#     else:
#         raise TypeError("Activation must be a string, nn.Module instance, or torch.nn class.")


if __name__ == "__main__":
    # Example usage
    activation_fn1 = get_activation_fn('relu')             # Returns nn.ReLU() instance
    activation_fn2 = get_activation_fn("softplus(beta=2)") # Returns nn.Softplus(beta=2) instance
    activation_fn3 = get_activation_fn(nn.Sigmoid())         # Returns nn.Sigmoid() instance.
    activation_fn4 = get_activation_fn(nn.Softplus(beta=2))       # Returns nn.LeakyReLU() instance.

    print(activation_fn1(torch.tensor([-1.0]) ))  # Output: ReLU()
    print(activation_fn2)  # Output: Softplus(beta=2)
    print(activation_fn3)  # Output: Sigmoid()
    print(activation_fn4)  # Output: Leaky