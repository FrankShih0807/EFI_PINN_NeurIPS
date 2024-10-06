from .examples import *
from common import BaseNetwork, grad
import torch
import torch.nn as nn
from .old_code.diff_equations import *
from .examples.cooling import physics_loss

class PINN(object):
    def __init__(self, ) -> None:
        pass



if __name__ == "__main__":
    
    input_dim = 1
    output_dim = 1
    net_arch = [100, 100, 100, 100]
    
    net = BaseNetwork(input_dim, output_dim, net_arch, nn.Tanh)

    
    
    epochs = 30000
    loss2 = physics_loss