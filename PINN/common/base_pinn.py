import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from copy import deepcopy
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
# from matplotlib import cm
import time
import random

from abc import ABC, abstractmethod


class BasePINN(ABC):
    def __init__(self,
                 input_dim,
                 output_dim,
                 net_arch,
                 physics_law,
                 physics_loss,
                 physics_loss_weight,
                 physics_kwargs,
                 ):
        pass

    @abstractmethod
    def _build_network(self):
        '''define self.net'''
        raise NotImplementedError()
    
    
    def train(self, ):
        pass
        