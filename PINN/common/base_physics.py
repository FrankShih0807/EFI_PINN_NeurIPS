import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from PINN.common.grad_tool import grad



class PhysicsModel(object):
    def __init__(self, **kwargs) -> None:

        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.X, self.y = self._data_generation()
        self.input_dim = self.X.shape[1]
        self.output_dim = self.y.shape[1]
        self.n_samples = self.X.shape[0]

    
    def physics_law(self):
        ''' Implement the physics law here '''
        raise NotImplementedError()
    
    def physics_loss(self):
        ''' Implement the physics loss here '''
        raise NotImplementedError()