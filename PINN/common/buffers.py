
from abc import ABC, abstractmethod
import collections
import random
import torch
from collections import namedtuple
from copy import deepcopy
import numpy as np


    
class EvaluationBuffer(object):
    def __init__(self, burn:float=0.5) -> None:
        # self.size = size
        self.burn = burn
        self.memory = []
        self.total_ensemble = None
        self.new_data_added = False
        
    def add(self, value_tensor):
        value_tensor = value_tensor.unsqueeze(dim = 0).squeeze(dim=-1)
        self.memory.append(value_tensor)
        self.new_data_added = True
    
    def get_ci(self, p=0.05):
        if self.new_data_added:
            self.total_ensemble = torch.cat(self.memory, dim=0)
            self.new_data_added = False
        q = torch.tensor([p/2, 1-p/2])
        quantiles = self.total_ensemble.quantile(q=q, dim=0)
        return quantiles[0], quantiles[1]
    
    def get_mean(self):
        if self.new_data_added:
            self.total_ensemble = torch.cat(self.memory, dim=0)
            self.new_data_added = False
        return self.total_ensemble.mean(dim=0)
    
    def last(self, n=1):
        return self.memory[-n:]   
        # return self.memory[-1]
    
    def reset(self):
        self.memory = []
        self.total_ensemble = None
        self.new_data_added = False
        print('reset buffer')

class ScalarBuffer(object):
    def __init__(self, burn:float=0.5) -> None:
        self.burn = burn
        self.samples = []
    
    def add(self, value):
        self.samples.append(value)
        
    def get_ci(self, p=0.05):
        quantiles = np.quantile(self.samples, [p/2, 1-p/2])
        return quantiles[0], quantiles[1]
    
    def get_mean(self):
        return np.mean(self.samples)

    def last(self, n=1):
        return self.samples[-n:]
        # return self.samples[-1]
    
    def reset(self):
        self.samples = []
        print('reset buffer')

    


if __name__ == '__main__':
    # buffer = EvaluationBuffer()
    buffer = ScalarBuffer()
    for i in range(500):
        buffer.add(i+1)
        # print(len(buffer))
        print(len(buffer.total_samples))
        
        # print(buffer.ensemble_tensor.shape)
    # print(buffer.mean())
    print(len(buffer))
    
    print(buffer.get_ci())
    print(buffer.get_mean())
    # print(buffer.quantile())
    # print(buffer.last())
    print('---')
        