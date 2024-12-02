
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
        self.total_ensemble = None
        self.n_ensemble = 0
        
    def add(self, value_tensor):
        value_tensor = value_tensor.unsqueeze(dim = 0).squeeze(dim=-1)
        

        if self.total_ensemble is None:
            self.total_ensemble = value_tensor.clone()
        else:
            self.total_ensemble = torch.cat([self.total_ensemble, value_tensor])

    
    def get_ci(self, p=0.05):
        self.ensemble_tensor = self.total_ensemble[int(self.burn*self.total_ensemble.shape[0]):]
        if self.ensemble_tensor is not None:
            q = torch.tensor([p/2, 1-p/2])
            quantiles = self.ensemble_tensor.quantile(q=q, dim=0)
        return quantiles[0], quantiles[1]
    
    def get_mean(self):
        self.ensemble_tensor = self.total_ensemble[int(self.burn*self.total_ensemble.shape[0]):]
        return self.ensemble_tensor.mean(dim=0)
        
    def __len__(self):
        self.ensemble_tensor = self.total_ensemble[int(self.burn*self.total_ensemble.shape[0]):]
        return self.ensemble_tensor.shape[0]


if __name__ == '__main__':
    buffer = EvaluationBuffer()
    for i in range(500):
        buffer.add(torch.randn(5,1))
        print(len(buffer))
        
        # print(buffer.ensemble_tensor.shape)
    # print(buffer.mean())
    print(len(buffer))
    
    print(buffer.get_ci())
    print(buffer.get_mean())
    # print(buffer.quantile())
    # print(buffer.last())
    print('---')
        