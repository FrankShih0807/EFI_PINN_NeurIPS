
from abc import ABC, abstractmethod
import collections
import random
import torch
from collections import namedtuple
from copy import deepcopy
import numpy as np


    
class PredictionBuffer(object):
    def __init__(self, size) -> None:
        self.size = size
        self.ensemble_tensor = None
        
    def add(self, value_tensor):
        value_tensor = value_tensor.unsqueeze(dim = 0).squeeze(dim=-1)
            
        if self.ensemble_tensor is None:
            self.ensemble_tensor = value_tensor.clone()
        else:
            self.ensemble_tensor = torch.cat([self.ensemble_tensor, value_tensor])
        if self.ensemble_tensor.shape[0] > self.size:
            self.ensemble_tensor = self.ensemble_tensor[-self.size::]

    def mean(self):
        return torch.mean(self.ensemble_tensor, dim=0)
    
    def last(self):
        return self.ensemble_tensor[-1,...]
    
    def quantile(self, p=0.05):
        if self.ensemble_tensor is not None:
            q = torch.tensor([p/2, 0.5, 1-p/2])
            quantiles = self.ensemble_tensor.quantile(q=q, dim=0)
            return quantiles[0], quantiles[1], quantiles[2]
    
    def prediction(self):
        lo, med, hi = self.quantile()
        center = self.last()
        upper = center + (hi-lo)/2
        lower = center - (hi-lo)/2
        return lower, upper
        
    def __len__(self):
        return self.ensemble_tensor.shape[0]


if __name__ == '__main__':
    buffer = PredictionBuffer(10)
    for i in range(20):
        buffer.add(torch.rand(5,1))
        # print(buffer.ensemble_tensor.shape)
        print(buffer.mean())
        print(buffer.prediction())
        print(buffer.quantile())
        print(buffer.last())
        print(len(buffer))
        print('---')
        