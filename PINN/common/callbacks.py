# modified from stable_baselines3

from abc import ABC, abstractmethod
from PINN.common.logger import Logger
from typing import Any, Callable, Dict, List, Optional, Union
from PINN.common.buffers import EvaluationBuffer
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    logger: Logger

    def __init__(self):
        super().__init__()
        # An alias for physical model
        self.physics_model = None
        # Number of time the callback was called
        self.n_trains = 0  # type: int
        self.n_evals = 0  # type: int

    # Type hint as string to avoid circular import
    def init_callback(self, model, eval_freq=1000, burn=0.5) -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self.physics_model = model.physics_model
        self.logger = model.logger
        self.eval_buffer = EvaluationBuffer(burn=burn)
        self.epochs = self.model.epochs
        if eval_freq == -1:
            self.eval_freq = self.epochs // 10
        else:
            self.eval_freq = eval_freq
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training(self) -> None:
        # Update num_timesteps in case training was done before
        self._on_training()
        self.n_trains += 1
        if self.n_trains % self.eval_freq == 0:
            self.on_eval()

    def _on_training(self) -> None:
        pass
    
    def on_eval(self) -> None:
        self._on_eval()
        self.n_evals += 1
    
    def _on_eval(self) -> None:
        pass

    def on_training_end(self) -> None:
        print("training end")
        self._on_training_end()
    
    def _on_training_end(self) -> None:
        pass
    
    
        
if __name__ == '__main__':
    x = y = np.arange(0.05, 0.85, 0.1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    x_length = X.shape[0]
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    
    Z = torch.stack([X,Y], dim=-1)
    print(X)
    print(Y)
    
