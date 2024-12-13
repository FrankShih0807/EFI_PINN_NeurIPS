# modified from stable_baselines3

from abc import ABC, abstractmethod
from PINN.common.logger import Logger
from typing import Any, Callable, Dict, List, Optional, Union
from PINN.common.buffers import EvaluationBuffer


class BaseCallback(ABC):
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
        self.dataset = model.dataset
        self.physics_model = model.physics_model
        self.dataset = model.dataset
        self.logger = model.logger
        self.save_path = model.save_path
        self.epochs = self.model.epochs
        self.eval_freq = eval_freq
        self.device = self.model.device
        self.eval_buffer = EvaluationBuffer(burn=burn)
        
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training(self) -> None:
        self._on_training()
        self.n_trains += 1

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
    
    
