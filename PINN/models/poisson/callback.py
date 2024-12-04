from PINN.common.callbacks import BaseCallback
# from PINN.common.buffers import EvaluationBuffer

class PoissonCallback(BaseCallback):
    def __init__(self, model, eval_freq=1000, burn=0.5):
        super().__init__(model, eval_freq, burn)
    
    def _init_callback(self) -> None:
        
    
    
    def _on_training(self):
        