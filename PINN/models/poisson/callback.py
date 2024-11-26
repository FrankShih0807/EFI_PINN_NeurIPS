from PINN.common.callbacks import BaseCallback
# from PINN.common.buffers import EvaluationBuffer

class PinnCallback(BaseCallback):
    def __init__(self, pinn, n_epoch):
        super().__init__(pinn, n_epoch)

    def _on_training(self):
        