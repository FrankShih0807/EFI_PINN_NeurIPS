from PINN.common.callbacks import BaseCallback
import torch
import torch.nn.functional as F
# from PINN.common.buffers import EvaluationBuffer

class PoissonCallback(BaseCallback):
    def __init__(self):
        super().__init__()
    
    def _init_callback(self) -> None:
        self.eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        self.eval_y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        
        self.eval_X_cpu = self.eval_X.clone().detach().cpu()
        self.eval_y_cpu = self.eval_y.clone().detach().cpu()

    
    def _on_training(self):
        pred_y = self.model.net(self.eval_X).detach().cpu().numpy()
        self.eval_buffer.update(pred_y)
        
    
    def _on_eval(self):
        pred_y_mean = self.eval_buffer.get_mean()
        ci_low, ci_high = self.eval_buffer.get_ci()
        ci_range = (ci_high - ci_low).mean().item()
        cr = ((ci_low <= self.eval_y_cpu.flatten()) & (self.eval_y_cpu.flatten() <= ci_high)).float().mean().item()
        mse = F.mse_loss(pred_y_mean, self.eval_y_cpu.flatten(), reduction='mean').item()
        
        self.logger.record('eval/ci_range', ci_range)
        self.logger.record('eval/coverage_rate', cr)
        self.logger.record('eval/mse', mse)
        
        self.physics_model.save_evaluation(self.model, self.save_path)
        self.physics_model.save_temp_frames(self.model, self.n_evals, self.save_path)