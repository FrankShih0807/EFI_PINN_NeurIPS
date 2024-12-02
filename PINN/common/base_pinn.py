import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from collections import deque

# from PINN.common.torch_layers import BaseDNN
from PINN.common.torch_layers import DropoutDNN
from PINN.common.logger import Logger, configure
from PINN.common.buffers import EvaluationBuffer



class BasePINN(object):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        lambda_pde=1,
        dropout_rate=0.0,
        save_path=None,
        device='cpu',
        verbose=1,
    ) -> None:
        super().__init__()
        self.physics_model = physics_model
        self.dataset = dataset.copy()

        # Physics loss
        self.differential_operator = self.physics_model.differential_operator
        self.lambda_pde = lambda_pde

        # Dropout config
        self.dropout_rate = dropout_rate
        
        # Common configs
        self.lr = lr
        self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        self.save_path = save_path
        self.device = device
        self.physics_model.plot_true_solution(save_path)

        # To device
        self.X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'solution'], dim=0).to(self.device)
        self.y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'solution'], dim=0).to(self.device)
                           
        self.eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        self.eval_y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        
        self.input_dim = self.X.shape[1]
        self.output_dim = self.y.shape[1]
        
        self.verbose = verbose
        if self.verbose == 1:
            format_strings = ["stdout", "csv"]
        else:
            format_strings = ["csv"]
        
        
        self.logger = configure(self.save_path, format_strings)
        self._get_scheduler()
        self._pinn_init()

    def _pinn_init(self):
        # init pinn net and optimiser
        self.net = DropoutDNN(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
            dropout_rate=self.dropout_rate,
        )
        # self.net = BaseDNN(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
        self.net.to(self.device)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
    
    def _get_scheduler(self):
        pass

    def update(self):
        ''' Implement the network parameter update here '''
        raise NotImplementedError()
    
    def train(self, epochs, eval_freq=-1, burn=0.5):
        if eval_freq == -1:
            eval_freq = epochs // 10
        self.eval_buffer = EvaluationBuffer(burn=burn)
        self.burn_steps = int(epochs * burn)
        self.n_eval = 0

        for ep in range(epochs):
            self.progress = (ep+1) / epochs
            tic = time.time()
            sol_loss, pde_loss = self.update()
            toc = time.time()
            

            self.eval_buffer.add(self.net(self.eval_X).detach())
            
            self.logger.record('train/progress', self.progress)
            self.logger.record('train/epoch', ep+1)
            self.logger.record_mean('train/sol_loss', sol_loss)
            self.logger.record_mean('train/pde_loss', pde_loss)
            self.logger.record_mean('train/time', toc-tic)
            
            

            ## 3. Loss calculation
            if (ep+1) % eval_freq == 0:
                self.evaluate_metric()
                self.physics_model.save_evaluation(self, self.save_path)
                self.physics_model.save_temp_frames(self, self.n_eval, self.save_path)
                
                self.logger.dump()
                self.n_eval += 1
        self.physics_model.create_gif(self.save_path)
    
    
    def evaluate_metric(self):
        eval_mean = self.eval_buffer.get_mean()
        ci_lower, ci_upper = self.eval_buffer.get_ci()
        ci_range = (ci_upper - ci_lower).mean().item()
        cr = ((ci_lower <= self.eval_y.flatten()) & (self.eval_y.flatten() <= ci_upper)).float().detach().cpu().mean().item()
        mse = F.mse_loss(eval_mean, self.eval_y.flatten(), reduction='mean').item()
        
        self.logger.record('eval/ci_range', ci_range)
        self.logger.record('eval/coverage_rate', cr)
        self.logger.record('eval/mse', mse)
        
    
    # def summary(self):
        
    #     y_pred_mat = torch.stack(self.collection[self.burn_steps+1::], dim=0)
    #     y_pred_upper = torch.quantile(y_pred_mat, 0.975, dim=0)
    #     y_pred_lower = torch.quantile(y_pred_mat, 0.025, dim=0)
    #     y_pred_mean = torch.mean(y_pred_mat, dim=0)
    #     y_pred_median = torch.quantile(y_pred_mat, 0.5, dim=0)
    #     y_covered = (y_pred_lower <= self.eval_y.clone().detach().cpu()) & (self.eval_y.clone().detach().cpu() <= y_pred_upper)
        
    #     summary_dict = {
    #         'y_preds_upper': y_pred_upper,
    #         'y_preds_lower': y_pred_lower,
    #         'y_preds_mean': y_pred_mean,
    #         'y_preds_median': y_pred_median,
    #         'y_covered': y_covered,
    #         'x_eval': self.eval_X.clone().detach().cpu().numpy(),
    #     }
        
    #     return summary_dict