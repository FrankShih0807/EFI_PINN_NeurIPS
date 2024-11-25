import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from collections import deque

from PINN.common.torch_layers import BaseDNN
from PINN.common.logger import Logger, configure



class BasePINN(object):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        lambda_pde=1,
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
        self._pinn_init()
        self._get_scheduler()

    def _pinn_init(self):
        # init pinn net and optimiser
        self.net = BaseDNN(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
        self.net.to(self.device)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
    
    def _get_scheduler(self):
        pass

    def update(self):
        ''' Implement the network parameter update here '''
        raise NotImplementedError()
    
    def train(self, epochs, eval_freq=1000):
        self.collection = []
        
        eval_losses = []
        sol_losses = []
        pde_losses = []
        
        # tic = time.time()
        for ep in range(epochs):
            self.progress = (ep+1) / epochs
            tic = time.time()
            sol_loss, pde_loss = self.update()
            toc = time.time()
            
            sol_losses.append(sol_loss)
            pde_losses.append(pde_loss)
            
            
            
            self.logger.record('train/progress', self.progress)
            self.logger.record('train/epoch', ep+1)
            self.logger.record_mean('train/sol_loss', sol_loss)
            self.logger.record_mean('train/pde_loss', pde_loss)
            self.logger.record_mean('train/time', toc-tic)
            ## 3. Loss calculation
            if (ep+1) % eval_freq == 0:
                eval_loss = self.mse_loss(self.eval_y, self.net(self.eval_X)).item()
                eval_losses.append(eval_loss)
                self.logger.record('eval/loss', eval_loss)
                self.logger.dump()


            # if ep > epochs - 1000:
            y_pred = self.evaluate().detach().cpu()
            self.collection.append(y_pred)
        
        self.physics_model.save_evaluation(self, self.save_path)
        return eval_losses, sol_losses, pde_losses
    

    def predict(self, X):
        self.net.eval()
        out = self.net(X)
        return out.detach().cpu().numpy()
    
    def evaluate(self):
        y = self.net(self.eval_X).detach()
        return y
    
    def summary(self):
        y_pred_mat = torch.stack(self.collection[-1000::], dim=0)
        y_pred_upper = torch.quantile(y_pred_mat, 0.975, dim=0)
        y_pred_lower = torch.quantile(y_pred_mat, 0.025, dim=0)
        y_pred_mean = torch.mean(y_pred_mat, dim=0)
        y_pred_median = torch.quantile(y_pred_mat, 0.5, dim=0)
        y_covered = (y_pred_lower <= self.eval_y.clone().detach().cpu()) & (self.eval_y.clone().detach().cpu() <= y_pred_upper)
        
        summary_dict = {
            'y_preds_upper': y_pred_upper,
            'y_preds_lower': y_pred_lower,
            'y_preds_mean': y_pred_mean,
            'y_preds_median': y_pred_median,
            'y_covered': y_covered,
            'x_eval': self.eval_X.clone().detach().cpu().numpy(),
        }
        
        return summary_dict