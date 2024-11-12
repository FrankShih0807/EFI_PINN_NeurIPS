import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from collections import deque

from PINN.common.torch_layers import BaseDNN



class BasePINN(object):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        physics_loss_weight=1,
        save_path=None,
        device='cpu'
    ) -> None:
        super().__init__()
        self.physics_model = physics_model
        self.dataset = dataset.copy()
        for key, value in self.physics_model.__dict__.items():
            setattr(self, key, value)

        # Physics loss
        self.physics_loss = self.physics_model.physics_loss
        self.differential_operator = self.physics_model.differential_operator
        self.physics_loss_weight = physics_loss_weight
        
        # Common configs
        self.lr = lr
        self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
        self.mse_loss = nn.MSELoss()
        
        self.save_path = save_path
        self.device = device
        self.physics_model.plot_true_solution(save_path)

        # To device
        # self.X = self.X.to(self.device)
        # self.y = self.y.to(self.device)
        self.X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'solution'], dim=0).to(self.device)
        self.y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'solution'], dim=0).to(self.device)
                           
        self.eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        self.eval_y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        
        self.input_dim = self.X.shape[1]
        self.output_dim = self.y.shape[1]
        # self.physics_X = self.physics_X.to(self.device)
        
        # self._pinn_init()
        # self.net.to(self.device)

    def _pinn_init(self):
        # init pinn net and optimiser
        self.net = BaseDNN(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
        self.net.to(self.device)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
    
        

    def update(self):
        ''' Implement the network parameter update here '''
        raise NotImplementedError()
    
    def train(self, epochs, eval_freq=1000):
        self._pinn_init()
        self.collection = []
        
        losses = []
        
        tic = time.time()
        for ep in range(epochs):
            self.update()
            
            ## 3. Loss calculation
            if (ep+1) % eval_freq == 0:
                toc = time.time()
                loss = self.mse_loss(self.y, self.net(self.X))
                losses.append(loss.item())
                print(f"Epoch {ep+1}/{epochs}, loss: {losses[-1]:.2f}, time: {toc-tic:.2f}s")
                tic = time.time()
                
            if ep > epochs - 1000:
                y_pred = self.evaluate().detach().cpu()
                self.collection.append(y_pred)
        
        self.physics_model.save_evaluation(self, self.save_path)
        return losses
    

    def predict(self, X):
        self.net.eval()
        out = self.net(X)
        return out.detach().cpu().numpy()
    
    def evaluate(self):
        y = self.net(self.eval_X).detach()
        return y
    
    def summary(self):
        y_pred_mat = torch.stack(self.collection, dim=0)
        y_pred_upper = torch.quantile(y_pred_mat, 0.975, dim=0)
        y_pred_lower = torch.quantile(y_pred_mat, 0.025, dim=0)
        y_pred_mean = torch.mean(y_pred_mat, dim=0)
        return y_pred_upper, y_pred_lower, y_pred_mean