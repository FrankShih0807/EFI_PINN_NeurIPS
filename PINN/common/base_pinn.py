import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PINN.common.torch_layers import BaseDNN



class BasePINN(object):
    def __init__(
        self,
        physics_model,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        physics_loss_weight=1,
        save_path=None,
    ) -> None:
        super().__init__()
        self.physics_model = physics_model
        for key, value in self.physics_model.__dict__.items():
            setattr(self, key, value)
        
        # Physics loss
        self.physics_loss = self.physics_model.physics_loss
        self.physics_loss_weight = physics_loss_weight
        
        # Common configs
        self.lr = lr
        self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
        self.mse_loss = nn.MSELoss()
        
        self.collection = []
        self.save_path = save_path
        self.physics_model.plot_true_solution(save_path)

    def _pinn_init(self):
        # init pinn net and optimiser
        self.net = BaseDNN(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
    
    # def save_hyperparameters(self):
        
        

    def update(self):
        ''' Implement the network parameter update here '''
        raise NotImplementedError()
    
    def train(self, epochs):
        self._pinn_init()
        
        losses = []
        for ep in range(epochs):
            self.update()
            
            ## 3. Loss calculation
            if (ep+1) % int(epochs / 10) == 0:
                loss = self.mse_loss(self.y, self.net(self.X))
                losses.append(loss.item())
                print(f"Epoch {ep+1}/{epochs}, loss: {losses[-1]:.2f}")
                
            if ep > epochs - 1000:
                y_pred = self.evaluate()
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