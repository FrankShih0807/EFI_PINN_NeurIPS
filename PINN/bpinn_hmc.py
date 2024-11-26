import torch
import hamiltorch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from PINN.common.base_pinn import BasePINN
from PINN.models.poisson import Poisson
from PINN.common.torch_layers import BayesianPINNNet


class BayesianPINN(BasePINN):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[50, 50],
        activation_fn=nn.Tanh(),
        step_size=0.0006,
        burn=2000,
        num_samples=5000,
        L=6, 
        lam1=1.0,
        lam2=1.0,
        save_path=None,
        device='cpu',
    ) -> None:
        super().__init__(physics_model, dataset, hidden_layers, activation_fn, save_path=save_path, device=device)
        
        self.step_size = step_size
        self.burn = burn
        self.num_samples = num_samples
        self.L = L
        self.lam1 = lam1
        self.lam2 = lam2
        self.tau_list = []

        diff_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'differential'], dim=0)
        diff_y = self.lam1 * torch.cat([d['y'] for d in self.dataset if d['category'] == 'differential'], dim=0)

        sol_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'solution'], dim=0)
        sol_y = self.lam2 * torch.cat([d['y'] for d in self.dataset if d['category'] == 'solution'], dim=0)

        eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0)

        self.num_bd = sol_X.shape[0]
        self.X = torch.cat([diff_X, sol_X], dim=0).to(self.device)
        self.y = torch.cat([diff_y, sol_y], dim=0).to(self.device)
        self.eval_X = torch.cat([eval_X, sol_X], dim=0).to(self.device)

    def _pinn_init(self):
        self.net = BayesianPINNNet(self.lam1, self.lam2, self.physics_model, self.num_bd)
        for param in self.net.parameters():
            torch.nn.init.normal_(param)
        self.net.to(self.device)
        # self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        for w in self.net.parameters():
            self.tau_list.append(1.0)
        self.tau_list = torch.tensor(self.tau_list).to(self.device)

    def sample_posterior(self):
        self._pinn_init()
        params_init = hamiltorch.util.flatten(self.net).to(self.device).clone()
        self.params_hmc = hamiltorch.sample_model(
            self.net, self.X, self.y, model_loss='regression', params_init=params_init,
            num_samples=self.num_samples, step_size=self.step_size, burn=self.burn,
            num_steps_per_sample=self.L, tau_list=self.tau_list, tau_out=1
        )

    def predict(self, X):
        # self.net.eval()
        f_pred_list = []
        u_pred_list = []
        for i in range(self.num_samples - self.burn):
            params = hamiltorch.util.unflatten(self.net, self.params_hmc[i])
            hamiltorch.util.update_model_params_in_place(self.net, params)
            f_pred = self.net(X)[:-self.num_bd]
            u_pred = self.net.fnn(X)[:-self.num_bd]
            f_pred_list.append(f_pred)
            u_pred_list.append(u_pred)
        f_pred = torch.stack(f_pred_list).detach().cpu()
        u_pred = torch.stack(u_pred_list).detach().cpu()
        
        return f_pred, u_pred

    # def evaluate(self):
    #     y_pred, u_pred = self.predict(self.eval_X)
    #     y_mean = torch.mean(y_pred, dim=0)
    #     y_std = torch.std(y_pred, dim=0)
    #     y_up, y_low = y_mean - 2 * y_std, y_mean + 2 * y_std
    #     return y_mean, y_up, y_low

    def summary(self):
        f_pred, u_pred = self.predict(self.eval_X)

        f_pred_upper = torch.quantile(f_pred, 0.975, dim=0)
        f_pred_lower = torch.quantile(f_pred, 0.025, dim=0)
        f_pred_mean = torch.mean(f_pred, dim=0)
        f_pred_median = torch.quantile(f_pred, 0.5, dim=0)

        u_pred_upper = torch.quantile(u_pred, 0.975, dim=0)
        u_pred_lower = torch.quantile(u_pred, 0.025, dim=0)
        u_pred_mean = torch.mean(u_pred, dim=0)
        u_pred_median = torch.quantile(u_pred, 0.5, dim=0)
        u_covered = (u_pred_lower <= self.eval_y.clone().detach().cpu()) & (self.eval_y.clone().detach().cpu() <= u_pred_upper)

        summary_dict = {
            'f_preds_upper': f_pred_upper,
            'f_preds_lower': f_pred_lower,
            'f_preds_mean': f_pred_mean,
            'f_preds_median': f_pred_median,
            'y_preds_upper': u_pred_upper,
            'y_preds_lower': u_pred_lower,
            'y_preds_mean': u_pred_mean,
            'y_preds_median': u_pred_median,
            'y_covered': u_covered, 
            'x_eval': self.eval_X.clone().detach().cpu().numpy(),
        }
        

        return summary_dict
    
    def train(self, epochs):
        # self._pinn_init()
        
        tic = time.time()
        self.sample_posterior()
        toc = time.time()
        print(f"Sampling time: {toc-tic:.2f}s")
        
        self.physics_model.save_evaluation(self, self.save_path)

    
if __name__ == "__main__":
    # hamiltorch.set_random_seed(123)
    # torch.manual_seed(123)
    # np.random.seed(123)

    # Initialize the physics model
    physics_model = Poisson(boundary_sd=0.05, diff_sd=0.0)
    dataset = physics_model.generate_data(100, device='cpu')

    # Initialize the Bayesian PINN model
    model = BayesianPINN(physics_model, dataset=dataset, device='cpu', step_size=0.0002, lam1=100.0, lam2=30.0)
    num_bd = model.num_bd

    # Perform HMC sampling
    model.sample_posterior()

    # Evaluate the model
    pred_dict = model.summary()

    pred_upper = pred_dict['y_preds_upper'].flatten()
    pred_lower = pred_dict['y_preds_lower'].flatten()
    pred_mean = pred_dict['y_preds_mean'].flatten()

    # evaluate the model
    X_test = model.eval_X.flatten()[:-num_bd]
    u_test = physics_model.physics_law(X_test)

    # Plot the results
    sns.set_theme()

    plt.plot(X_test.detach().cpu().numpy(), pred_mean.detach().cpu().numpy(), label = 'mean')
    plt.fill_between(X_test.detach().cpu().numpy(), pred_upper.detach().cpu().numpy(), pred_lower.detach().cpu().numpy(), alpha=0.2, color='g', label='95% CI')
    plt.plot(X_test.detach().cpu().numpy(), u_test.detach().cpu().numpy(), label = 'True')
    plt.legend()
    plt.ylabel('u')
    plt.xlabel('x')
    plt.show()
