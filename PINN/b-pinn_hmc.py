import torch
import hamiltorch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PINN.common.base_pinn import BasePINN
from PINN.models.poisson import Poisson

class PoissonPINN(torch.nn.Module):
    def __init__(self, width, lam1, lam2):
        super(PoissonPINN, self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )
        self.lam1 = lam1
        self.lam2 = lam2
        
    def forward(self, X):
        x = X[:-2].requires_grad_(True)
        u = self.fnn(x)
        u_x = torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), create_graph = True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs = torch.ones_like(u), create_graph = True)[0]
        u_bd = self.fnn(X[-2:])
        return torch.cat([u_xx * self.lam1 * 0.01, u_bd * self.lam2], dim = 0)

class BayesianPINN(BasePINN):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[50, 50],
        activation_fn=nn.Tanh(),
        lr=1e-3,
        physics_loss_weight=1,
        save_path=None,
        device='cpu',
        sigma=0.01,
        step_size=0.0006,
        burn=2000,
        num_samples=5000,
        L=6
    ) -> None:
        super().__init__(physics_model, dataset, hidden_layers, activation_fn, lr, physics_loss_weight, save_path, device)
        self.sigma = sigma
        self.step_size = step_size
        self.burn = burn
        self.num_samples = num_samples
        self.L = L
        self.tau_list = []

        for d in self.dataset:
            if d['category'] == 'differential':
                self.X = d['X']
                self.y = d['y']

        for d in self.dataset:
            if d['category'] == 'solution':
                self.X = torch.cat([self.X, d['X']], dim=0)
                self.y = torch.cat([self.y, d['y']], dim=0)

    def _pinn_init(self):
        self.net = PoissonPINN(self.hidden_layers[0], self.physics_model.lam1, self.physics_model.lam2)
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
        y_pred_list = []
        for i in range(self.num_samples - self.burn):
            params = hamiltorch.util.unflatten(self.net, self.params_hmc[i])
            hamiltorch.util.update_model_params_in_place(self.net, params)
            y_pred = self.net(X)
            y_pred_list.append(y_pred)
        y_pred = torch.stack(y_pred_list)
        return y_pred

    def evaluate(self):
        y_pred = self.predict(self.eval_X)
        y_mean = torch.mean(y_pred, dim=0)
        y_std = torch.std(y_pred, dim=0)
        y_up, y_low = y_mean - 2 * y_std, y_mean + 2 * y_std
        return y_mean, y_up, y_low

    def summary(self):
        y_pred = self.predict(self.eval_X)
        y_pred_upper = torch.quantile(y_pred, 0.975, dim=0)
        y_pred_lower = torch.quantile(y_pred, 0.025, dim=0)
        y_pred_mean = torch.mean(y_pred, dim=0)
        return y_pred_upper, y_pred_lower, y_pred_mean

    
if __name__ == "__main__":
    hamiltorch.set_random_seed(123)
    torch.manual_seed(123)
    np.random.seed(123)

    # Initialize the physics model
    physics_model = Poisson()
    dataset = physics_model.generate_data(16, device='cpu')

    # Initialize the Bayesian PINN model
    model = BayesianPINN(physics_model, dataset=dataset, device='cpu')

    # Perform HMC sampling
    model.sample_posterior()

    # Evaluate the model
    y_mean, y_up, y_low = model.evaluate()
    y_pred_upper, y_pred_lower, y_pred_mean = model.summary()
    y_pred_upper = y_pred_upper.flatten()
    y_pred_lower = y_pred_lower.flatten()
    y_pred_mean = y_pred_mean.flatten()

    # True solution
    for d in dataset:
        if d['category'] == 'differential':
            X_data = d['X']
            y_data = d['y']

    # evaluate the model
    X_test = model.eval_X.flatten()
    y_test = physics_model.differential_function(X_test)

    # Plot the results
    sns.set_theme()

    plt.plot(X_test.detach().cpu().numpy(), y_mean.detach().cpu().numpy(), label = 'mean')
    plt.plot(X_test.detach().cpu().numpy(), y_up.detach().cpu().numpy())
    plt.plot(X_test.detach().cpu().numpy(), y_low.detach().cpu().numpy())
    # plt.plot(X_test.detach().cpu().numpy(), y_pred_mean.detach().cpu().numpy(), label = 'mean')
    # plt.fill_between(X_test.detach().cpu().numpy(), y_pred_upper.detach().cpu().numpy(), y_pred_lower.detach().cpu().numpy(), alpha=0.2, color='g', label='95% CI')

    plt.plot(X_test.detach().cpu().numpy(), y_test.detach().cpu().numpy(), label = 'True')
    plt.scatter(X_data.detach().cpu().numpy(), y_data.detach().cpu().numpy(), label = 'Data')
    plt.legend()
    plt.ylabel('u_xx')
    plt.xlabel('x')
    plt.show()


    # # Plot the true solution and the predictions
    # physics_model.plot_true_solution()
    
    # preds_upper, preds_lower, preds_mean = model.summary()
    # preds_upper = preds_upper.flatten()
    # preds_lower = preds_lower.flatten()
    # preds_mean = preds_mean.flatten()     

    # X = torch.linspace(-0.7, 0.7, steps=100)
    # y = physics_model.physics_law(X)
    
    # sns.set_theme()
    # plt.plot(X, y, alpha=0.8, color='b', label='Equation')
    # plt.plot(X, preds_mean.detach().numpy(), alpha=0.8, color='g', label='PINN')
    # plt.fill_between(X, preds_upper.detach().numpy(), preds_lower.detach().numpy(), alpha=0.2, color='g', label='95% CI')
    # plt.legend()
    # plt.ylabel('Value')
    # plt.xlabel('X')
    # plt.show()