import torch
import hamiltorch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from PINN.common.base_pinn import BasePINN
from PINN.models.poisson import Poisson, PoissonCallback
from PINN.common.torch_layers import BayesianPINNNet


class BayesianPINN(BasePINN):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[50, 50],
        activation_fn=nn.Tanh(),
        step_size=0.02,
        L=6, 
        sigma_diff=0.01,
        sigma_sol=0.01,
        save_path=None,
        device='cpu',
    ) -> None:
        self.step_size = step_size
        self.L = L
        self.sigma_diff = sigma_diff
        self.sigma_sol = sigma_sol

        self.dataset = dataset.copy()

        diff_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'differential'], dim=0)
        diff_y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'differential'], dim=0) / self.sigma_diff

        sol_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'solution'], dim=0)
        sol_y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'solution'], dim=0) / self.sigma_sol

        self.num_bd = sol_X.shape[0]

        super().__init__(
            physics_model=physics_model,
            dataset=dataset,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            save_path=save_path,
            device=device,
        )
        
        self.X = torch.cat([diff_X, sol_X], dim=0).to(self.device)
        self.y = torch.cat([diff_y, sol_y], dim=0).to(self.device)

        self.mse_loss = nn.MSELoss(reduction="sum")

    def _pinn_init(self):
        self.bnet = BayesianPINNNet(self.sigma_diff, self.sigma_sol, self.physics_model, self.num_bd)
        self.net = self.bnet.fnn
        for param in self.net.parameters():
            torch.nn.init.normal_(param)
        self.bnet.to(self.device)
        self.net.to(self.device)
        self.tau_list = []
        for w in self.net.parameters():
            self.tau_list.append(1.0)
        self.tau_list = torch.tensor(self.tau_list).to(self.device)

        self.params_init = hamiltorch.util.flatten(self.net).to(self.device).clone()
        self.params_hmc = []

    def sample_posterior(self, num_samples):
        params_hmc = hamiltorch.sample_model(
            self.bnet, self.X, self.y, model_loss='regression', params_init=self.params_init,
            num_samples=num_samples, step_size=self.step_size, burn=0,
            num_steps_per_sample=self.L, tau_list=self.tau_list, tau_out=1, verbose=False
        )

        return params_hmc

    def train(self, epochs, eval_freq=-1, burn=0.5, callback=None):
        self.epochs = epochs
        if eval_freq == -1:
            eval_freq = epochs // 10
        # self.eval_buffer = EvaluationBuffer(burn=burn)
        # self.burn_steps = int(epochs * burn)
        self.callback = callback
        self.callback.init_callback(self, eval_freq=eval_freq, burn=burn)
        self.n_eval = 0

        for ep in range(epochs):
            self.progress = (ep+1) / epochs
            tic = time.time()
            params_hmc = self.sample_posterior(num_samples=2)
            toc = time.time()
            
            self.params_hmc += params_hmc
            self.params_init = self.params_hmc[-1].clone()
            params = hamiltorch.util.unflatten(self.net, self.params_hmc[-1])
            hamiltorch.util.update_model_params_in_place(self.net, params)

            self.callback.on_training()

            self.logger.record('train/progress', self.progress)
            self.logger.record('train/epoch', ep+1)
            self.logger.record('train/time', toc-tic)

            if (ep+1) % eval_freq == 0:
                self.callback.on_eval()
                self.logger.dump()

        self.callback.on_training_end()


if __name__ == "__main__":
    # hamiltorch.set_random_seed(123)
    # torch.manual_seed(123)
    # np.random.seed(123)

    # Initialize the physics model
    physics_model = Poisson(sol_sd=0.05, diff_sd=0.0)
    dataset = physics_model.generate_data(device='cpu')

    # Initialize the Bayesian PINN model
    model = BayesianPINN(physics_model, dataset=dataset, device='cpu', step_size=0.0002, lam_diff=100.0, lam_sol=30.0, num_samples=100)
    num_bd = model.num_bd

    # Perform HMC sampling
    hmc_params = model.sample_posterior(num_samples=10)
    print(len(hmc_params))
    print(hmc_params[0].shape)
    # model.train(epochs=5000, eval_freq=500, burn=0.5, callback=PoissonCallback())

    # # Evaluate the model
    # pred_dict = model.summary()

    # pred_upper = pred_dict['y_preds_upper'].flatten()
    # pred_lower = pred_dict['y_preds_lower'].flatten()
    # pred_mean = pred_dict['y_preds_mean'].flatten()

    # # evaluate the model
    # X_test = model.eval_X.flatten()[:-num_bd]
    # u_test = physics_model.physics_law(X_test)

    # # Plot the results
    # sns.set_theme()

    # plt.plot(X_test.detach().cpu().numpy(), pred_mean.detach().cpu().numpy(), label = 'mean')
    # plt.fill_between(X_test.detach().cpu().numpy(), pred_upper.detach().cpu().numpy(), pred_lower.detach().cpu().numpy(), alpha=0.2, color='g', label='95% CI')
    # plt.plot(X_test.detach().cpu().numpy(), u_test.detach().cpu().numpy(), label = 'True')
    # plt.legend()
    # plt.ylabel('u')
    # plt.xlabel('x')
    # plt.show()
