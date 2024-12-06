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
        # burn=2000,
        num_samples=5000,
        L=6, 
        lam_diff=1.0,
        lam_sol=1.0,
        save_path=None,
        device='cpu',
    ) -> None:
        self.step_size = step_size
        # self.burn = burn
        self.num_samples = num_samples
        self.L = L
        self.lam_diff = lam_diff
        self.lam_sol = lam_sol
        # self.tau_list = []

        self.dataset = dataset.copy()

        diff_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'differential'], dim=0)
        diff_y = self.lam_diff * torch.cat([d['y'] for d in self.dataset if d['category'] == 'differential'], dim=0)

        sol_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'solution'], dim=0)
        sol_y = self.lam_sol * torch.cat([d['y'] for d in self.dataset if d['category'] == 'solution'], dim=0)

        eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0)

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
        # self.eval_X = torch.cat([eval_X, sol_X], dim=0).to(self.device)

        self.mse_loss = nn.MSELoss(reduction="sum")

    def _pinn_init(self):
        self.bnet = BayesianPINNNet(self.lam_diff, self.lam_sol, self.physics_model, self.num_bd)
        # self.net = BayesianPINNNet(self.lam_diff, self.lam_sol, self.physics_model, self.num_bd)
        self.net = self.bnet.fnn
        for param in self.net.parameters():
            torch.nn.init.normal_(param)
        self.bnet.to(self.device)
        self.net.to(self.device)
        # self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        self.tau_list = []
        for w in self.net.parameters():
            self.tau_list.append(1.0)
        self.tau_list = torch.tensor(self.tau_list).to(self.device)

        self.params_init = hamiltorch.util.flatten(self.net).to(self.device).clone()
        self.params_hmc = []

    def sample_posterior(self, num_samples):
        # self._pinn_init()

        # params_init = hamiltorch.util.flatten(self.net).to(self.device).clone()
        # self.params_hmc = hamiltorch.sample_model(
        #     self.net, self.X, self.y, model_loss='regression', params_init=params_init,
        #     num_samples=self.num_samples, step_size=self.step_size, burn=self.burn,
        #     num_steps_per_sample=self.L, tau_list=self.tau_list, tau_out=1
        # )

        # params_hmc = hamiltorch.sample_model(
        #     self.net, self.X, self.y, model_loss='regression', params_init=self.params_init,
        #     num_samples=num_samples, step_size=self.step_size, burn=0,
        #     num_steps_per_sample=self.L, tau_list=self.tau_list, tau_out=1, verbose=False
        # )

        params_hmc = hamiltorch.sample_model(
            self.bnet, self.X, self.y, model_loss='regression', params_init=self.params_init,
            num_samples=num_samples, step_size=self.step_size, burn=0,
            num_steps_per_sample=self.L, tau_list=self.tau_list, tau_out=1, verbose=False
        )

        return params_hmc

    # def predict(self, X):
    #     # self.net.eval()
    #     f_pred_list = []
    #     u_pred_list = []
    #     for i in range(self.num_samples - self.burn):
    #         params = hamiltorch.util.unflatten(self.net, self.params_hmc[i])
    #         hamiltorch.util.update_model_params_in_place(self.net, params)
    #         f_pred = self.net(X)[:-self.num_bd]
    #         u_pred = self.net.fnn(X)[:-self.num_bd]
    #         f_pred_list.append(f_pred)
    #         u_pred_list.append(u_pred)
    #     f_pred = torch.stack(f_pred_list).detach().cpu()
    #     u_pred = torch.stack(u_pred_list).detach().cpu()

    #     return f_pred, u_pred

    # def evaluate(self):
    #     y_pred, u_pred = self.predict(self.eval_X)
    #     y_mean = torch.mean(y_pred, dim=0)
    #     y_std = torch.std(y_pred, dim=0)
    #     y_up, y_low = y_mean - 2 * y_std, y_mean + 2 * y_std
    #     return y_mean, y_up, y_low

    # def summary(self):
    #     f_pred, u_pred = self.predict(self.eval_X)

    #     f_pred_upper = torch.quantile(f_pred, 0.975, dim=0)
    #     f_pred_lower = torch.quantile(f_pred, 0.025, dim=0)
    #     f_pred_mean = torch.mean(f_pred, dim=0)
    #     f_pred_median = torch.quantile(f_pred, 0.5, dim=0)

    #     u_pred_upper = torch.quantile(u_pred, 0.975, dim=0)
    #     u_pred_lower = torch.quantile(u_pred, 0.025, dim=0)
    #     u_pred_mean = torch.mean(u_pred, dim=0)
    #     u_pred_median = torch.quantile(u_pred, 0.5, dim=0)
    #     u_covered = (u_pred_lower <= self.eval_y.clone().detach().cpu()) & (self.eval_y.clone().detach().cpu() <= u_pred_upper)

    #     summary_dict = {
    #         'f_preds_upper': f_pred_upper,
    #         'f_preds_lower': f_pred_lower,
    #         'f_preds_mean': f_pred_mean,
    #         'f_preds_median': f_pred_median,
    #         'y_preds_upper': u_pred_upper,
    #         'y_preds_lower': u_pred_lower,
    #         'y_preds_mean': u_pred_mean,
    #         'y_preds_median': u_pred_median,
    #         'y_covered': u_covered, 
    #         'x_eval': self.eval_X.clone().detach().cpu().numpy(),
    #     }

    #     return summary_dict

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
            params_hmc = self.sample_posterior(num_samples=1)
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
                # ##########
                # params_hmc = self.sample_posterior(num_samples=eval_freq)
                # self.params_hmc += params_hmc
                # self.params_init = self.params_hmc[-1].clone()
                # for i in range(eval_freq):
                #     hamiltorch.util.update_model_params_in_place(self.net, hamiltorch.util.unflatten(self.net, params_hmc[i]))
                #     self.callback.on_training()


                self.callback.on_eval()
                self.logger.dump()

        # for rd in range(int(epochs / eval_freq)):
        #     self.progress = (rd+1) / int(epochs / eval_freq)
        #     tic = time.time()
        #     params_hmc = self.sample_posterior(num_samples=eval_freq)
        #     toc = time.time()
        #     print(f"Sampling time: {toc-tic:.2f}s")
        #     print(f"Progress: {self.progress:.2f}")

        #     self.params_hmc += params_hmc
        #     self.params_init = self.params_hmc[-1].clone()

        #     hamiltorch.util.update_model_params_in_place(self.net, hamiltorch.util.unflatten(self.net, self.params_hmc[-1]))
        #     eval_loss = self.mse_loss(self.net.fnn(self.eval_X)[:-self.num_bd], self.eval_y).item()
        #     print(f"Eval loss: {eval_loss:.4f}")
        #     eval_losses.append(eval_loss)

        # tic = time.time()
        # self.sample_posterior()
        # toc = time.time()
        # print(f"Sampling time: {toc-tic:.2f}s")

        # self.physics_model.save_evaluation(self, self.save_path)
        # self.physics_model.create_gif(self.save_path)
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
