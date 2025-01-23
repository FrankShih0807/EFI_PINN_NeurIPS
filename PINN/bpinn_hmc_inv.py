import torch
import hamiltorch
import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import time

from PINN import util_bpinn

from PINN.common.base_pinn import BasePINN
# from PINN.models.poisson import Poisson, PoissonCallback
from PINN.common.torch_layers import BayesianNet
from PINN.common.scheduler import get_schedule


class BayesianPINN_Inverse(BasePINN):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[50, 50],
        activation_fn=nn.Tanh(),
        step_size=0.0002,
        L=6, 
        sigma_diff=0.01,
        sigma_sol=0.01,
        pretrain_epochs=0,
        save_path=None,
        device='cpu',
    ) -> None:
        self.step_size = step_size
        self.L = L
        self.sigma_diff = sigma_diff
        self.sigma_sol = sigma_sol
        self.pretrain_epochs = pretrain_epochs

        self.pe_dim = physics_model.pe_dim

        super().__init__(
            physics_model=physics_model,
            dataset=dataset,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            save_path=save_path,
            device=device,
        )

        ###### something not beautiful here ######
        self.data_list = {}
        self.data_list['x_u'] = self.sol_X
        self.data_list['y_u'] = self.sol_y
        self.data_list['x_f'] = self.diff_X
        self.data_list['y_f'] = self.diff_y

        # self.data_eval_list = {}
        # self.data_eval_list['x_u'] = self.eval_X
        # self.data_eval_list['y_u'] = self.eval_y
        # self.data_eval_list['x_f'] = self.eval_X
        # self.data_eval_list['y_f'] = self.

        self.num_bd = self.sol_X.shape[0]
        
        self.X = torch.cat([self.diff_X, self.sol_X], dim=0).to(self.device)

        self.mse_loss = nn.MSELoss(reduction="sum")

    def _pinn_init(self):
        sigma_diff = self.sigma_diff(progress=0.0)
        sigma_sol = self.sigma_sol(progress=0.0)
        self.net = BayesianNet(hidden_layers=self.hidden_layers)
        for param in self.net.parameters():
            torch.nn.init.normal_(param)
        self.net.to(self.device)
        self.tau_list = []
        for w in self.net.parameters():
            self.tau_list.append(1.0)
        self.tau_list = torch.tensor(self.tau_list).to(self.device)

        self.net_params_init = hamiltorch.util.flatten(self.net).to(self.device).clone()
        if self.pe_dim > 0:
            self.pe_variables = torch.zeros(self.pe_dim)
            self.params_init = torch.cat([self.pe_variables, self.net_params_init])
        else:
            self.pe_variables = None
            self.params_init = self.net_params_init
        # self.pe_variables = torch.zeros(self.pe_dim)
        # self.params_init = torch.cat([self.pe_variables, self.net_params_init])

        # print(self.params_init.shape)
        # raise
        
        self.params_hmc = []

    def _get_scheduler(self):
        self.step_size = get_schedule(self.step_size)
        self.sigma_diff = get_schedule(self.sigma_diff)
        self.sigma_sol = get_schedule(self.sigma_sol)

    def model_loss(self, data, fmodel, params_unflattened, tau_likes, gradients, params_single=None):
        x_u = data['x_u']
        y_u = data['y_u']
        pred_u = fmodel[0](x_u, params=params_unflattened[0])
        ll = - 0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
        x_f = data['x_f']
        x_f = x_f.detach().requires_grad_()
        u = fmodel[0](x_f, params=params_unflattened[0])
        u_x = gradients(u,x_f)[0]
        u_xx = gradients(u_x,x_f)[0]
        pred_f = 0.01*u_xx + params_single[0]*torch.tanh(u)
        y_f = data['y_f']
        ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
        output = [pred_u,pred_f]

        # if torch.cuda.is_available():
        #     del x_u, y_u, x_f, y_f, u, u_x, u_xx, pred_u, pred_f
        #     torch.cuda.empty_cache()

        return ll, output

    def sample_posterior(self, num_samples):
        # update training parameters
        annealing_period = 0.3
        annealing_progress = self.progress / annealing_period
        step_size = self.step_size(annealing_progress)
        sigma_diff = self.sigma_diff(annealing_progress)
        sigma_sol = self.sigma_sol(annealing_progress)
        # step_size = 0.02 * min(sigma_diff, self.sigma_sol)

        # self.bnet.sigma_diff = sigma_diff
        self.y = torch.cat([self.diff_y / (sigma_diff * 2 ** 0.5), self.sol_y / (sigma_sol * 2 ** 0.5)], dim=0).to(self.device)

        params_hmc = util_bpinn.sample_model_bpinns(
            [self.net], self.data_list, model_loss=self.model_loss, num_samples=num_samples, 
            num_steps_per_sample=self.L, step_size=step_size, burn=0, tau_priors=1.0, 
            tau_likes=[1 / sigma_sol ** 2, 1 / sigma_diff ** 2], device=self.device, 
            n_params_single=(self.pe_dim if self.pe_dim > 0 else None), pde=True, pinns=False, 
            params_init_val=self.params_init,
        )


        # params_hmc = hamiltorch.sample_model(
        #     self.bnet, self.X, self.y, model_loss='regression', params_init=self.params_init,
        #     num_samples=num_samples, step_size=step_size, burn=0,
        #     num_steps_per_sample=self.L, tau_list=self.tau_list, tau_out=1, verbose=False
        # )

        return params_hmc

    def train(self, epochs, eval_freq=-1, burn=0.5, callback=None):
        self.epochs = epochs
        if eval_freq == -1:
            eval_freq = epochs // 10
        self.callback = callback
        self.callback.init_callback(self, eval_freq=eval_freq, burn=burn)
        self.n_eval = 0

        for ep in range(epochs):
            self.progress = (ep+1) / epochs
            tic = time.time()
            params_hmc = self.sample_posterior(num_samples=2)
            toc = time.time()
            
            # print(len(params_hmc))
            # print(params_hmc[0].shape)
            # raise

            self.params_hmc += params_hmc
            if self.pe_dim > 0:
                self.pe_variables = self.params_hmc[-1][:self.pe_dim].clone()
            self.params_init = self.params_hmc[-1].clone()

            # print(self.params_hmc[-1].shape)
            # print(self.params_hmc[-1][self.pe_dim:].shape)
            # print(self.params_hmc[-1][:self.pe_dim])
            # raise

            params = hamiltorch.util.unflatten(self.net, self.params_hmc[-1][self.pe_dim:])
            hamiltorch.util.update_model_params_in_place(self.net, params)

            self.callback.on_training()

            self.logger.record('train/progress', self.progress)
            self.logger.record('train/epoch', ep+1)
            self.logger.record('train/time', toc-tic)

            if (ep+1) % eval_freq == 0:

                # print(self.pe_variables)

                self.callback.on_eval()
                self.logger.dump()

        self.callback.on_training_end()


# if __name__ == "__main__":
#     # hamiltorch.set_random_seed(123)
#     # torch.manual_seed(123)
#     # np.random.seed(123)

#     # Initialize the physics model
#     physics_model = Poisson(sol_sd=0.05, diff_sd=0.0)
#     dataset = physics_model.generate_data(device='cpu')

#     # Initialize the Bayesian PINN model
#     model = BayesianPINN_Inverse(physics_model, dataset=dataset, device='cpu', step_size=0.0002, lam_diff=100.0, lam_sol=30.0, num_samples=100)
#     num_bd = model.num_bd

#     # Perform HMC sampling
#     hmc_params = model.sample_posterior(num_samples=10)
#     print(len(hmc_params))
#     print(hmc_params[0].shape)
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
