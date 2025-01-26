import torch
import hamiltorch
import torch.nn as nn
import time

from PINN import util_bpinn
from PINN.common.base_pinn import BasePINN
from PINN.common.torch_layers import BayesianNet
from PINN.common.scheduler import get_schedule


class BayesianPINN(BasePINN):
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

        self.data_list = {}

        self.num_bd = self.sol_X.shape[0]
        
        self.X = torch.cat([self.diff_X, self.sol_X], dim=0).to(self.device)

        self.mse_loss = nn.MSELoss(reduction="sum")

    def _pinn_init(self):
        self.net = BayesianNet(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers)
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
        
        self.params_hmc = []

    def _get_scheduler(self):
        self.step_size = get_schedule(self.step_size)
        self.sigma_diff = get_schedule(self.sigma_diff)
        self.sigma_sol = get_schedule(self.sigma_sol)

    def model_loss(self, data, fmodel, params_unflattened, tau_likes, gradients, params_single=None):
        sol_X = self.sol_X
        sol_y = self.sol_y
        diff_X = self.diff_X
        diff_y = self.diff_y

        pred_sol = fmodel[0](sol_X, params=params_unflattened[0])
        ll = - 0.5 * tau_likes[0] * ((pred_sol - sol_y) ** 2).sum(0)

        def net_within_loss(X):
            return fmodel[0](X, params=params_unflattened[0])
        
        pred_diff = self.differential_operator(net_within_loss, diff_X, params_single)
        ll = ll - 0.5 * tau_likes[1] * ((pred_diff - diff_y) ** 2).sum(0)
        output = [pred_sol, pred_diff]

        return ll, output

    def sample_posterior(self, num_samples):
        # update training parameters
        annealing_period = 0.3
        annealing_progress = self.progress / annealing_period
        step_size = self.step_size(annealing_progress)
        sigma_diff = self.sigma_diff(annealing_progress)
        sigma_sol = self.sigma_sol(annealing_progress)

        self.y = torch.cat([self.diff_y / (sigma_diff * 2 ** 0.5), self.sol_y / (sigma_sol * 2 ** 0.5)], dim=0).to(self.device)

        params_hmc = util_bpinn.sample_model_bpinns(
            [self.net], self.data_list, model_loss=self.model_loss, num_samples=num_samples, 
            num_steps_per_sample=self.L, step_size=step_size, burn=0, tau_priors=1.0, 
            tau_likes=[1 / sigma_sol ** 2, 1 / sigma_diff ** 2], device=self.device, 
            n_params_single=(self.pe_dim if self.pe_dim > 0 else None), pde=True, pinns=False, 
            params_init_val=self.params_init,
        )

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

            self.params_hmc += params_hmc
            if self.pe_dim > 0:
                self.pe_variables = self.params_hmc[-1][:self.pe_dim].clone()
            self.params_init = self.params_hmc[-1].clone()

            params = hamiltorch.util.unflatten(self.net, self.params_hmc[-1][self.pe_dim:])
            hamiltorch.util.update_model_params_in_place(self.net, params)

            self.callback.on_training()

            self.logger.record('train/progress', self.progress)
            self.logger.record('train/epoch', ep+1)
            self.logger.record('train/time', toc-tic)

            if (ep+1) % eval_freq == 0:
                self.callback.on_eval()
                self.logger.dump()

        self.callback.on_training_end()

