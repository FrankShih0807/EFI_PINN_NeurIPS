import torch
import torch.nn as nn
import torch.nn.functional as F
from PINN.common.base_pinn import BasePINN
from PINN.common.torch_layers import BaseDNN
from PINN.common.torch_layers import BayesHypernet
from torch import optim
import torch.distributions as dist
import numpy as np
from PINN.common.scheduler import get_schedule
from torch.nn.utils import parameters_to_vector

class BayesHyperPINN(BasePINN):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        units=[16, 32, 64],
        annealing_period=0.1,
        lr=1e-3,
        lambda_pde=1,
        positive_output=False,
        save_path=None,
        device='cpu'
    ) -> None:

        self.units = units
        self.positive_output = positive_output
        self.pe_dim = physics_model.pe_dim
        
        super().__init__(
            physics_model=physics_model,
            dataset=dataset,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            lr=lr,
            lambda_pde=lambda_pde,
            save_path=save_path,
            device=device,
        )
        self.annealing_period = annealing_period
        
    def _get_scheduler(self):
        # self.lr = get_schedule(self.lr)
        # self.sgd_momentum = get_schedule(self.sgd_momentum)
        # self.sgld_lr = get_schedule(self.sgld_lr)
        # self.sgld_alpha = get_schedule(self.sgld_alpha)
        self.lambda_pde = get_schedule(self.lambda_pde)
        # self.lam = get_schedule(self.lam)
        # self.lambda_theta = get_schedule(self.lambda_theta)
        
    def _pinn_init(self):
        # init pinn net and optimiser
        self.net = BayesHypernet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
            # activation_fn=self.activation_fn,
            units=self.units,
            # positive_output=self.positive_output,
        )
        self.net.to(self.device)
        if self.pe_dim > 0:
            self.pe_variables = torch.randn(self.pe_dim, requires_grad=True, device=self.device)
            self.optimiser = optim.Adam(list(self.net.parameters()) + [self.pe_variables], lr=self.lr)
        else:
            self.pe_variables = None
            self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)
        # self.crit = lambda x, y: torch.sum(-1 * dist.Normal(0., 9.).log_prob(x - y))
        
    def pde_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'differential':
                diff_o = self.differential_operator(self.net, d['X'], pe_variables = self.pe_variables)
                loss += self.mse_loss(diff_o, d['y'])
        return loss

    def solution_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'solution':
                loss += self.mse_loss(d['y'], self.net(d['X']))
        return loss


    def optimize_encoder(self, steps=1000):
        base_net = BaseDNN(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
        ).to(self.device)
        base_net.eval()
        optimiser = optim.Adam(self.net.parameters(), lr=3e-4)
        # optimiser = optim.SGD(self.net.parameters(), lr=1e-3)
        print('Pretraining BayesHyper...')

        for _ in range(steps):
            self.net.train()
            
            self.net.encode_weights()
            
            loss = 0
            for p, q in zip(self.net.layers, base_net.parameters()):
                loss += F.mse_loss(p, q, reduction="sum")

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        print('BayesHyper pretraining done.')

    def update(self):
        annealing_progress = self.progress / self.annealing_period
        lambda_pde = self.lambda_pde(annealing_progress)
        
        self.net.encode_weights()
        self.optimiser.zero_grad()
        cur_anneal = np.clip(1 - 10 * self.progress , 0., 1.0)
        # print(cur_anneal)
        kl = self.net.kl()
        
        sol_loss = self.solution_loss()
        pde_loss = self.pde_loss()
        loss = sol_loss + lambda_pde * pde_loss
        # loss = sol_loss + lambda_pde * pde_loss + cur_anneal * kl

        loss.backward()
        self.optimiser.step()

        self.logger.record('train_param/lambda_pde', lambda_pde, exclude='csv')

        return sol_loss.item(), pde_loss.item()
    
    def train(self, epochs=10000, eval_freq=-1, burn=0.1, callback=None):
        # base_net = BaseDNN(
        #     input_dim=self.input_dim,
        #     output_dim=self.output_dim,
        #     hidden_layers=self.hidden_layers,
        #     activation_fn=self.activation_fn,
        # ).to(self.device)
        # base_net.eval()
        
        # Convert BaseDNN parameters to vector
        # param_vector = parameters_to_vector(base_net.parameters()).to(self.device)

        # Optimize encoder network
        self.optimize_encoder()
        super().train(epochs, eval_freq, burn, callback)
