import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import seaborn as sns
from PINN.common import SGLD
from PINN.common.torch_layers import EFI_Net
from PINN.common.base_pinn import BasePINN
from PINN.models.european_call import EuropeanCall


class PINN_EFI(BasePINN):
    def __init__(
        self,
        physics_model,
        hidden_layers=[15, 15],
        lr=1e-3,
        physics_loss_weight=10,
        sgld_lr=1e-3,
        lambda_y=1,
        lambda_theta=1,
    ) -> None:
        super().__init__(physics_model, hidden_layers, lr, physics_loss_weight)
        
        # EFI configs
        self.sgld_lr = sgld_lr
        self.lambda_y = lambda_y
        self.lambda_theta = lambda_theta
        
        self.noise_sd = physics_model.noise_sd

    
    def _pinn_init(self):
        # init EFI net and optimiser
        self.net = EFI_Net(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
        
        # init latent noise and sampler
        self.Z = torch.randn_like(self.y).requires_grad_()
        self.sampler = SGLD([self.Z], self.sgld_lr)
        

    def update(self):
        ## 1. Latent variable sampling (Sample Z)
        self.net.eval()
        theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        Z_loss = self.lambda_y * y_loss + self.lambda_theta * theta_loss + torch.mean(self.Z**2)/2/self.noise_sd**2
        

        self.sampler.zero_grad()
        Z_loss.backward()
        self.sampler.step()

        ## 2. DNN weights update (Optimize W)
        
        self.net.train()
        theta_loss = self.net.theta_encode(self.X, self.y, self.Z)
        y_loss = self.mse_loss(self.y, self.net(self.X) + self.Z)
        prior_loss = - self.net.gmm_prior_loss() / self.n_samples
        
        w_loss = self.lambda_y * (y_loss + prior_loss) + self.physics_loss_weight * self.physics_loss(self.net) + self.lambda_theta * theta_loss 

        self.optimiser.zero_grad()
        w_loss.backward()
        self.optimiser.step()
    
    def evaluate(self):
        y = self.net(self.physics_X).detach()
        return y


if __name__ == '__main__':
    sns.set_theme()
    torch.manual_seed(1234)


    physics_model = EuropeanCall()
    

    pinn_efi = PINN_EFI(physics_model=physics_model, physics_loss_weight=50, lr=1e-5, sgld_lr=1e-4, lambda_y=1, lambda_theta=10)

    losses = pinn_efi.train(epochs=10000)



    # preds = pinn_efi.predict(times.reshape(-1,1))
    preds_upper, preds_lower, preds_mean = pinn_efi.summary()
    preds_upper = preds_upper.flatten()
    preds_lower = preds_lower.flatten()
    preds_mean = preds_mean.flatten()
    # print(preds.shape)
    
    

