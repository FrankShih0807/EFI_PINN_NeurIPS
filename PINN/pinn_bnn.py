import torch
import torch.optim as optim
import torchbnn as bnn
from PINN.common.base_pinn import BasePINN
from PINN.common.torch_layers import BayesianNN
from PINN.models.cooling import Cooling
import seaborn as sns
import matplotlib.pyplot as plt

class PINN_BNN(BasePINN):
    def __init__(
        self,
        physics_model,
        hidden_layers=[15, 15],
        lr=1e-3,
        physics_loss_weight=10,
    ) -> None:
        super().__init__(physics_model, hidden_layers, lr, physics_loss_weight)
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = 1e-1
        

    def _pinn_init(self):
        self.net = BayesianNN(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=self.hidden_layers, activation_fn=self.activation_fn)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.lr)
    
    def update(self):
        self.optimiser.zero_grad()
        outputs = self.net(self.X)
        loss = self.mse_loss(self.y, outputs) + self.kl_weight * self.kl_loss(self.net)
        loss += self.physics_loss_weight * self.physics_loss(self.net)
        
        loss.backward()
        self.optimiser.step()
        
        return loss


if __name__ == '__main__':
    sns.set_theme()
    torch.manual_seed(1234)
    
    Tenv = 25
    T0 = 100
    R = 0.005
    t_end = 300
    t_extend = 1500
    physics_model = Cooling(Tenv=Tenv, T0=T0, R=R, t_end=t_end, t_extend=t_extend)
    
    times = torch.linspace(0, t_extend, t_extend)
    temps = physics_model.physics_law(times)

    pinn = PINN_BNN(physics_model=physics_model, physics_loss_weight=10, lr=1e-3)

    losses = pinn.train(epochs=30000)



    # preds = pinn_efi.predict(times.reshape(-1,1))
    preds_upper, preds_lower, preds_mean = pinn.summary()
    preds_upper = preds_upper.flatten()
    preds_lower = preds_lower.flatten()
    preds_mean = preds_mean.flatten()
    # print(preds.shape)

    plt.plot(times, temps, alpha=0.8, color='b', label='Equation')
    # plt.plot(t, T, 'o')
    plt.plot(times, preds_mean, alpha=0.8, color='g', label='PINN-BNN')
    plt.vlines(t_end, Tenv, T0, color='r', linestyles='dashed', label='no data beyond this point')
    plt.fill_between(times, preds_upper, preds_lower, alpha=0.2, color='g', label='95% CI')
    plt.legend()
    plt.ylabel('Temperature (C)')
    plt.xlabel('Time (s)')
    plt.savefig('temp_pred_bnn.png')