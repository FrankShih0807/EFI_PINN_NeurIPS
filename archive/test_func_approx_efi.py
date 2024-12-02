from PINN.pinn_efi import PINN_EFI
from archive.pinn_dropout import PINN_DROPOUT
from PINN.pinn import PINN
from PINN.pretrain_efi import NONLINEAR_EFI

import torch
from torch.nn.utils import parameters_to_vector
import matplotlib.pyplot as plt
import seaborn as sns
from PINN.models.nonlinear import Nonlinear



if __name__ == '__main__':
    
    # torch.manual_seed(1234)

    t_start = 0
    t_end = 10
    t_extend = 10
    noise_sd = 0.1
    physics_model = Nonlinear(t_start=t_start, t_end=t_end, t_extend=t_extend, noise_sd=noise_sd)
    
    times = torch.linspace(t_start, t_extend, (t_extend - t_start) * 10)
    Y1_true, Y2_true = physics_model.physics_law(times)

    # model = PINN_EFI(physics_model=physics_model, physics_loss_weight=50, lr=1e-4, sgld_lr=1e-4, lambda_y=50, lambda_theta=0.1)
    # model = PINN(physics_model=physics_model, physics_loss_weight=50, lr=1e-3, hidden_layers=[20, 20])
    model = NONLINEAR_EFI(physics_model=physics_model, 
                            physics_loss_weight=50, 
                            lr=1e-4, 
                            hidden_layers=[15, 15],
                            sgld_lr=1e-4, 
                            lambda_y=50, 
                            lambda_theta=1)

    losses = model.train(epochs=10000)

    # theta_vec = parameters_to_vector(model.net.parameters())



    preds_upper, preds_lower, preds_mean = model.summary()
    print(preds_upper.shape, preds_lower.shape, preds_mean.shape)

    Y1_upper = preds_upper[:,0]
    Y1_lower = preds_lower[:,0]
    Y1_mean = preds_mean[:,0]
    
    Y2_upper = preds_upper[:,1]
    Y2_lower = preds_lower[:,1]
    Y2_mean = preds_mean[:,1]


    sns.set_theme()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    
    ax1.plot(times, Y1_true.flatten().numpy(), 'b', label='True Y1')
    ax1.plot(times, Y1_mean, 'g', label='Estimated Y1')
    ax1.fill_between(times, Y1_upper, Y1_lower, alpha=0.2, color='g', label='95% CI')
    ax1.scatter(model.X, model.y[:,0], color='r', label='Data', marker='x')
    ax1.set_ylabel('Y1(t)')
    # ax1.set_ylim(-5, 5)
    ax1.legend()
    
    ax2.plot(times, Y2_true.flatten().numpy(), 'b', label='True Y2')
    ax2.plot(times, Y2_mean, 'g', label='Estimated Y2')
    ax2.fill_between(times, Y2_upper, Y2_lower, alpha=0.2, color='g', label='95% CI')
    ax2.scatter(model.X, model.y[:,1], color='r', label='Data', marker='x')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Y2(t)')
    # ax2.set_ylim(-5, 5)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
