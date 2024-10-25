from PINN.pinn_efi import PINN_EFI
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PINN.models.sincos import SinCos



if __name__ == '__main__':
    
    # torch.manual_seed(1234)

    t_end = 20
    t_extend = 25
    noise_sd = 0.1
    physics_model = SinCos(t_end=t_end, t_extend=t_extend, noise_sd=noise_sd)
    
    times = torch.linspace(0, t_extend, t_extend * 10)
    Y1_true, Y2_true = physics_model.physics_law(times)

    model = PINN_EFI(physics_model=physics_model, physics_loss_weight=50, lr=1e-4, sgld_lr=1e-4, lambda_y=10, lambda_theta=1)

    losses = model.train(epochs=10000)



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
    ax1.set_ylim(-5, 5)
    ax1.legend()
    
    ax2.plot(times, Y2_true.flatten().numpy(), 'b', label='True Y2')
    ax2.plot(times, Y2_mean, 'g', label='Estimated Y2')
    ax2.fill_between(times, Y2_upper, Y2_lower, alpha=0.2, color='g', label='95% CI')
    ax2.scatter(model.X, model.y[:,1], color='r', label='Data', marker='x')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Y2(t)')
    ax2.set_ylim(-5, 5)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
