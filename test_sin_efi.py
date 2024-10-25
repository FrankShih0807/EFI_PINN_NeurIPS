from PINN.pinn_efi import PINN_EFI
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PINN.models.sin import Sin



if __name__ == '__main__':
    
    # torch.manual_seed(1234)

    t_end = 20
    t_extend = 25
    noise_sd = 0.1
    physics_model = Sin(t_end=t_end, t_extend=t_extend, noise_sd=noise_sd)
    
    times = torch.linspace(0, t_extend, t_extend * 10)
    Y_true = physics_model.physics_law(times)

    model = PINN_EFI(physics_model=physics_model, physics_loss_weight=50, lr=1e-4, sgld_lr=1e-4, lambda_y=1, lambda_theta=1)

    losses = model.train(epochs=10000)



    preds_upper, preds_lower, preds_mean = model.summary()
    print(preds_upper.shape, preds_lower.shape, preds_mean.shape)

    Y_upper = preds_upper.flatten()
    Y_lower = preds_lower.flatten()
    Y_mean = preds_mean.flatten()


    sns.set_theme()
    
    plt.plot(times, Y_true.flatten().numpy(), 'b', label='True Y')
    plt.plot(times, Y_mean, 'g', label='Estimated Y')
    plt.fill_between(times, Y_upper, Y_lower, alpha=0.2, color='g', label='95% CI')
    plt.scatter(model.X, model.y, color='r', label='Data', marker='x')
    plt.ylabel('Y(t)')
    plt.xlabel('t')
    plt.ylim(-5, 5)
    plt.legend()
    
    plt.show()
