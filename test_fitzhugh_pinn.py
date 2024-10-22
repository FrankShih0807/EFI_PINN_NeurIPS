from PINN.pinn import PINN
from PINN.models.fitzhugh_nagumo import FitzHugh_Nagumo
import torch
import matplotlib.pyplot as plt
import seaborn as sns




if __name__ == '__main__':
    
    physics_model = FitzHugh_Nagumo()
    model = PINN(physics_model=physics_model, physics_loss_weight=1, lr=1e-3, hidden_layers=[50, 50, 50, 50])
    
    # for key, value in model.__dict__.items():
    #     print('{}: {}'.format(key, value))
        
        
    ts = torch.linspace(0, physics_model.t_extend, 1000)
    
    losses = model.train(epochs=30000, eval_x=ts.view(-1,1))
    
    preds_upper, preds_lower, preds_mean = model.summary()
    print(preds_upper.shape, preds_lower.shape, preds_mean.shape)
    
    V_upper = preds_upper[:,0]
    V_lower = preds_lower[:,0]
    V_mean = preds_mean[:,0]
    
    R_upper = preds_upper[:,1]
    R_lower = preds_lower[:,1]
    R_mean = preds_mean[:,1]
    
    V_true, R_true = physics_model.physics_law(ts, -1.0, 1.0)
    
    


    sns.set_theme()
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    
    ax1.plot(ts, V_true.flatten().numpy(), 'b', label='True V')
    ax1.plot(ts, V_mean, 'g', label='Estimated V')
    ax1.fill_between(ts, V_upper, V_lower, alpha=0.2, color='g', label='95% CI')
    ax1.scatter(model.X, model.y[:,0], color='r', label='Data', marker='x')
    ax1.set_ylabel('V(t)')
    ax1.set_ylim(-2, 4)
    ax1.legend()
    
    ax2.plot(ts, R_true.flatten().numpy(), 'b', label='True R')
    ax2.plot(ts, R_mean, 'g', label='Estimated R')
    ax2.fill_between(ts, R_upper, R_lower, alpha=0.2, color='g', label='95% CI')
    ax2.scatter(model.X, model.y[:,1], color='r', label='Data', marker='x')
    ax2.set_xlabel('t')
    ax2.set_ylabel('R(t)')
    ax2.set_ylim(-1, 2)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()