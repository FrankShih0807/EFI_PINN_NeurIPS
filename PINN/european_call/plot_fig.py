import matplotlib.pyplot as plt
import numpy as np
from PINN.models.european_call import EuropeanCall


def plot_3D():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    # ax = fig.gca(projection='3d')
    ax.plot_surface(S_grid,
                    t_grid, 
                    preds_mean, 
                    cmap='plasma')

    # ax.plot_surface(S_grid,
    #                 t_grid, 
    #                 preds_lower, 
    #                 color='b',
    #                 alpha=0.3)

    # ax.plot_surface(S_grid,
    #                 t_grid, 
    #                 preds_upper, 
    #                 color='g',
    #                 alpha=0.3)

    # fig.colorbar(im, shrink=0.5, aspect=5, pad=0.07)
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Option Price')
    ax.view_init(elev=30, azim=225)
    plt.savefig('PINN/european_call/european_call_efi.png')
    plt.show()
    
if __name__ == '__main__':
    
    model = EuropeanCall()
    
    
    data = np.load('PINN/european_call/output.npz')

    preds_upper = data['preds_upper']
    preds_lower = data['preds_lower']
    preds_mean = data['preds_mean']
    S_grid = data['S_grid']
    t_grid = data['t_grid']

    # plot_3D()
    
    print(t_grid[:,0])
    true_price = model.physics_law(S_grid[:,0], t_grid[:,0])
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(S_grid[:,0], preds_mean[:,0], label='PINN_EFI')
    ax.plot(S_grid[:,0], true_price, label='True Price')
    ax.fill_between(S_grid[:,0], preds_upper[:,0], preds_lower[:,0], alpha=0.2, color='g', label='95% CI')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Option Price')
    ax.legend()
    plt.show()
    
    