import torch
import torch.nn as nn
from PINN.common.base_pinn import BasePINN
from PINN.models.poisson import Poisson

class PINN(BasePINN):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        lambda_pde=1,
        save_path=None,
        device='cpu'
    ) -> None:
        super().__init__(physics_model, dataset, hidden_layers, activation_fn, lr, lambda_pde, save_path, device)
    
    

    
    def pde_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'differential':
                diff_o = self.differential_operator(self.net, d['X'])
                loss += self.mse_loss(diff_o, d['y'])
        return loss
    
    def solution_loss(self):
        loss = 0
        for i, d in enumerate(self.dataset):
            if d['category'] == 'solution':
                loss += self.mse_loss(d['y'], self.net(d['X']))
        return loss
    
    def update(self):
        self.optimiser.zero_grad()
        sol_loss = self.solution_loss()
        pde_loss = self.pde_loss()
        loss = sol_loss + self.lambda_pde * pde_loss

        loss.backward()
        self.optimiser.step()
        
        return sol_loss.item(), pde_loss.item()



# if __name__ == "__main__":
#     physics_model = Poisson(lam1=1.0, lam2=1.0)
#     dataset = physics_model.generate_data(30, device='cpu')
#     model = PINN(physics_model, dataset, device='cpu')
#     model._pinn_init()
    
#     for epoch in range(20000):
#         loss, solution_loss, pde_loss = model.update()
#         if epoch % 1000 == 0:
#             print(f"Epoch {epoch}, Loss: {loss:.4f}, Solution Loss: {solution_loss:.4f}, PDE Loss: {pde_loss:.4f}")

#     X = torch.linspace(physics_model.t_start, physics_model.t_end, steps=100).reshape(100, -1).requires_grad_(True)
#     y = model.net(X).detach().numpy()
#     print(y)
#     y_true = physics_model.physics_law(X).detach().numpy()
#     f_true = physics_model.differential_function(X).detach().numpy()
#     f = model.differential_operator(model.net, X).detach().numpy()
#     X = X.detach().numpy()

#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     ax[0].plot(X, y, label='Predicted Solution')
#     ax[0].plot(X, y_true, label='True Solution')
#     ax[0].set_title('Solution')
#     ax[0].legend()

#     ax[1].plot(X, f, label='Predicted Differential')
#     ax[1].plot(X, f_true, label='True Differential')
#     ax[1].set_title('Differential')
#     ax[1].legend()

#     plt.show()