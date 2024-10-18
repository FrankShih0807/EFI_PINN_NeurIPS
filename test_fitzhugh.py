from PINN.pinn_efi import PINN_EFI
from PINN.models.fitzhugh_nagumo import FitzHugh_Nagumo
import torch




if __name__ == '__main__':
    
    physics_model = FitzHugh_Nagumo()
    model = PINN_EFI(physics_model=physics_model, physics_loss_weight=10, lr=1e-5, sgld_lr=1e-4, lambda_y=10, lambda_theta=10)
    
    for key, value in model.__dict__.items():
        print('{}: {}'.format(key, value))
        
        
    ts = torch.linspace(0, physics_model.t_extend, 1000)
    
    losses = model.train(epochs=10000, eval_x=ts.view(-1,1))
    
    