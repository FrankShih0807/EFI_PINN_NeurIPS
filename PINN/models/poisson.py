import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from PINN.common.grad_tool import grad
from PINN.common.base_physics import PhysicsModel
from PINN.common.utils import PINNDataset
from PIL import Image

class Poisson(PhysicsModel):
    def __init__(self, 
                 t_start=-0.7,
                 t_end=0.7, 
                 boundary_sd=0.01,
                 diff_sd=0.01,
                 ):
        super().__init__(t_start=t_start, t_end=t_end, boundary_sd=boundary_sd, diff_sd=diff_sd)

    def generate_data(self, n_samples, device):
        dataset = PINNDataset(device=device)
        X, y = self.get_solu_data()
        diff_X, diff_y = self.get_diff_data(n_samples)
        eval_X, eval_y = self.get_eval_data()
        dataset.add_data(X, y, category='solution', noise_sd=self.boundary_sd)
        dataset.add_data(diff_X, diff_y, category='differential', noise_sd=self.diff_sd)
        dataset.add_data(eval_X, eval_y, category='evaluation', noise_sd=0)
        
        return dataset
    
    def get_eval_data(self):
        X = torch.linspace(self.t_start, self.t_end, steps=100).reshape(100, -1)
        y = self.physics_law(X)
        return X, y
    
    def get_solu_data(self):
        # X = torch.tensor([self.t_start, self.t_end, -0.5, 0.5]).view(-1, 1)
        X = torch.tensor([self.t_start, self.t_end]).repeat_interleave(5).view(-1, 1)
        # X = torch.linspace(self.t_start, self.t_end, steps=5).repeat_interleave(5).reshape(-1, 1)
        # X = torch.tensor([self.t_start, self.t_end]).view(-1, 1)
        y = self.physics_law(X)
        y += self.boundary_sd * torch.randn_like(y)
        return X, y
    
    def get_diff_data(self, n_samples, replicate=1):
        X = torch.linspace(self.t_start, self.t_end, steps=n_samples).repeat_interleave(replicate).view(-1, 1)
        y = self.differential_function(X)
        y += self.diff_sd * torch.randn_like(y)
        return X, y

    
    def physics_law(self, X):
        y = torch.sin(6 * X) ** 3
        return y
    
    def differential_function(self, X):
        y = -1.08 * torch.sin(6 * X) * (torch.sin(6 * X) ** 2 - 2 * torch.cos(6 * X) ** 2)
        return y
    
    def differential_operator(self, model: torch.nn.Module, physics_X):
        u = model(physics_X)
        # u_x = torch.autograd.grad(u, physics_X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # u_xx = torch.autograd.grad(u_x, physics_X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad(u, physics_X)[0]
        u_xx = grad(u_x, physics_X)[0]
        pde = 0.01 * u_xx
        
        return pde

    def plot_true_solution(self, save_path=None):
        X = torch.linspace(self.t_start, self.t_end, steps=100)
        y = self.physics_law(X)
        
        sns.set_theme()
        plt.plot(X, y, label='Equation')
        plt.legend()
        plt.ylabel('u')
        plt.xlabel('x')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'true_solution.png'))
        plt.close()
        
    def save_evaluation(self, model, save_path=None):
        # preds_upper, preds_lower, preds_mean = model.summary()
        # pred_dict = model.summary()
        
        # preds_upper = pred_dict['y_preds_upper'].flatten()
        # preds_lower = pred_dict['y_preds_lower'].flatten()
        # preds_mean = pred_dict['y_preds_mean'].flatten()

        X = torch.linspace(self.t_start, self.t_end, steps=100)
        y = self.physics_law(X)
        
        preds_mean = model.eval_buffer.get_mean()
        preds_upper, preds_lower = model.eval_buffer.get_ci()
        
        # if save_path is None:
        #     save_path = './evaluation_results'
        
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        
        # np.savez(os.path.join(save_path, 'evaluation_data.npz'), preds_upper=preds_upper, preds_lower=preds_lower, preds_mean=preds_mean)
        # np.savez(os.path.join(save_path, 'evaluation_data.npz') , **pred_dict)
        
        sns.set_theme()
        plt.plot(X, y, alpha=0.8, color='b', label='True')
        plt.plot(X, preds_mean, alpha=0.8, color='g', label='Mean')

        plt.fill_between(X, preds_upper, preds_lower, alpha=0.2, color='g', label='95% CI')
        plt.legend()
        plt.ylabel('u')
        plt.xlabel('x')
        plt.savefig(os.path.join(save_path, 'pred_solution.png'))
        plt.close()
        
        
    def save_temp_frames(self, model, epoch, save_path=None):
        X = torch.linspace(self.t_start, self.t_end, steps=100)
        y = self.physics_law(X)
        
        preds_mean = model.eval_buffer.get_mean()
        preds_upper, preds_lower = model.eval_buffer.get_ci()
        
        temp_dir = os.path.join(save_path, 'temp_frames')
        os.makedirs(temp_dir, exist_ok=True)
        
        
        sns.set_theme()
        plt.subplots(figsize=(6, 6))
        plt.plot(X, y, alpha=0.8, color='b', label='True')
        plt.plot(X, preds_mean, alpha=0.8, color='g', label='Mean')
        plt.ylim(-1.5, 1.5)
        
        plt.fill_between(X, preds_upper, preds_lower, alpha=0.2, color='g', label='95% CI')
        plt.legend()
        plt.ylabel('u')
        plt.xlabel('x')
        
        frame_path = os.path.join(temp_dir, f"frame_{epoch}.png")
        plt.savefig(frame_path)
        plt.close()

    def create_gif(self, save_path):
        frames = []
        temp_dir = os.path.join(save_path, 'temp_frames')
        n_frames = len(os.listdir(temp_dir))
        for epoch in range(n_frames):
            frame_path = os.path.join(temp_dir, f"frame_{epoch}.png")
            frames.append(Image.open(frame_path))
        # frame_files = sorted(os.listdir(temp_dir))  # Sort by file name to maintain order
        # print(frame_files)
        # frames = [Image.open(os.path.join(temp_dir, frame)) for frame in frame_files]
        
        frames[0].save(
            os.path.join(save_path, "training_loss.gif"),
            save_all=True,
            append_images=frames[1:],
            duration=500,
            loop=0
        )
        for frame_path in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, frame_path))
        os.rmdir(temp_dir)
    
    def get_pretrain_eval(self, base_model: torch.nn.Module):
        with torch.no_grad():
            X = torch.linspace(self.t_start, self.t_end, steps=100)
            base_model = base_model.to(X.device)
            preds = base_model(X.unsqueeze(1))

        self.pretrain_eval = preds


# if __name__ == "__main__":
#     physics = Poisson()
#     print(physics.model_params)
    
#     X, y = physics.X, physics.y
    
#     X_plot = torch.linspace(-0.7, 0.7, 100)
#     y_plot = physics.physics_law(X_plot)
    
#     plt.plot(X_plot, y_plot, label='Equation')
#     plt.plot(X, y, 'x', label='Training data')
#     plt.legend()
#     plt.ylabel('Value')
#     plt.xlabel('X')
#     plt.show()