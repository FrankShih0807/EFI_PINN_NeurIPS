import torch
import torch.nn as nn
import time
from PINN.common.logger import Logger, configure

class BasePINN(object):
    def __init__(
        self,
        physics_model,
        dataset,
        hidden_layers=[15, 15],
        activation_fn=nn.Softplus(beta=10),
        lr=1e-3,
        lambda_pde=1,
        save_path=None,
        device='cpu',
        verbose=1,
    ) -> None:
        super().__init__()
        self.physics_model = physics_model
        self.dataset = dataset.copy()

        # Physics loss
        self.differential_operator = self.physics_model.differential_operator
        self.lambda_pde = lambda_pde

        # Common configs
        self.lr = lr
        self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        self.save_path = save_path
        self.device = device
        
        try:
            self.physics_model.plot_true_solution(save_path)
        except:
            print("No true solution to plot")

        # To device
        self.sol_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'solution'], dim=0).to(self.device)
        self.sol_y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'solution'], dim=0).to(self.device)
                           
        self.eval_X = torch.cat([d['X'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        self.eval_y = torch.cat([d['y'] for d in self.dataset if d['category'] == 'evaluation'], dim=0).to(self.device)
        
        self.input_dim = self.sol_X.shape[1]
        self.output_dim = self.sol_y.shape[1]
        
        self.verbose = verbose
        if self.verbose == 1:
            format_strings = ["stdout", "csv"]
        else:
            format_strings = ["csv"]
        
        
        self.logger = configure(self.save_path, format_strings)
        self._get_scheduler()
        self._pinn_init()

    def _pinn_init(self):
        ''' Implement the network and optimiser initialisation here '''
        raise NotImplementedError()
    
    def _get_scheduler(self):
        ''' Implement the learning rate scheduler here '''
        pass

    def update(self):
        ''' Implement the network parameter update here '''
        raise NotImplementedError()
    
    def train(self, epochs, eval_freq=-1, burn=0.5, callback=None):
        self.epochs = epochs
        if eval_freq == -1:
            eval_freq = epochs // 10
        self.callback = callback
        self.callback.init_callback(self, eval_freq=eval_freq, burn=burn)
        self.n_eval = 0

        for ep in range(epochs):
            self.progress = (ep+1) / epochs
            tic = time.time()
            sol_loss, pde_loss = self.update()
            toc = time.time()
            
            self.callback.on_training()
            
            self.logger.record('train/progress', self.progress)
            self.logger.record('train/epoch', ep+1)
            self.logger.record_mean('train/sol_loss', sol_loss)
            self.logger.record_mean('train/pde_loss', pde_loss)
            self.logger.record_mean('train/time', toc-tic)
            
            ## 3. Loss calculation
            if (ep+1) % eval_freq == 0:
                self.callback.on_eval()
                self.logger.dump()


        self.callback.on_training_end()
    
    