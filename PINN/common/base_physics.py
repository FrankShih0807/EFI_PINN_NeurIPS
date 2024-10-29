

class PhysicsModel(object):
    def __init__(self, **kwargs) -> None:
        ''' Initialize the physics model '''
        print('Model:', self.__class__.__name__)
        self.model_params = dict()
        for key, value in kwargs.items():
            setattr(self, key, value)
            print('{}: {}'.format(key, value))
            self.model_params[key] = value
            
        self.X, self.y = self._data_generation()
        self.eval_X = self._eval_data_generation()
        
        self.input_dim = self.X.shape[1]
        self.output_dim = self.y.shape[1]
        self.n_samples = self.X.shape[0]
    
    def _data_generation(self):
        ''' Implement the data generation here and return X, y '''
        raise NotImplementedError()
    
    def _eval_data_generation(self):
        ''' Implement the evaluation data generation here and return X '''
        raise NotImplementedError
    
    def physics_law(self):
        ''' Implement the physics law here '''
        raise NotImplementedError()
    
    def physics_loss(self):
        ''' Implement the physics loss here '''
        raise NotImplementedError()
    
    def plot_true_solution(self, save_path=None):
        ''' Plot the true solution here '''
        pass
    def save_evaluation(self, model, save_path=None):
        ''' Plot the predicted solution here '''
        pass