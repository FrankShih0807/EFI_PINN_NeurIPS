from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

def get_schedule(schedule):
    if isinstance(schedule, str):
        schedule_type = schedule.split('_')[0]
        if schedule_type in ['linear', 'lin']:
            start_value, end_value = map(float, schedule.split('_')[1:])
            return linear_schedule(start_value, end_value)
        elif schedule_type in ['polynomial', 'poly']:
            start_value, scale, power = map(float, schedule.split('_')[1:])
            return polynomial_schedule(start_value, scale, power)
        elif schedule_type in ['log']:
            start_value, end_value = map(float, schedule.split('_')[1:])
            return log_schedule(start_value, end_value)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
    elif isinstance(schedule, float):
        return constant_schedule(schedule)

def constant_schedule(value):
    def schedule(progress):
        return value
    return schedule

def linear_schedule(start_value, end_value):
    if start_value >= end_value:
        def schedule(progress):
            progress = np.clip(progress, 0.0, 1.0)
            return max(start_value - (start_value - end_value) * progress, end_value)
    else:
        def schedule(progress):
            progress = np.clip(progress, 0.0, 1.0)
            return min(start_value + (end_value - start_value) * progress, end_value)
    return schedule

def log_schedule(start_value, end_value):
    def schedule(progress):
        progress = np.clip(progress, 0.0, 1.0)
        return np.exp(np.log(start_value) + (np.log(end_value) - np.log(start_value)) * progress)
    return schedule

def polynomial_schedule(start_value, scale, power):
    def schedule(progress):
        progress = np.clip(progress, 0.0, 1.0)
        return start_value / (1 + (scale * progress) ** power)
    return schedule


class ConstantParamScheduler(_LRScheduler):
    def __init__(self, optimizer, param_name, start_value,last_epoch=-1):
        self.param_name = param_name
        self.start_value = start_value
        super(ConstantParamScheduler, self).__init__(optimizer, last_epoch)

    def get_param_value(self):
        return self.start_value

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.param_value = self.get_param_value()
        for param_group in self.optimizer.param_groups:
            param_group[self.param_name] = self.param_value

class LinearParamScheduler(_LRScheduler):
    def __init__(self, optimizer, param_name, start_value, end_value, total_steps, last_epoch=-1):
        self.param_name = param_name
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = int(total_steps)
        super(LinearParamScheduler, self).__init__(optimizer, last_epoch)

    def get_param_value(self):
        if self.last_epoch > self.total_steps:
            return self.end_value
        return max(self.start_value + (self.end_value - self.start_value) * (self.last_epoch / self.total_steps), self.end_value)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.param_value = self.get_param_value()
        for param_group in self.optimizer.param_groups:
            param_group[self.param_name] = self.param_value

            
class PolynomialParamScheduler(_LRScheduler):
    def __init__(self, optimizer, param_name, start_value, power, steps=100, last_epoch=-1):
        self.param_name = param_name
        self.start_value = start_value
        self.power = power
        self.steps = steps
        super(PolynomialParamScheduler, self).__init__(optimizer, last_epoch)

    def get_param_value(self):
        return self.start_value * (self.steps/(self.last_epoch+self.steps)) ** self.power

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.param_value = self.get_param_value()
        for param_group in self.optimizer.param_groups:
            param_group[self.param_name] = self.param_value
            
            
            
class GeneralParamScheduler(_LRScheduler):
    def __init__(self, optimizer, param_name, start_value, cycle_len=0, last_epoch=-1):
        self.param_name = param_name
        self.start_value = start_value
        self.cycle_len = cycle_len
        super(GeneralParamScheduler, self).__init__(optimizer, last_epoch)

    def get_param_value(self):
        return self.start_value/2 * (1 + np.cos(np.pi * np.mod(self.last_epoch, self.cycle_len)/self.cycle_len))

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.param_value = self.get_param_value()
        for param_group in self.optimizer.param_groups:
            param_group[self.param_name] = self.param_value
            
if __name__ == '__main__':
    
    # schedule = get_anneal_schedule('lin_0.1_0.01', 101)
    # print(schedule(50))
    import matplotlib.pyplot as plt
    
    def plot_schedule(schedule, steps):
        x = np.linspace(0, 1, steps)
        y = [schedule(xi) for xi in x]
        plt.plot(x, y)
        plt.show()

    schedule = get_schedule('lin_0.1_0.01')
    
    schedule = get_schedule('poly_0.1_10_0.95')
    
    schedule = get_schedule('log_0.1_0.01')
    plot_schedule(schedule, 100)
    