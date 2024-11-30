import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def trancated_l1_loss(input, threshold=1.0, scale=1.0):
    loss = scale * torch.where(input.abs() > threshold, torch.zeros_like(input.abs()).to(input.device), input.abs())
    return loss

def k_exp_loss(input, k=2.0, xi=1.0, scale=1.0):
    x = (input.abs()/xi).pow(k)
    loss = scale * xi * x * torch.exp(-x)
    return loss

def gmm_loss(input, sigma_1, sigma_0, ratio=1.0):
    gaussian_1 = torch.exp(-0.5 * (input / sigma_1) ** 2) / (sigma_1 * (2 ) ** 0.5)
    gaussian_0 = torch.exp(-0.5 * (input / sigma_0) ** 2) / (sigma_0 * (2 ) ** 0.5)
    loss = -torch.log(ratio * gaussian_1 + (1 - ratio) * gaussian_0)
    return loss


if __name__ == '__main__':
    
    device = "mps"
    input = torch.linspace(-3, 3, steps=100).to(device)
    
    loss1 = trancated_l1_loss(input, threshold=1).to("cpu").numpy()
    loss2 = k_exp_loss(input, k=2, xi=1).to("cpu").numpy()
    loss3 = gmm_loss(input, sigma_1=1, sigma_0=0.01, ratio=0.1).to("cpu").numpy()
    

    plt.plot(input.to('cpu').numpy(), loss1)
    plt.plot(input.to('cpu').numpy(), loss2)
    plt.plot(input.to('cpu').numpy(), loss3)
    plt.show()