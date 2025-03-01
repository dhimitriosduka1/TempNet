import torch

def get_next_temperature(tau_min, tau_max, period, global_it):
    return tau_min + 0.5 * (tau_max - tau_min) * (1 + torch.cos(torch.tensor(2 * torch.pi * global_it / period)))