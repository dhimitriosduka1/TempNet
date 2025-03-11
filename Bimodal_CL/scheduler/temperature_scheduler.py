import torch
from collections import Counter


def get_next_temperature(tau_min, tau_max, period, global_it, offset):
    return tau_min + 0.5 * (tau_max - tau_min) * (
        1 + torch.cos(torch.tensor((2 * torch.pi * global_it / period) + offset))
    )


def get_per_class_temperature(classes_, tau_min, tau_max):
    counter = Counter(classes_)
    counter = counter.most_common()

    min_samples = counter[-1][1]
    max_samples = counter[0][1]

    per_class_temperature = {}
    for class_, count in counter:
        per_class_temperature[class_] = (
            (count - min_samples) / (max_samples - min_samples)
        ) * (tau_max - tau_min) + tau_min

    return per_class_temperature
