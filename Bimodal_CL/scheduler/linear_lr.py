from torch.optim.lr_scheduler import _LRScheduler


class LinearLRScheduler(_LRScheduler):
    """
    Custom PyTorch learning rate scheduler that implements a constant learning rate.
    Learning rate remains fixed at lr_value throughout the entire training process.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_value (float): Constant learning rate value.
        last_epoch (int): The index of the last epoch. Default: -1.
    """

    def __init__(self, optimizer, lr_value, last_epoch=-1):
        # Set custom attributes before calling parent's __init__
        self.lr_value = lr_value

        super(LinearLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate.
        """
        return [self.lr_value for _ in self.base_lrs]
