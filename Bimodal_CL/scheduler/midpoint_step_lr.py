from torch.optim.lr_scheduler import _LRScheduler


class MidpointStepLRScheduler(_LRScheduler):
    """
    Custom PyTorch learning rate scheduler that implements a step function.
    Learning rate starts at lr_start and jumps to lr_end at the midpoint of training.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_start (float): Starting (minimum) learning rate.
        lr_end (float): Ending (maximum) learning rate.
        total_steps (int): Total number of training steps/epochs.
        last_epoch (int): The index of the last epoch. Default: -1.
    """

    def __init__(self, optimizer, lr_start, lr_end, num_epochs, last_epoch=-1):
        # Set custom attributes before calling parent's __init__. Otherwise, self.midpoint will be undefined.
        # This is because the parent's __init__ method calls self.step() which calls get_lr().
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.num_epochs = num_epochs
        self.midpoint = self.num_epochs // 2

        super(MidpointStepLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate.
        """
        if self.last_epoch < self.midpoint:
            return [self.lr_start for _ in self.base_lrs]
        else:
            return [self.lr_end for _ in self.base_lrs]
