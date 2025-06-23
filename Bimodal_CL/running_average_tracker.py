import torch
import numpy as np
from typing import Union, Optional


class RunningAverageTracker:
    """Simple Exponential Moving Average tracker for any metric."""

    def __init__(self, alpha: float = 0.9, name: str = "running_average"):
        """
        Args:
            alpha: Smoothing factor (0 < alpha < 1). Higher = more smoothing.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

        self.alpha = alpha
        self.ema = None
        self.step_count = 0
        self.name = name

    def update(self, value: Union[float, torch.Tensor, np.ndarray]) -> float:
        """
        Update EMA with new value.

        Args:
            value: New metric value

        Returns:
            Current EMA value
        """
        # Convert to scalar
        if isinstance(value, torch.Tensor):
            scalar_value = value.item()
        elif isinstance(value, np.ndarray):
            scalar_value = value.item()
        else:
            scalar_value = float(value)

        # Update EMA
        if self.ema is None:
            self.ema = scalar_value
        else:
            self.ema = self.alpha * self.ema + (1.0 - self.alpha) * scalar_value

        self.step_count += 1
        return self.ema

    def get_value(self) -> Optional[float]:
        """Get current EMA value."""
        return self.ema

    def reset(self):
        """Reset the tracker."""
        self.ema = None
        self.step_count = 0

    def get_name(self) -> str:
        """Get the name of the tracker."""
        return self.name