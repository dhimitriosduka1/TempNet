import torch
import numpy as np
import torch.distributed as dist


# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class MMStatsEvaluator:
    def __init__(self, world_size=8, running_average_trackers=None):
        self.world_size = world_size
        self.gather_layer = GatherLayer.apply
        self.running_average_trackers = running_average_trackers

    def evaluate(self, features, other_features):
        # Gather all stats from all processes
        if self.world_size > 1:
            features = torch.cat(self.gather_layer(features), dim=0)
            other_features = torch.cat(self.gather_layer(other_features), dim=0)

        metrics = {}

        # First, let's compute the domain gap between the two modalities
        modality_gap = self._compute_modality_gap(features, other_features)
        self.running_average_trackers["modality_gap"].update(modality_gap)

        metrics["modality_gap"] = modality_gap
        metrics["modality_gap/running_average"] = self.running_average_trackers[
            "modality_gap"
        ].get_value()

        return metrics

    def format(self, metrics, prefix="train"):
        formatted_metrics = {}
    
        for key, value in metrics.items():
            formatted_metrics[f"{prefix}/{key}"] = value

        return formatted_metrics

    def _compute_modality_gap(self, features, other_features):
        """Compute the modality gap between the two modalities. Embeddings must be normalized to unit length."""
        features_mean = features.mean(dim=0)
        other_features_mean = other_features.mean(dim=0)
        return torch.abs(torch.norm(features_mean - other_features_mean))
