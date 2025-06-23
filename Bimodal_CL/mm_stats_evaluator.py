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
        self.class_template = "class_{}"
        self.superclass_template = "superclass_{}"

    def evaluate(self, features, other_features, classes, superclasses, gather=False):
        # Gather all stats from all processes
        if self.world_size > 1 and gather:
            features = torch.cat(self.gather_layer(features), dim=0)
            other_features = torch.cat(self.gather_layer(other_features), dim=0)
            classes = torch.cat(self.gather_layer(classes), dim=0)
            superclasses = torch.cat(self.gather_layer(superclasses), dim=0)

        unique_classes = torch.unique(classes).tolist()
        unique_superclasses = torch.unique(superclasses).tolist()

        metrics = {}

        # First, let's compute the domain gap between the two modalities
        modality_gap = self._compute_modality_gap(features, other_features)
        self.running_average_trackers["modality"].update(modality_gap)

        metrics["modality/gap"] = modality_gap
        metrics["modality/gap_running_average"] = self.running_average_trackers[
            "modality"
        ].get_value()

        # Modality gap for each class
        for class_ in unique_classes:
            class_features = features[classes == class_]
            class_other_features = other_features[classes == class_]
            class_modality_gap = self._compute_modality_gap(
                class_features, class_other_features
            )
            self.running_average_trackers[self.class_template.format(class_)].update(
                class_modality_gap
            )

            metrics[f"{self.class_template.format(class_)}/gap"] = class_modality_gap
            metrics[f"{self.class_template.format(class_)}/gap_running_average"] = (
                self.running_average_trackers[
                    self.class_template.format(class_)
                ].get_value()
            )

        # Modality gap for each superclass
        for superclass_ in unique_superclasses:
            superclass_features = features[superclasses == superclass_]
            superclass_other_features = other_features[superclasses == superclass_]
            superclass_modality_gap = self._compute_modality_gap(
                superclass_features, superclass_other_features
            )

            self.running_average_trackers[
                self.superclass_template.format(superclass_)
            ].update(superclass_modality_gap)

            metrics[f"{self.superclass_template.format(superclass_)}/gap"] = (
                superclass_modality_gap
            )

            metrics[
                f"{self.superclass_template.format(superclass_)}/gap_running_average"
            ] = self.running_average_trackers[
                self.superclass_template.format(superclass_)
            ].get_value()

        return metrics

    def format(self, metrics, prefix="val"):
        formatted_metrics = {}

        for key, value in metrics.items():
            formatted_metrics[f"{prefix}/{key}"] = value

        return formatted_metrics

    def _compute_modality_gap(self, features, other_features):
        """Compute the modality gap between the two modalities. Embeddings must be normalized to unit length."""
        features_mean = features.mean(dim=0)
        other_features_mean = other_features.mean(dim=0)
        return torch.abs(torch.norm(features_mean - other_features_mean))
