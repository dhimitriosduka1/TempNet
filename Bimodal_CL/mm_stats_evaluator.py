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

    def evaluate(
        self, image_features, text_features, classes, superclasses, gather=False
    ):
        # Gather all stats from all processes
        if self.world_size > 1 and gather:
            image_features = torch.cat(self.gather_layer(image_features), dim=0)
            text_features = torch.cat(self.gather_layer(text_features), dim=0)
            classes = torch.cat(self.gather_layer(classes), dim=0)
            superclasses = torch.cat(self.gather_layer(superclasses), dim=0)

        metrics = {}

        # /////////////////////// Modality Gap ///////////////////////
        # First, let's compute the domain gap between the two modalities
        modality_gap = self._compute_modality_gap(image_features, text_features)
        self.running_average_trackers["modality"]["gap"].update(modality_gap)

        metrics["modality/gap"] = modality_gap
        metrics["modality/gap_ema"] = self.running_average_trackers["modality"][
            "gap"
        ].get_value()

        # Then, let's compute the modality gap per class
        per_class_modality_gap = self._compute_modality_gap_per_class(
            image_features, text_features, classes
        )
        metrics.update(per_class_modality_gap)

        # Then, let's compute the modality gap per superclass
        per_superclass_modality_gap = self._compute_modality_gap_per_superclass(
            image_features, text_features, superclasses
        )
        metrics.update(per_superclass_modality_gap)

        # /////////////////////// Average Pairwise Distance ///////////////////////
        # First, let's compute the average pairwise distance between the two modalities
        per_modality_average_pairwise_distance = (
            self._compute_average_pairwise_distance_per_modality(
                image_features, text_features
            )
        )
        metrics.update(per_modality_average_pairwise_distance)

        # Then, let's compute the average pairwise distance per class
        per_class_average_pairwise_distance = (
            self._compute_average_pairwise_distance_per_class(
                image_features, text_features, classes
            )
        )
        metrics.update(per_class_average_pairwise_distance)

        # Then, let's compute the average pairwise distance per superclass
        per_superclass_average_pairwise_distance = (
            self._compute_average_pairwise_distance_per_superclass(
                image_features, text_features, superclasses
            )
        )
        metrics.update(per_superclass_average_pairwise_distance)

        return metrics

    def format(self, metrics, prefix="val"):
        formatted_metrics = {}

        for key, value in metrics.items():
            formatted_metrics[f"{prefix}/{key}"] = value

        return formatted_metrics

    def _compute_modality_gap(self, features, other_features):
        """Compute the modality gap between the two modalities."""
        features_mean = features.mean(dim=0)
        other_features_mean = other_features.mean(dim=0)
        return torch.abs(torch.norm(features_mean - other_features_mean))

    def _compute_modality_gap_per_class(self, features, other_features, classes):
        """Compute the modality gap between the two modalities for each class."""
        unique_classes = torch.unique(classes).tolist()
        metrics = {}

        # Modality gap for each class
        for class_ in unique_classes:
            class_features = features[classes == class_]
            class_other_features = other_features[classes == class_]
            class_modality_gap = self._compute_modality_gap(
                class_features, class_other_features
            )
            self.running_average_trackers[self.class_template.format(class_)][
                "gap"
            ].update(class_modality_gap)

            metrics[f"{self.class_template.format(class_)}/gap"] = class_modality_gap
            metrics[f"{self.class_template.format(class_)}/gap_ema"] = (
                self.running_average_trackers[self.class_template.format(class_)][
                    "gap"
                ].get_value()
            )

        return metrics

    def _compute_modality_gap_per_superclass(
        self, features, other_features, superclasses
    ):
        """Compute the modality gap between the two modalities for each superclass."""
        unique_superclasses = torch.unique(superclasses).tolist()
        metrics = {}

        # Modality gap for each superclass
        for superclass_ in unique_superclasses:
            superclass_features = features[superclasses == superclass_]
            superclass_other_features = other_features[superclasses == superclass_]
            superclass_modality_gap = self._compute_modality_gap(
                superclass_features, superclass_other_features
            )

            self.running_average_trackers[self.superclass_template.format(superclass_)][
                "gap"
            ].update(superclass_modality_gap)

            metrics[f"{self.superclass_template.format(superclass_)}/gap"] = (
                superclass_modality_gap
            )

            metrics[f"{self.superclass_template.format(superclass_)}/gap_ema"] = (
                self.running_average_trackers[
                    self.superclass_template.format(superclass_)
                ]["gap"].get_value()
            )

        return metrics

    def _compute_average_pairwise_distance(self, features):
        """Compute the average pairwise distance between the two modalities. Embeddings must be normalized to unit length."""
        pairwise_dists = torch.cdist(features, features, p=2)

        # Get upper triangle (excluding diagonal)
        pairwise_dists = pairwise_dists[
            torch.triu(torch.ones_like(pairwise_dists), diagonal=1) == 1
        ]

        # Compute the average and max pairwise distance
        return torch.mean(pairwise_dists)

    def _compute_average_pairwise_distance_per_modality(
        self, image_features, text_features
    ):
        """Compute the average pairwise distance between the two modalities."""
        metrics = {}
        average_pairwise_distance_image = self._compute_average_pairwise_distance(
            image_features
        )
        self.running_average_trackers["modality"][
            "average_pairwise_distance_image"
        ].update(average_pairwise_distance_image)

        average_pairwise_distance_text = self._compute_average_pairwise_distance(
            text_features
        )
        self.running_average_trackers["modality"][
            "average_pairwise_distance_text"
        ].update(average_pairwise_distance_text)

        metrics["modality/average_pairwise_distance_image"] = (
            average_pairwise_distance_image
        )
        metrics["modality/average_pairwise_distance_text"] = (
            average_pairwise_distance_text
        )
        metrics["modality/average_pairwise_distance_image_ema"] = (
            self.running_average_trackers["modality"][
                "average_pairwise_distance_image"
            ].get_value()
        )
        metrics["modality/average_pairwise_distance_text_ema"] = (
            self.running_average_trackers["modality"][
                "average_pairwise_distance_text"
            ].get_value()
        )

        return metrics

    def _compute_average_pairwise_distance_per_class(
        self, image_features, text_features, classes
    ):
        """Compute the average pairwise distance between the two modalities for each class."""
        unique_classes = torch.unique(classes).tolist()
        metrics = {}

        for class_ in unique_classes:
            class_key = self.class_template.format(class_)
            per_class_image_features = image_features[classes == class_]
            per_class_text_features = text_features[classes == class_]
            per_class_average_pairwise_distance = (
                self._compute_average_pairwise_distance(per_class_image_features)
            )
            self.running_average_trackers[class_key][
                "average_pairwise_distance_image"
            ].update(per_class_average_pairwise_distance)

            metrics[f"{class_key}/average_pairwise_distance_image"] = (
                per_class_average_pairwise_distance
            )
            metrics[f"{class_key}/average_pairwise_distance_image_ema"] = (
                self.running_average_trackers[class_key][
                    "average_pairwise_distance_image"
                ].get_value()
            )

            per_class_average_pairwise_distance = (
                self._compute_average_pairwise_distance(per_class_text_features)
            )
            self.running_average_trackers[class_key][
                "average_pairwise_distance_text"
            ].update(per_class_average_pairwise_distance)

            metrics[f"{class_key}/average_pairwise_distance_text"] = (
                per_class_average_pairwise_distance
            )
            metrics[f"{class_key}/average_pairwise_distance_text_ema"] = (
                self.running_average_trackers[class_key][
                    "average_pairwise_distance_text"
                ].get_value()
            )

        return metrics

    def _compute_average_pairwise_distance_per_superclass(
        self, image_features, text_features, superclasses
    ):
        """Compute the average pairwise distance between the two modalities for each superclass."""
        unique_superclasses = torch.unique(superclasses).tolist()
        metrics = {}

        for superclass_ in unique_superclasses:
            superclass_key = self.superclass_template.format(superclass_)
            per_superclass_image_features = image_features[superclasses == superclass_]
            per_superclass_average_pairwise_distance = (
                self._compute_average_pairwise_distance(per_superclass_image_features)
            )
            self.running_average_trackers[superclass_key][
                "average_pairwise_distance_image"
            ].update(per_superclass_average_pairwise_distance)

            metrics[f"{superclass_key}/average_pairwise_distance_image"] = (
                per_superclass_average_pairwise_distance
            )
            metrics[f"{superclass_key}/average_pairwise_distance_image_ema"] = (
                self.running_average_trackers[superclass_key][
                    "average_pairwise_distance_image"
                ].get_value()
            )

            per_superclass_text_features = text_features[superclasses == superclass_]
            per_superclass_average_pairwise_distance = (
                self._compute_average_pairwise_distance(per_superclass_text_features)
            )
            self.running_average_trackers[superclass_key][
                "average_pairwise_distance_text"
            ].update(per_superclass_average_pairwise_distance)

            metrics[f"{superclass_key}/average_pairwise_distance_text"] = (
                per_superclass_average_pairwise_distance
            )
            metrics[f"{superclass_key}/average_pairwise_distance_text_ema"] = (
                self.running_average_trackers[superclass_key][
                    "average_pairwise_distance_text"
                ].get_value()
            )

        return metrics
