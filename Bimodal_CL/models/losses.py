"""
implementation of other two-way contrastive losses
"""

import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb
import utils


# Function responsible for extracting different statistics from the custom per sample temperature tensor
def get_temperature_statistics(per_sample_temperature, prefix="train/"):
    min_per_sample_temperature = per_sample_temperature.min().item()
    max_per_sample_temperature = per_sample_temperature.max().item()
    avg_per_sample_temperature = per_sample_temperature.mean().item()
    median_per_sample_temperature = per_sample_temperature.median().item()
    quantile_0_5_per_sample_temperature = (
        per_sample_temperature.float().quantile(0.5).item()
    )

    temp_of_positives = per_sample_temperature.diag()
    positive_samples_min_temperature = temp_of_positives.min().item()
    positive_samples_max_temperature = temp_of_positives.max().item()
    positive_samples_avg_temperature = temp_of_positives.mean().item()
    positive_samples_median_temperature = temp_of_positives.median().item()
    positive_samples_quantile_0_5_temperature = (
        temp_of_positives.float().quantile(0.5).item()
    )

    temp_of_negatives = per_sample_temperature[
        ~torch.eye(per_sample_temperature.size(0), dtype=bool)
    ]
    negative_samples_min_temperature = temp_of_negatives.min().item()
    negative_samples_max_temperature = temp_of_negatives.max().item()
    negative_samples_avg_temperature = temp_of_negatives.mean().item()
    negative_samples_median_temperature = temp_of_negatives.median().item()
    negative_samples_quantile_0_5_temperature = (
        temp_of_negatives.float().quantile(0.5).item()
    )

    return {
        f"{prefix}min_per_sample_temperature": min_per_sample_temperature,
        f"{prefix}max_per_sample_temperature": max_per_sample_temperature,
        f"{prefix}avg_per_sample_temperature": avg_per_sample_temperature,
        f"{prefix}median_per_sample_temperature": median_per_sample_temperature,
        f"{prefix}quantile_0.5_per_sample_temperature": quantile_0_5_per_sample_temperature,
        f"{prefix}positive_samples_min_temperature": positive_samples_min_temperature,
        f"{prefix}positive_samples_max_temperature": positive_samples_max_temperature,
        f"{prefix}positive_samples_avg_temperature": positive_samples_avg_temperature,
        f"{prefix}positive_samples_median_temperature": positive_samples_median_temperature,
        f"{prefix}positive_samples_quantile_0.5_temperature": positive_samples_quantile_0_5_temperature,
        f"{prefix}negative_samples_min_temperature": negative_samples_min_temperature,
        f"{prefix}negative_samples_max_temperature": negative_samples_max_temperature,
        f"{prefix}negative_samples_avg_temperature": negative_samples_avg_temperature,
        f"{prefix}negative_samples_median_temperature": negative_samples_median_temperature,
        f"{prefix}negative_samples_quantile_0.5_temperature": negative_samples_quantile_0_5_temperature,
    }


def get_temperature_statistics_1d(per_sample_temperature, prefix="train/"):
    return {
        f"{prefix}min_temperature": per_sample_temperature.min().item(),
        f"{prefix}max_temperature": per_sample_temperature.max().item(),
        f"{prefix}avg_temperature": per_sample_temperature.mean().item(),
        f"{prefix}median_temperature": per_sample_temperature.median().item(),
        f"{prefix}quantile_0.5_temperature": per_sample_temperature.float()
        .quantile(0.5)
        .item(),
    }


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


class CLIP_Loss(nn.Module):

    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        i2i_temperature=0.01,
        t2t_temperature=0.01,
        personalized_tau=False,
        image_tau=None,
        text_tau=None,
    ):
        super(CLIP_Loss, self).__init__()
        self.world_size = world_size

        self.temperature = temperature
        self.i2i_temperature = i2i_temperature
        self.t2t_temperature = t2t_temperature

        self.personalized_tau = (
            personalized_tau  # if true, then temperatures are learnable
        )
        self.image_tau = image_tau
        self.text_tau = text_tau

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_i2i_temperature(self, i2i_temperature):
        self.i2i_temperature = i2i_temperature

    def set_t2t_temperature(self, t2t_temperature):
        self.t2t_temperature = t2t_temperature

    def forward(
        self,
        image_features,
        text_features,
        augmented_image_features,
        augmented_text_features,
        image_idx=None,
        text_idx=None,
        args=None,
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

            if args.enable_i2i_loss:
                augmented_image_features = torch.cat(
                    GatherLayer.apply(augmented_image_features), dim=0
                )

            if args.enable_t2t_loss:
                augmented_text_features = torch.cat(
                    GatherLayer.apply(augmented_text_features), dim=0
                )

        if self.personalized_tau:
            image_temp = self.image_tau[image_idx]
            text_temp = self.text_tau[text_idx]
            sim = torch.einsum("i d, j d -> i j", text_features, image_features)
            labels = torch.arange(image_features.shape[0], device=image_features.device)
            total_loss = (
                F.cross_entropy(sim / text_temp, labels)
                + F.cross_entropy(sim.t() / image_temp, labels)
            ) / 2

        else:
            sim = (
                torch.einsum("i d, j d -> i j", text_features, image_features)
                / self.temperature
            )
            labels = torch.arange(image_features.shape[0], device=image_features.device)

            i2t_loss = F.cross_entropy(sim.t(), labels)
            t2i_loss = F.cross_entropy(sim, labels)

            total_loss = (i2t_loss + t2i_loss) / 2

            log_obj = {
                "train/temperature": self.temperature,
                "train/i2i_temperature": self.i2i_temperature,
                "train/t2t_temperature": self.t2t_temperature,
                "train/t2i_loss": t2i_loss.item(),
                "train/i2t_loss": i2t_loss.item(),
            }

            # Compute i2i loss if augmented_image_features is not None:
            if args.enable_i2i_loss:
                sim_i2i = (
                    image_features @ augmented_image_features.T
                ) / self.i2i_temperature

                labels = torch.arange(
                    augmented_image_features.shape[0], device=image_features.device
                )
                i2i_loss = F.cross_entropy(sim_i2i, labels)
                total_loss += args.i2i_loss_weight * i2i_loss

                log_obj["train/i2i_loss"] = i2i_loss.item()

            # Compute t2t loss if augmented_text_features is not None:
            if args.enable_t2t_loss:
                sim_t2t = (
                    text_features @ augmented_text_features.T
                ) / self.t2t_temperature
                labels = torch.arange(
                    augmented_text_features.shape[0], device=text_features.device
                )
                t2t_loss = F.cross_entropy(sim_t2t, labels)
                total_loss += args.t2t_loss_weight * t2t_loss

                log_obj["train/t2t_loss"] = t2t_loss.item()

            if utils.is_main_process():
                wandb.log(
                    log_obj,
                    step=wandb.run.step,
                )

        return total_loss


class CLIP_with_TempNet_Loss(nn.Module):

    def __init__(
        self,
        world_size=8,
        feature_dim=256,
        rho=7.0,
    ):
        super(CLIP_with_TempNet_Loss, self).__init__()
        self.world_size = world_size
        self.feature_dim = feature_dim
        self.rho = rho

        self.image_temp_gen = TempGenerator(
            feature_dim=feature_dim, rho=self.rho
        ).cuda()

        self.text_temp_gen = TempGenerator(feature_dim=feature_dim, rho=self.rho).cuda()

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_i2i_temperature(self, i2i_temperature):
        pass

    def set_t2t_temperature(self, t2t_temperature):
        pass

    def forward(self, image_features, text_features):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # generate temperatures
        tau_image = self.image_temp_gen(image_features.detach())
        tau_text = self.text_temp_gen(text_features.detach())

        sim = torch.einsum("i d, j d -> i j", text_features, image_features)
        labels = torch.arange(image_features.shape[0], device=image_features.device)

        i2t_loss = F.cross_entropy(sim.t() / tau_image, labels)
        t2i_loss = F.cross_entropy(sim / tau_text, labels)

        total_loss = (i2t_loss + t2i_loss) / 2

        log_obj = {
            "train/t2i_loss": t2i_loss.item(),
            "train/i2t_loss": i2t_loss.item(),
        }

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class Sim_Based_CLIP_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        alpha=0.1,
    ):
        super(Sim_Based_CLIP_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.alpha = alpha

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(
        self,
        image_features,
        text_features,
        augmented_image_features,
        augmented_text_features,
        args=None,
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

            if args.enable_i2i_loss:
                augmented_image_features = torch.cat(
                    GatherLayer.apply(augmented_image_features), dim=0
                )

            if args.enable_t2t_loss:
                augmented_text_features = torch.cat(
                    GatherLayer.apply(augmented_text_features), dim=0
                )

        sim = text_features @ image_features.T
        per_sample_temperature = self.temperature + self.alpha * torch.sqrt(
            (sim + 1.0) / 2.0
        )

        sim /= per_sample_temperature

        labels = torch.arange(image_features.shape[0], device=image_features.device)

        i2t_loss = F.cross_entropy(sim.t(), labels)
        t2i_loss = F.cross_entropy(sim, labels)

        total_loss = (i2t_loss + t2i_loss) / 2

        log_obj = {
            "train/temperature": self.temperature,
            "train/t2i_loss": t2i_loss.item(),
            "train/i2t_loss": i2t_loss.item(),
        }

        # Compute i2i loss if augmented_image_features is not None:
        if args.enable_i2i_loss:
            sim_i2i = (image_features @ augmented_image_features.T) / self.temperature
            labels = torch.arange(
                augmented_image_features.shape[0], device=image_features.device
            )
            i2i_loss = F.cross_entropy(sim_i2i, labels)
            total_loss += args.i2i_loss_weight * i2i_loss

            log_obj["train/i2i_loss"] = i2i_loss.item()

        # Compute t2t loss if augmented_text_features is not None:
        if args.enable_t2t_loss:
            sim_t2t = (text_features @ augmented_text_features.T) / self.temperature
            labels = torch.arange(
                augmented_text_features.shape[0], device=text_features.device
            )
            t2t_loss = F.cross_entropy(sim_t2t, labels)
            total_loss += args.t2t_loss_weight * t2t_loss

            log_obj["train/t2t_loss"] = t2t_loss.item()

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class CLIP_MoE_Text_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        alpha=0.05,
    ):
        super(CLIP_MoE_Text_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.alpha = alpha

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(
        self,
        image_features,
        text_features,
        text_expert_features,
        args=None,
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
            text_expert_features = torch.cat(
                GatherLayer.apply(text_expert_features), dim=0
            )

        labels = torch.arange(image_features.shape[0], device=image_features.device)

        # Normal CLIP loss
        clip_sim = (text_features @ image_features.T) / self.temperature
        i2t_loss = F.cross_entropy(clip_sim.t(), labels)
        t2i_loss = F.cross_entropy(clip_sim, labels)
        clip_loss = (i2t_loss + t2i_loss) / 2

        # Expert based t2t loss
        t2t_expert_sim = text_expert_features @ text_expert_features.T
        per_sample_temperature = self.temperature + self.alpha * torch.sqrt(
            (t2t_expert_sim + 1.0) / 2.0
        )

        # t2t loss on CLIP space
        t2t_sim = (text_features @ text_features.T) / per_sample_temperature
        t2t_loss = F.cross_entropy(t2t_sim, labels)

        total_loss = clip_loss + args.t2t_loss_weight * t2t_loss

        log_obj = {
            "train/temperature": self.temperature,
            "train/i2t_loss": i2t_loss.item(),
            "train/t2i_loss": t2i_loss.item(),
            "train/t2t_loss": t2t_loss.item(),
            "train/clip_loss": clip_loss.item(),
        }

        log_obj.update(get_temperature_statistics(per_sample_temperature))

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class Scheduled_CLIP_MoE_Text_Loss(nn.Module):
    def __init__(self, world_size=8, temperature=0.01, alpha=0.05, total_steps=None):
        super(Scheduled_CLIP_MoE_Text_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.alpha = alpha
        self.total_steps = total_steps

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(
        self,
        image_features,
        text_features,
        text_expert_features,
        current_step,
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
            text_expert_features = torch.cat(
                GatherLayer.apply(text_expert_features), dim=0
            )

        labels = torch.arange(image_features.shape[0], device=image_features.device)

        # Normal CLIP loss
        clip_sim = (text_features @ image_features.T) / self.temperature
        i2t_loss = F.cross_entropy(clip_sim.t(), labels)
        t2i_loss = F.cross_entropy(clip_sim, labels)
        clip_loss = (i2t_loss + t2i_loss) / 2

        # Expert based t2t loss
        t2t_expert_sim = text_expert_features @ text_expert_features.T
        per_sample_temperature = self.temperature + self.alpha * torch.sqrt(
            (t2t_expert_sim + 1.0) / 2.0
        )

        # t2t loss on CLIP space
        t2t_sim = (text_features @ text_features.T) / per_sample_temperature
        t2t_loss = F.cross_entropy(t2t_sim, labels)

        normalized_current_step = current_step / self.total_steps

        clip_loss_weight = (normalized_current_step - 1.0) ** 2
        sim_loss_weight = normalized_current_step**2

        total_loss = clip_loss_weight * clip_loss + sim_loss_weight * t2t_loss

        log_obj = {
            "train/temperature": self.temperature,
            "train/i2t_loss": i2t_loss.item(),
            "train/t2i_loss": t2i_loss.item(),
            "train/t2t_loss": t2t_loss.item(),
            "train/clip_loss": clip_loss.item(),
            "train/clip_loss_weight": clip_loss_weight,
            "train/sim_loss_weight": sim_loss_weight,
        }

        log_obj.update(get_temperature_statistics(per_sample_temperature))

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class CLIP_MoE_Vision_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        alpha=0.05,
    ):
        super(CLIP_MoE_Vision_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.alpha = alpha

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(
        self,
        image_features,
        text_features,
        image_expert_features,
        args=None,
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
            image_expert_features = torch.cat(
                GatherLayer.apply(image_expert_features), dim=0
            )

        labels = torch.arange(image_features.shape[0], device=image_features.device)

        # Normal CLIP loss
        clip_sim = (text_features @ image_features.T) / self.temperature
        i2t_loss = F.cross_entropy(clip_sim.t(), labels)
        t2i_loss = F.cross_entropy(clip_sim, labels)
        clip_loss = (i2t_loss + t2i_loss) / 2

        # Expert based i2i loss
        i2i_expert_sim = image_expert_features @ image_expert_features.T
        per_sample_temperature = self.temperature + self.alpha * torch.sqrt(
            (i2i_expert_sim + 1.0) / 2.0
        )

        # i2i loss on CLIP space
        i2i_sim = (image_features @ image_features.T) / per_sample_temperature
        i2i_loss = F.cross_entropy(i2i_sim, labels)

        total_loss = clip_loss + args.i2i_loss_weight * i2i_loss

        log_obj = {
            "train/temperature": self.temperature,
            "train/i2t_loss": i2t_loss.item(),
            "train/t2i_loss": t2i_loss.item(),
            "train/i2i_loss": i2i_loss.item(),
            "train/clip_loss": clip_loss.item(),
            "train/sim_based_loss_alpha": self.alpha,
            "train/i2i_loss_weight": args.i2i_loss_weight,
        }

        log_obj.update(get_temperature_statistics(per_sample_temperature))

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class CLIP_MoE_Vision_And_Text_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        alpha=0.05,
    ):
        super(CLIP_MoE_Vision_And_Text_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.alpha = alpha

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(
        self,
        image_features,
        text_features,
        text_expert_features,
        image_expert_features,
        args=None,
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
            image_expert_features = torch.cat(
                GatherLayer.apply(image_expert_features), dim=0
            )
            text_expert_features = torch.cat(
                GatherLayer.apply(text_expert_features), dim=0
            )

        labels = torch.arange(image_features.shape[0], device=image_features.device)

        # Normal CLIP loss
        clip_sim = (text_features @ image_features.T) / self.temperature
        i2t_loss = F.cross_entropy(clip_sim.t(), labels)
        t2i_loss = F.cross_entropy(clip_sim, labels)
        clip_loss = (i2t_loss + t2i_loss) / 2

        # Expert based i2i loss
        i2i_expert_sim = image_expert_features @ image_expert_features.T
        per_sample_temperature_i2i = self.temperature + self.alpha * torch.sqrt(
            (i2i_expert_sim + 1.0) / 2.0
        )

        # i2i loss on CLIP space
        i2i_sim = (image_features @ image_features.T) / per_sample_temperature_i2i
        i2i_loss = F.cross_entropy(i2i_sim, labels)

        # Expert based t2t loss
        t2t_expert_sim = text_expert_features @ text_expert_features.T
        per_sample_temperature_t2t = self.temperature + self.alpha * torch.sqrt(
            (t2t_expert_sim + 1.0) / 2.0
        )

        # t2t loss on CLIP space
        t2t_sim = (text_features @ text_features.T) / per_sample_temperature_t2t
        t2t_loss = F.cross_entropy(t2t_sim, labels)

        total_loss = (
            clip_loss
            + args.i2i_loss_weight * i2i_loss
            + args.t2t_loss_weight * t2t_loss
        )

        log_obj = {
            "train/temperature": self.temperature,
            "train/i2t_loss": i2t_loss.item(),
            "train/t2i_loss": t2i_loss.item(),
            "train/i2i_loss": i2i_loss.item(),
            "train/clip_loss": clip_loss.item(),
            "train/sim_based_loss_alpha": self.alpha,
            "train/i2i_loss_weight": args.i2i_loss_weight,
            "train/t2t_loss": t2t_loss.item(),
            "train/t2t_loss_weight": args.t2t_loss_weight,
        }

        log_obj.update(
            get_temperature_statistics(per_sample_temperature_i2i, prefix="train/i2i/")
        )

        log_obj.update(
            get_temperature_statistics(per_sample_temperature_t2t, prefix="train/t2t/")
        )

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class CLIP_MoE_Blending_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        alpha=0.05,
        sim_blend_ratio=0.2,
    ):
        super(CLIP_MoE_Blending_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.alpha = alpha

        self.sim_blend_ratio = sim_blend_ratio

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(
        self,
        image_features,
        text_features,
        text_expert_features,
        image_expert_features,
        args=None,
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
            text_expert_features = torch.cat(
                GatherLayer.apply(text_expert_features), dim=0
            )

        labels = torch.arange(image_features.shape[0], device=image_features.device)

        # Normal CLIP loss
        clip_sim = (text_features @ image_features.T) / self.temperature
        i2t_loss = F.cross_entropy(clip_sim.t(), labels)
        t2i_loss = F.cross_entropy(clip_sim, labels)
        clip_loss = (i2t_loss + t2i_loss) / 2

        # Expert based t2t loss
        t2t_expert_sim = text_expert_features @ text_expert_features.T
        t2t_sim = text_features @ text_features.T

        # Mix the two similarities together
        blended_sim = (
            1.0 - self.sim_blend_ratio
        ) * t2t_sim + self.sim_blend_ratio * t2t_expert_sim

        per_sample_temperature = self.temperature + self.alpha * torch.sqrt(
            (blended_sim + 1.0) / 2.0
        )

        # i2i loss on CLIP space
        t2t_sim = (text_features @ text_features.T) / per_sample_temperature
        t2t_loss = F.cross_entropy(t2t_sim, labels)

        total_loss = clip_loss + args.t2t_loss_weight * t2t_loss

        temp_of_positives = per_sample_temperature.diag()
        temp_of_negatives = per_sample_temperature[
            ~torch.eye(per_sample_temperature.size(0), dtype=bool)
        ]

        log_obj = {
            "train/temperature": self.temperature,
            "train/i2t_loss": i2t_loss.item(),
            "train/t2i_loss": t2i_loss.item(),
            "train/t2t_loss": t2t_loss.item(),
            "train/clip_loss": clip_loss.item(),
            "train/max_per_sample_temperature": per_sample_temperature.max().item(),
            "train/min_per_sample_temperature": per_sample_temperature.min().item(),
            "train/median_per_sample_temperature": per_sample_temperature.median().item(),
            "train/quantile_0.5_per_sample_temperature": per_sample_temperature.float()
            .quantile(0.5)
            .item(),
            "train/positive_samples_min_temperature": temp_of_positives.min().item(),
            "train/positive_samples_max_temperature": temp_of_positives.max().item(),
            "train/positive_samples_avg_temperature": temp_of_positives.mean().item(),
            "train/positive_samples_median_temperature": temp_of_positives.median().item(),
            "train/positive_samples_quantile_0.5_temperature": temp_of_positives.float()
            .quantile(0.5)
            .item(),
            "train/negative_samples_min_temperature": temp_of_negatives.min().item(),
            "train/negative_samples_max_temperature": temp_of_negatives.max().item(),
            "train/negative_samples_avg_temperature": temp_of_negatives.mean().item(),
            "train/negative_samples_median_temperature": temp_of_negatives.median().item(),
            "train/negative_samples_quantile_0.5_temperature": temp_of_negatives.float()
            .quantile(0.5)
            .item(),
            "train/sim_blend_ratio": self.sim_blend_ratio,
        }

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class CLIP_MoE_Text_Supervision(nn.Module):
    def __init__(self, world_size=8, temperature=0.01, alpha=0.05):
        super(CLIP_MoE_Text_Supervision, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.alpha = alpha

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(
        self,
        image_features,
        text_features,
        text_expert_features,
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
            text_expert_features = torch.cat(
                GatherLayer.apply(text_expert_features), dim=0
            )

        labels = torch.arange(image_features.shape[0], device=image_features.device)

        clip_similarity = text_features @ image_features.T

        t2t_expert_sim = text_expert_features @ text_expert_features.T
        per_sample_temperature = self.temperature + self.alpha * torch.sqrt(
            (t2t_expert_sim + 1.0) / 2.0
        )
        t2i_loss = F.cross_entropy(clip_similarity / per_sample_temperature, labels)

        i2t_loss = F.cross_entropy((clip_similarity / self.temperature).t(), labels)

        total_loss = (i2t_loss + t2i_loss) / 2

        log_obj = {
            "train/temperature": self.temperature,
            "train/i2t_loss": i2t_loss.item(),
            "train/t2i_loss": t2i_loss.item(),
        }

        log_obj.update(get_temperature_statistics(per_sample_temperature))

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class Scheduled_CLIP_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        alpha=0.1,
        total_steps=None,
        clip_scheduled_loss_type="none",
        disable_temo_modulation=False,
    ):
        super(Scheduled_CLIP_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.alpha = alpha
        self.total_steps = total_steps
        self.clip_scheduled_loss_type = clip_scheduled_loss_type
        self.disable_temo_modulation = disable_temo_modulation

        assert (
            total_steps is not None
        ), "total_steps must be provided for scheduled loss"

        assert (
            self.clip_scheduled_loss_type != "none"
        ), "clip_scheduled_loss_type must be provided for scheduled loss"

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(
        self,
        image_features,
        text_features,
        current_step,
        include_alignment_loss=False,
        per_sample_temp_similarity="t2i",
        per_sample_temp_mapping="adaptive_with_base",
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        if per_sample_temp_similarity == "t2i":
            sim_for_temp = text_features @ image_features.T
        elif per_sample_temp_similarity == "i2t":
            sim_for_temp = image_features @ text_features.T
        elif per_sample_temp_similarity == "t2t":
            sim_for_temp = text_features @ text_features.T
        elif per_sample_temp_similarity == "i2i":
            sim_for_temp = image_features @ image_features.T
        else:
            raise ValueError(
                f"Unknown use_similarity: {per_sample_temp_similarity}. Use t2i, i2t, t2t or i2i"
            )

        # First, compute normal CLIP loss.
        sim = text_features @ image_features.T
        sim_clip_loss = sim / self.temperature
        labels = torch.arange(image_features.shape[0], device=image_features.device)

        clip_t2i_loss = F.cross_entropy(sim_clip_loss, labels)
        clip_i2t_loss = F.cross_entropy(sim_clip_loss.t(), labels)

        clip_total_loss = (clip_i2t_loss + clip_t2i_loss) / 2

        if self.disable_temo_modulation:
            per_sample_temperature = self.temperature
        else:
            print(f"Using per_sample_temp_mapping: {per_sample_temp_mapping}")
            if per_sample_temp_mapping == "adaptive_with_base":
                per_sample_temperature = self.temperature + self.alpha * torch.sqrt(
                    (sim_for_temp + 1.0) / 2.0
                )
            elif per_sample_temp_mapping == "adaptive_without_base":
                per_sample_temperature = (
                    self.alpha * torch.sqrt((sim_for_temp + 1.0) / 2.0) + 1e-9
                )
            elif per_sample_temp_mapping == "cosine":
                min_temperature = self.temperature
                max_temperature = self.alpha + self.temperature
                per_sample_temperature = min_temperature + 0.5 * (
                    max_temperature - min_temperature
                ) * (1.0 + torch.cos(torch.pi * (1.0 + sim_for_temp)))
            else:
                raise ValueError(
                    f"Unknown per_sample_temp_mapping: {per_sample_temp_mapping}. Use adaptive_with_base or adaptive_without_base"
                )

        sim_per_sample = sim_for_temp / per_sample_temperature
        labels = torch.arange(image_features.shape[0], device=image_features.device)

        sim_per_sample_t2i_loss = F.cross_entropy(sim_per_sample, labels)
        sim_per_sample_i2t_loss = F.cross_entropy(sim_per_sample.t(), labels)

        sim_total_loss = (sim_per_sample_i2t_loss + sim_per_sample_t2i_loss) / 2

        normalized_current_step = current_step / self.total_steps

        if self.clip_scheduled_loss_type == "quadratic":
            clip_loss_weight = (normalized_current_step - 1.0) ** 2
            sim_loss_weight = normalized_current_step**2
        elif self.clip_scheduled_loss_type == "linear":
            clip_loss_weight = 1.0 - normalized_current_step
            sim_loss_weight = normalized_current_step
        else:
            raise ValueError(
                f"Unknown clip_scheduled_loss_type: {self.clip_scheduled_loss_type}"
            )

        # Combine the two losses using a weighted sum
        total_loss = (
            clip_loss_weight * clip_total_loss + sim_loss_weight * sim_total_loss
        )

        if include_alignment_loss:
            alignment_loss = -torch.mean(torch.log(torch.diagonal((sim + 1.0) / 2.0)))
            total_loss += alignment_loss

        log_obj = {
            "train/temperature": self.temperature,
            "train/t2i_loss": clip_t2i_loss.item(),
            "train/i2t_loss": clip_i2t_loss.item(),
            "train/sim_t2i_loss": sim_per_sample_t2i_loss.item(),
            "train/sim_i2t_loss": sim_per_sample_i2t_loss.item(),
            "train/clip_loss_weight": clip_loss_weight,
            "train/sim_loss_weight": sim_loss_weight,
        }

        if include_alignment_loss:
            log_obj["train/alignment_loss"] = alignment_loss.item()

        log_obj.update(get_temperature_statistics(per_sample_temperature))

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class Scheduled_Crossmodal_CLIP_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        alpha=0.1,
        total_steps=None,
        clip_scheduled_loss_type="none",
        include_unimodal_loss=False,
    ):
        super(Scheduled_Crossmodal_CLIP_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.alpha = alpha
        self.total_steps = total_steps
        self.clip_scheduled_loss_type = clip_scheduled_loss_type
        self.include_unimodal_loss = include_unimodal_loss

        assert (
            total_steps is not None
        ), "total_steps must be provided for Scheduled_Crossmodal_CLIP_Loss"

        assert (
            self.clip_scheduled_loss_type != "none"
        ), "clip_scheduled_loss_type must be provided for Scheduled_Crossmodal_CLIP_Loss"

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(
        self,
        image_features,
        text_features,
        current_step,
        clip_loss_weight,
        sim_loss_weight,
        per_sample_temp_mapping="adaptive_with_base",
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # First, compute normal CLIP loss.
        sim_t2i = text_features @ image_features.T
        sim_clip_loss = sim_t2i / self.temperature
        labels = torch.arange(image_features.shape[0], device=image_features.device)

        clip_t2i_loss = F.cross_entropy(sim_clip_loss, labels)
        clip_i2t_loss = F.cross_entropy(sim_clip_loss.t(), labels)

        clip_total_loss = (clip_i2t_loss + clip_t2i_loss) / 2

        sim_i2i = None
        sim_t2t = None

        per_sample_temp_i2i = None
        per_sample_temp_t2t = None

        if self.include_unimodal_loss:
            sim_i2i = image_features @ image_features.T
            sim_t2t = text_features @ text_features.T

        if per_sample_temp_mapping == "adaptive_with_base":
            per_sample_temp_t2i = self.temperature + self.alpha * torch.sqrt(
                (sim_t2i + 1.0) / 2.0
            )

            per_sample_temp_i2t = self.temperature + self.alpha * torch.sqrt(
                (sim_t2i.t() + 1.0) / 2.0
            )

            if self.include_unimodal_loss:
                per_sample_temp_i2i = self.temperature + self.alpha * torch.sqrt(
                    (sim_i2i + 1.0) / 2.0
                )

                per_sample_temp_t2t = self.temperature + self.alpha * torch.sqrt(
                    (sim_t2t + 1.0) / 2.0
                )

        elif per_sample_temp_mapping == "adaptive_without_base":
            per_sample_temp_t2i = self.alpha * torch.sqrt((sim_t2i + 1.0) / 2.0)

            per_sample_temp_i2t = self.alpha * torch.sqrt((sim_t2i.t() + 1.0) / 2.0)

            if self.include_unimodal_loss:
                per_sample_temp_i2i = self.alpha * torch.sqrt((sim_i2i + 1.0) / 2.0)

                per_sample_temp_t2t = self.alpha * torch.sqrt((sim_t2t + 1.0) / 2.0)

        elif per_sample_temp_mapping == "cosine":
            min_temperature = self.temperature
            max_temperature = self.alpha + self.temperature

            per_sample_temp_t2i = min_temperature + 0.5 * (
                max_temperature - min_temperature
            ) * (1.0 + torch.cos(torch.pi * (1.0 + sim_t2i)))

            per_sample_temp_i2t = min_temperature + 0.5 * (
                max_temperature - min_temperature
            ) * (1.0 + torch.cos(torch.pi * (1.0 + sim_t2i.t())))

            if self.include_unimodal_loss:
                per_sample_temp_i2i = min_temperature + 0.5 * (
                    max_temperature - min_temperature
                ) * (1.0 + torch.cos(torch.pi * (1.0 + sim_i2i)))

                per_sample_temp_t2t = min_temperature + 0.5 * (
                    max_temperature - min_temperature
                ) * (1.0 + torch.cos(torch.pi * (1.0 + sim_t2t)))

        else:
            raise ValueError(
                f"Unknown per_sample_temp_mapping: {per_sample_temp_mapping}. Use adaptive_with_base, adaptive_without_base or cosine"
            )

        labels = torch.arange(image_features.shape[0], device=image_features.device)

        sim_per_sample_t2i_loss = F.cross_entropy(sim_t2i / per_sample_temp_t2i, labels)
        sim_per_sample_i2t_loss = F.cross_entropy(
            sim_t2i.t() / per_sample_temp_i2t, labels
        )

        sim_total_loss = (sim_per_sample_i2t_loss + sim_per_sample_t2i_loss) / 2

        if self.include_unimodal_loss:
            sim_per_sample_i2i_loss = F.cross_entropy(
                sim_i2i / per_sample_temp_i2i, labels
            )

            sim_per_sample_t2t_loss = F.cross_entropy(
                sim_t2t / per_sample_temp_t2t, labels
            )

            sim_total_loss += sim_per_sample_i2i_loss
            sim_total_loss += sim_per_sample_t2t_loss

        normalized_current_step = current_step / self.total_steps

        if self.clip_scheduled_loss_type == "quadratic":
            clip_loss_weight = (normalized_current_step - 1.0) ** 2
            sim_loss_weight = normalized_current_step**2
        elif self.clip_scheduled_loss_type == "linear":
            clip_loss_weight = 1.0 - normalized_current_step
            sim_loss_weight = normalized_current_step
        elif self.clip_scheduled_loss_type == "fixed":
            assert (
                clip_loss_weight is not None and sim_loss_weight is not None
            ), "clip_loss_weight and sim_loss_weight must be provided for `fixed` scheduled loss"
        else:
            raise ValueError(
                f"Unknown clip_scheduled_loss_type: {self.clip_scheduled_loss_type}"
            )

        # Combine the two losses using a weighted sum
        total_loss = (
            clip_loss_weight * clip_total_loss + sim_loss_weight * sim_total_loss
        )

        log_obj = {
            "train/temperature": self.temperature,
            "train/t2i_loss": clip_t2i_loss.item(),
            "train/i2t_loss": clip_i2t_loss.item(),
            "train/sim_t2i_loss": sim_per_sample_t2i_loss.item(),
            "train/sim_i2t_loss": sim_per_sample_i2t_loss.item(),
            "train/clip_loss_weight": clip_loss_weight,
            "train/sim_loss_weight": sim_loss_weight,
        }

        log_obj.update(
            get_temperature_statistics(per_sample_temp_i2t, prefix="train/i2t/")
        )

        log_obj.update(
            get_temperature_statistics(per_sample_temp_t2i, prefix="train/t2i/")
        )

        if self.include_unimodal_loss:
            log_obj.update(
                get_temperature_statistics(per_sample_temp_i2i, prefix="train/i2i/")
            )

            log_obj.update(
                get_temperature_statistics(per_sample_temp_t2t, prefix="train/t2t/")
            )

            log_obj["train/sim_i2i_loss"] = sim_per_sample_i2i_loss.item()
            log_obj["train/sim_t2t_loss"] = sim_per_sample_t2t_loss.item()

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class Scheduled_Crossmodal_CLIP_With_Augmentations_And_Unimodal_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        alpha=0.1,
        total_steps=None,
        clip_scheduled_loss_type="none",
        disable_temo_modulation=False,
        disable_crossmodal_minfonce=False,
        disable_i2i_temo_loss=False,
        disable_t2t_temo_loss=False,
        reversed_scheduler=False,
        enable_non_modulated_unimodal_losses=False,
    ):
        super(
            Scheduled_Crossmodal_CLIP_With_Augmentations_And_Unimodal_Loss, self
        ).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.alpha = alpha
        self.total_steps = total_steps
        self.clip_scheduled_loss_type = clip_scheduled_loss_type
        self.disable_temo_modulation = disable_temo_modulation
        self.disable_crossmodal_minfonce = disable_crossmodal_minfonce

        self.disable_i2i_temo_loss = disable_i2i_temo_loss
        self.disable_t2t_temo_loss = disable_t2t_temo_loss

        self.reversed_scheduler = reversed_scheduler

        self.enable_non_modulated_unimodal_losses = enable_non_modulated_unimodal_losses

        print(f"===> Using disable_temo_modulation: {disable_temo_modulation}")

        print(f"===> Using enable_non_modulated_unimodal_losses: {enable_non_modulated_unimodal_losses}")

        assert (
            total_steps is not None
        ), "total_steps must be provided for Scheduled_Crossmodal_CLIP_Loss"

        assert (
            self.clip_scheduled_loss_type != "none"
        ), "clip_scheduled_loss_type must be provided for Scheduled_Crossmodal_CLIP_Loss"

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(
        self,
        image_features,
        text_features,
        augmented_image_features,
        augmented_text_features,
        current_step,
        clip_loss_weight,
        sim_loss_weight,
    ):

        local_temperature = 0.01
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

            augmented_image_features = torch.cat(
                GatherLayer.apply(augmented_image_features), dim=0
            )

            augmented_text_features = torch.cat(
                GatherLayer.apply(augmented_text_features), dim=0
            )

        # First, compute normal CLIP loss.
        sim_t2i = text_features @ image_features.T
        sim_clip_loss = sim_t2i / self.temperature
        labels = torch.arange(image_features.shape[0], device=image_features.device)

        clip_t2i_loss = F.cross_entropy(sim_clip_loss, labels)
        clip_i2t_loss = F.cross_entropy(sim_clip_loss.t(), labels)

        info_nce_loss = (clip_i2t_loss + clip_t2i_loss) / 2

        total_non_modulated = info_nce_loss

        if self.enable_non_modulated_unimodal_losses:
            i2i_similarity = image_features @ augmented_image_features.T
            t2t_similarity = text_features @ augmented_text_features.T

            i2i_loss = F.cross_entropy(i2i_similarity / self.temperature, labels)
            t2t_loss = F.cross_entropy(t2t_similarity / self.temperature, labels)

            total_non_modulated += (i2i_loss + t2t_loss)

        # First, compute the modulated InfoNCE loss components
        if self.disable_crossmodal_minfonce:
            modulated_info_nce_loss = 0.0
        else:
            if self.disable_temo_modulation:
                per_sample_temp_t2i = self.temperature
                per_sample_temp_i2t = self.temperature
            else:
                per_sample_temp_t2i = local_temperature + self.alpha * torch.sqrt(
                    (sim_t2i + 1.0) / 2.0
                )

                per_sample_temp_i2t = local_temperature + self.alpha * torch.sqrt(
                    (sim_t2i.t() + 1.0) / 2.0
                )

            modulated_t2i_loss = F.cross_entropy(sim_t2i / per_sample_temp_t2i, labels)
            modulated_i2t_loss = F.cross_entropy(
                sim_t2i.t() / per_sample_temp_i2t, labels
            )

            modulated_info_nce_loss = (modulated_t2i_loss + modulated_i2t_loss) / 2.0

        if self.disable_i2i_temo_loss:
            modulated_i2i_loss = 0.0
        else:
            # Next, compute the modulated unimodal loss components based on the original and augmented features
            sim_i2i = image_features @ augmented_image_features.T

            if self.disable_temo_modulation:
                per_sample_temp_i2i = self.temperature
            else:
                per_sample_temp_i2i = local_temperature + self.alpha * torch.sqrt(
                    (sim_i2i + 1.0) / 2.0
                )

            modulated_i2i_loss = F.cross_entropy(sim_i2i / per_sample_temp_i2i, labels)

        if self.disable_t2t_temo_loss:
            modulated_t2t_loss = 0.0
        else:
            sim_t2t = text_features @ augmented_text_features.T

            if self.disable_temo_modulation:
                per_sample_temp_t2t = self.temperature
            else:
                per_sample_temp_t2t = local_temperature + self.alpha * torch.sqrt(
                    (sim_t2t + 1.0) / 2.0
                )

            modulated_t2t_loss = F.cross_entropy(sim_t2t / per_sample_temp_t2t, labels)

            modulated_unimodal_loss = modulated_i2i_loss + modulated_t2t_loss

        # Compute the loss weights
        normalized_current_step = current_step / self.total_steps

        if self.clip_scheduled_loss_type == "quadratic":
            info_nce_loss_weight = (normalized_current_step - 1.0) ** 2
            modulated_unimodal_loss_weight = normalized_current_step**2
        elif self.clip_scheduled_loss_type == "linear":
            info_nce_loss_weight = 1.0 - normalized_current_step
            modulated_unimodal_loss_weight = normalized_current_step
        elif self.clip_scheduled_loss_type == "fixed":
            assert (
                clip_loss_weight is not None and sim_loss_weight is not None
            ), "clip_loss_weight and sim_loss_weight must be provided for `fixed` scheduled loss"

            info_nce_loss_weight = clip_loss_weight
            modulated_unimodal_loss_weight = sim_loss_weight
        else:
            raise ValueError(
                f"Unknown clip_scheduled_loss_type: {self.clip_scheduled_loss_type}"
            )

        if self.reversed_scheduler:
            info_nce_loss_weight, modulated_unimodal_loss_weight = (
                modulated_unimodal_loss_weight,
                info_nce_loss_weight,
            )

        # Combine the two losses using a weighted sum
        total_loss = (
            info_nce_loss_weight * total_non_modulated
            + modulated_unimodal_loss_weight
            * (modulated_info_nce_loss + modulated_unimodal_loss)
        )

        log_obj = {
            "train/local_temperature": local_temperature,
            "train/temperature": self.temperature,
            "train/t2i_loss": clip_t2i_loss.item(),
            "train/i2t_loss": clip_i2t_loss.item(),
            "train/sim_t2i_loss": modulated_t2i_loss.item(),
            "train/sim_i2t_loss": modulated_i2t_loss.item(),
            "train/sim_i2i_loss": modulated_i2i_loss.item(),
            "train/sim_t2t_loss": modulated_t2t_loss.item(),
            "train/clip_loss_weight": info_nce_loss_weight,
            "train/sim_loss_weight": modulated_unimodal_loss_weight,
            "train/total_non_modulated": total_non_modulated.item(),
        }

        log_obj.update(
            get_temperature_statistics(per_sample_temp_i2t, prefix="train/i2t/")
        )

        log_obj.update(
            get_temperature_statistics(per_sample_temp_t2i, prefix="train/t2i/")
        )

        log_obj.update(
            get_temperature_statistics(per_sample_temp_i2i, prefix="train/i2i/")
        )

        log_obj.update(
            get_temperature_statistics(per_sample_temp_t2t, prefix="train/t2t/")
        )

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class Scheduled_Crossmodal_Cosine_CLIP_With_Augmentations_And_Unimodal_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        i2i_temperature=0.01,
        t2t_temperature=0.01,
        total_steps=None,
        clip_scheduled_loss_type="none",
    ):
        super(
            Scheduled_Crossmodal_Cosine_CLIP_With_Augmentations_And_Unimodal_Loss, self
        ).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.total_steps = total_steps
        self.clip_scheduled_loss_type = clip_scheduled_loss_type

        self.i2i_temperature = i2i_temperature
        self.t2t_temperature = t2t_temperature

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_i2i_temperature(self, i2i_temperature):
        self.i2i_temperature = i2i_temperature

    def set_t2t_temperature(self, t2t_temperature):
        self.t2t_temperature = t2t_temperature

    def forward(
        self,
        image_features,
        text_features,
        augmented_image_features,
        augmented_text_features,
        current_step,
        i2i_loss_weight=1.0,
        t2t_loss_weight=1.0,
        exclude_modulated_info_nce_loss=False,
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

            augmented_image_features = torch.cat(
                GatherLayer.apply(augmented_image_features), dim=0
            )

            augmented_text_features = torch.cat(
                GatherLayer.apply(augmented_text_features), dim=0
            )

        # First, compute normal CLIP loss.
        sim_t2i = text_features @ image_features.T
        sim_clip_loss = sim_t2i / self.temperature
        labels = torch.arange(image_features.shape[0], device=image_features.device)

        clip_t2i_loss = F.cross_entropy(sim_clip_loss, labels)
        clip_i2t_loss = F.cross_entropy(sim_clip_loss.t(), labels)

        info_nce_loss = (clip_i2t_loss + clip_t2i_loss) / 2

        # We can use the same temperature scheduler for both the crossmodal and unimodal losses
        if not exclude_modulated_info_nce_loss:
            modulated_crossmodal_temperature = self.i2i_temperature

            modulated_t2i_loss = F.cross_entropy(
                sim_t2i / modulated_crossmodal_temperature, labels
            )
            modulated_i2t_loss = F.cross_entropy(
                sim_t2i.t() / modulated_crossmodal_temperature, labels
            )

            modulated_info_nce_loss = (modulated_t2i_loss + modulated_i2t_loss) / 2.0
        else:
            modulated_info_nce_loss = 0.0

        # Next, compute the modulated unimodal loss components based on the original and augmented features
        sim_i2i = image_features @ augmented_image_features.T
        modulated_i2i_loss = F.cross_entropy(sim_i2i / self.i2i_temperature, labels)

        sim_t2t = text_features @ augmented_text_features.T
        modulated_t2t_loss = F.cross_entropy(sim_t2t / self.t2t_temperature, labels)

        modulated_unimodal_loss = (
            i2i_loss_weight * modulated_i2i_loss + t2t_loss_weight * modulated_t2t_loss
        )

        # Compute the loss weights
        normalized_current_step = current_step / self.total_steps

        if self.clip_scheduled_loss_type == "quadratic":
            info_nce_loss_weight = (normalized_current_step - 1.0) ** 2
            modulated_unimodal_loss_weight = normalized_current_step**2
        elif self.clip_scheduled_loss_type == "linear":
            info_nce_loss_weight = 1.0 - normalized_current_step
            modulated_unimodal_loss_weight = normalized_current_step
        else:
            raise ValueError(
                f"Unknown clip_scheduled_loss_type: {self.clip_scheduled_loss_type}"
            )

        # Combine the two losses using a weighted sum
        total_loss = (
            info_nce_loss_weight * info_nce_loss
            + modulated_unimodal_loss_weight
            * (modulated_info_nce_loss + modulated_unimodal_loss)
        )

        log_obj = {
            "train/temperature": self.temperature,
            "train/i2i_temperature": self.i2i_temperature,
            "train/t2t_temperature": self.t2t_temperature,
            "train/t2i_loss": clip_t2i_loss.item(),
            "train/i2t_loss": clip_i2t_loss.item(),
            "train/clip_loss_weight": info_nce_loss_weight,
            "train/sim_loss_weight": modulated_unimodal_loss_weight,
            "train/i2i_loss_weight": i2i_loss_weight,
            "train/t2t_loss_weight": t2t_loss_weight,
        }

        if not exclude_modulated_info_nce_loss:
            log_obj.update(
                {
                    "train/sim_i2i_loss": modulated_i2i_loss.item(),
                    "train/sim_t2t_loss": modulated_t2t_loss.item(),
                }
            )

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class Scheduled_Crossmodal_With_Augmentations_CLIP_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        alpha=0.1,
        total_steps=None,
    ):
        super(Scheduled_Crossmodal_With_Augmentations_CLIP_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.alpha = alpha
        self.total_steps = total_steps

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(
        self,
        image_features,
        text_features,
        augmented_image_features,
        augmented_text_features,
        current_step,
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

            augmented_image_features = torch.cat(
                GatherLayer.apply(augmented_image_features), dim=0
            )

            augmented_text_features = torch.cat(
                GatherLayer.apply(augmented_text_features), dim=0
            )

        # First, compute normal CLIP loss.
        sim_t2i = text_features @ image_features.T
        sim_clip_loss = sim_t2i / self.temperature
        labels = torch.arange(image_features.shape[0], device=image_features.device)

        clip_t2i_loss = F.cross_entropy(sim_clip_loss, labels)
        clip_i2t_loss = F.cross_entropy(sim_clip_loss.t(), labels)

        clip_total_loss = (clip_i2t_loss + clip_t2i_loss) / 2

        sim_t2i_expert = augmented_text_features @ augmented_image_features.T
        per_sample_temp_t2i = self.temperature + self.alpha * torch.sqrt(
            (sim_t2i_expert + 1.0) / 2.0
        )

        per_sample_temp_i2t = self.temperature + self.alpha * torch.sqrt(
            (sim_t2i_expert.t() + 1.0) / 2.0
        )

        sim_per_sample_t2i_loss = F.cross_entropy(
            sim_t2i_expert / per_sample_temp_t2i, labels
        )
        sim_per_sample_i2t_loss = F.cross_entropy(
            sim_t2i_expert.t() / per_sample_temp_i2t, labels
        )

        sim_total_loss = (sim_per_sample_i2t_loss + sim_per_sample_t2i_loss) / 2

        normalized_current_step = current_step / self.total_steps
        clip_loss_weight = (normalized_current_step - 1.0) ** 2
        sim_loss_weight = normalized_current_step**2

        # Combine the two losses using a weighted sum
        total_loss = (
            clip_loss_weight * clip_total_loss + sim_loss_weight * sim_total_loss
        )

        log_obj = {
            "train/temperature": self.temperature,
            "train/t2i_loss": clip_t2i_loss.item(),
            "train/i2t_loss": clip_i2t_loss.item(),
            "train/sim_t2i_loss": sim_per_sample_t2i_loss.item(),
            "train/sim_i2t_loss": sim_per_sample_i2t_loss.item(),
            "train/clip_loss_weight": clip_loss_weight,
            "train/sim_loss_weight": sim_loss_weight,
        }

        log_obj.update(
            get_temperature_statistics(per_sample_temp_i2t, prefix="train/i2t/")
        )

        log_obj.update(
            get_temperature_statistics(per_sample_temp_t2i, prefix="train/t2i/")
        )

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class CLIP_Loss_PCT(nn.Module):
    def __init__(self, world_size=8, temperature=0.01):
        super(CLIP_Loss_PCT, self).__init__()
        self.world_size = world_size
        self.temperature = temperature

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(self, image_features, text_features, per_sample_temperature):
        temperature = self.temperature + per_sample_temperature

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        sim = (
            torch.einsum("i d, j d -> i j", text_features, image_features) / temperature
        )
        labels = torch.arange(image_features.shape[0], device=image_features.device)

        i2t_loss = F.cross_entropy(sim.t(), labels)
        t2i_loss = F.cross_entropy(sim, labels)

        total_loss = (i2t_loss + t2i_loss) / 2

        if utils.is_main_process():
            wandb.log(
                {
                    "train/temperature": self.temperature,
                    "train/t2i_loss": t2i_loss.item(),
                    "train/i2t_loss": i2t_loss.item(),
                },
                step=wandb.run.step,
            )

        return total_loss


class SogCLR_Loss(nn.Module):
    def __init__(
        self,
        N=2900000,
        gamma=0.1,
        temperature=0.07,
        world_size=8,
        use_per_sample_temp=False,
        alpha=0.05,
    ):

        # Inputs:
        #   N is number of samples in training set

        super(SogCLR_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.gamma = gamma
        self.temperature = temperature
        self.eps = 1e-14
        self.use_per_sample_temp = use_per_sample_temp
        self.alpha = alpha

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_i2i_temperature(self, i2i_temperature):
        pass

    def set_t2t_temperature(self, t2t_temperature):
        pass

    def forward(self, image_features, text_features, image_ids, text_ids, epoch):

        # Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

            # image_features = image_features.contiguous()
            # text_features = text_features.contiguous()
            # image_features_all = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0).contiguous()
            # text_features_all = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0).contiguous()

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum("i d, j d -> i j", image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]
        mask_neg = (1.0 - torch.eye(batch_size)).to(sim.device)

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        log_obj = {"train/temperature": self.temperature}

        if self.use_per_sample_temp:
            # Compute per-sample temperatures
            per_sample_temprature_i2t = self.temperature + self.alpha * torch.sqrt(
                (sim + 1.0) / 2.0
            )

            per_sample_temperature_t2i = self.temperature + self.alpha * torch.sqrt(
                (sim.t() + 1.0) / 2.0
            )

            image_diffs_d_temps = (
                (image_diffs / per_sample_temprature_i2t).clone().detach_()
            )
            text_diffs_d_temps = (
                (text_diffs / per_sample_temperature_t2i).clone().detach_()
            )

            log_obj.update(
                get_temperature_statistics(
                    per_sample_temprature_i2t, prefix="train/i2t/"
                )
            )

            log_obj.update(
                get_temperature_statistics(
                    per_sample_temperature_t2i, prefix="train/t2i/"
                )
            )
        else:
            image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
            text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = (
            torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * mask_neg
        )  # -b to avoid exp operation overflow
        exp_text_diffs = (
            torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * mask_neg
        )

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size - 1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size - 1)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0 - self.gamma) * self.s_I[image_ids] * torch.exp(
                old_b_I - self.b_I[image_ids]
            ) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T[text_ids] * torch.exp(
                old_b_T - self.b_T[text_ids]
            ) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (
            batch_size - 1
        )
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (
            batch_size - 1
        )

        image_loss = image_loss.mean()
        text_loss = text_loss.mean()

        total_loss = image_loss + text_loss

        log_obj.update(
            {
                "train/i2t_loss": image_loss.item(),
                "train/t2i_loss": text_loss.item(),
            }
        )

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class SogCLR_With_Cosine_And_Unimodal_Loss(nn.Module):
    def __init__(
        self,
        N=2900000,
        gamma=0.1,
        temperature=0.07,
        world_size=8,
        i2i_temperature=0.07,
        t2t_temperature=0.07,
    ):
        super(SogCLR_With_Cosine_And_Unimodal_Loss, self).__init__()

        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()

        self.gamma = gamma
        self.temperature = temperature
        self.eps = 1e-14

        self.i2i_temperature = i2i_temperature
        self.t2t_temperature = t2t_temperature

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_i2i_temperature(self, i2i_temperature):
        self.i2i_temperature = i2i_temperature

    def set_t2t_temperature(self, t2t_temperature):
        self.t2t_temperature = t2t_temperature

    def forward(
        self,
        image_features,
        text_features,
        augmented_image_features,
        augmented_text_features,
        image_ids,
        text_ids,
        epoch,
        i2i_loss_weight=1.0,
        t2t_loss_weight=1.0,
    ):

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

            augmented_image_features = torch.cat(
                GatherLayer.apply(augmented_image_features), dim=0
            )

            augmented_text_features = torch.cat(
                GatherLayer.apply(augmented_text_features), dim=0
            )

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum("i d, j d -> i j", image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]
        mask_neg = (1.0 - torch.eye(batch_size)).to(sim.device)

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = (
            torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * mask_neg
        )  # -b to avoid exp operation overflow
        exp_text_diffs = (
            torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * mask_neg
        )

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size - 1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size - 1)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0 - self.gamma) * self.s_I[image_ids] * torch.exp(
                old_b_I - self.b_I[image_ids]
            ) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T[text_ids] * torch.exp(
                old_b_T - self.b_T[text_ids]
            ) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (
            batch_size - 1
        )
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (
            batch_size - 1
        )

        image_loss = image_loss.mean()
        text_loss = text_loss.mean()

        sogclr_loss = image_loss + text_loss

        # Compute cosine modulated loss
        labels = torch.arange(image_features.shape[0], device=image_features.device)
        i2i_similarity = image_features @ augmented_image_features.t()
        t2t_similarity = text_features @ augmented_text_features.t()

        i2i_loss = F.cross_entropy(i2i_similarity / self.i2i_temperature, labels)

        t2t_loss = F.cross_entropy(t2t_similarity / self.t2t_temperature, labels)

        modulated_loss = i2i_loss * i2i_loss_weight + t2t_loss * t2t_loss_weight

        total_loss = sogclr_loss + modulated_loss

        log_obj = {}
        log_obj.update(
            {
                "train/i2t_loss": image_loss.item(),
                "train/t2i_loss": text_loss.item(),
                "train/sogclr_loss": sogclr_loss.item(),
                "train/modulated_loss": modulated_loss.item(),
                "train/i2i_loss_weight": i2i_loss_weight,
                "train/t2t_loss_weight": t2t_loss_weight,
                "train/i2i_temperature": self.i2i_temperature,
                "train/t2t_temperature": self.t2t_temperature,
                "train/temperature": self.temperature,
            }
        )

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


class Scheduled_SogCLR_Crossmodal_Loss(nn.Module):
    def __init__(
        self,
        N=2900000,
        gamma=0.1,
        temperature=0.07,
        world_size=8,
        alpha=0.05,
        total_steps=None,
    ):

        super(Scheduled_SogCLR_Crossmodal_Loss, self).__init__()
        self.world_size = world_size

        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()

        self.s_I_scheduled = torch.zeros(N).cuda()
        self.s_T_scheduled = torch.zeros(N).cuda()
        self.b_I_scheduled = torch.zeros(N).cuda()
        self.b_T_scheduled = torch.zeros(N).cuda()

        self.gamma = gamma
        self.temperature = temperature
        self.eps = 1e-14
        self.alpha = alpha
        self.total_steps = total_steps

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_i2i_temperature(self, i2i_temperature):
        pass

    def set_t2t_temperature(self, t2t_temperature):
        pass

    def forward(
        self, image_features, text_features, image_ids, text_ids, epoch, current_step
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum("i d, j d -> i j", image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]
        mask_neg = (1.0 - torch.eye(batch_size)).to(sim.device)

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = (
            torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * mask_neg
        )  # -b to avoid exp operation overflow
        exp_text_diffs = (
            torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * mask_neg
        )

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size - 1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size - 1)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0 - self.gamma) * self.s_I[image_ids] * torch.exp(
                old_b_I - self.b_I[image_ids]
            ) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T[text_ids] * torch.exp(
                old_b_T - self.b_T[text_ids]
            ) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (
            batch_size - 1
        )
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (
            batch_size - 1
        )

        image_loss = image_loss.mean()
        text_loss = text_loss.mean()

        standard_sogclr_loss = image_loss + text_loss

        # Compute the similarity based loss
        per_sample_temperature_i2t = self.temperature + self.alpha * torch.sqrt(
            (sim + 1.0) / 2.0
        )

        per_sample_temperature_t2i = self.temperature + self.alpha * torch.sqrt(
            (sim.t() + 1.0) / 2.0
        )

        image_diffs_d_temps = (
            (image_diffs / per_sample_temperature_i2t).clone().detach_()
        )
        text_diffs_d_temps = (text_diffs / per_sample_temperature_t2i).clone().detach_()

        # update b
        old_b_I = self.b_I_scheduled[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I_scheduled[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T_scheduled[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T_scheduled[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = (
            torch.exp(image_diffs_d_temps - self.b_I_scheduled[image_ids][:, None])
            * mask_neg
        )  # -b to avoid exp operation overflow
        exp_text_diffs = (
            torch.exp(text_diffs_d_temps - self.b_T_scheduled[text_ids][None, :])
            * mask_neg
        )

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size - 1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size - 1)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0 - self.gamma) * self.s_I_scheduled[image_ids] * torch.exp(
                old_b_I - self.b_I_scheduled[image_ids]
            ) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T_scheduled[text_ids] * torch.exp(
                old_b_T - self.b_T_scheduled[text_ids]
            ) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I_scheduled[image_ids] = s_I.squeeze()
        self.s_T_scheduled[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        sim_image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (
            batch_size - 1
        )

        sim_text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (
            batch_size - 1
        )

        sim_image_loss = sim_image_loss.mean()
        sim_text_loss = sim_text_loss.mean()

        modulated_loss = sim_image_loss + sim_text_loss

        # Compute the weights
        normalized_current_step = current_step / self.total_steps
        sogclr_loss_weight = (normalized_current_step - 1.0) ** 2
        sim_loss_weight = normalized_current_step**2

        # Combine the two losses
        total_loss = (
            sogclr_loss_weight * standard_sogclr_loss + sim_loss_weight * modulated_loss
        )

        log_obj = {
            "train/temperature": self.temperature,
            "train/i2t_loss": image_loss.item(),
            "train/t2i_loss": text_loss.item(),
            "train/sogclr_loss": standard_sogclr_loss.item(),
            "train/sim_i2t_loss": sim_image_loss,
            "train/sim_t2i_loss": sim_text_loss,
            "train/sim_loss": modulated_loss,
            "train/sogclr_loss_weight": sogclr_loss_weight,
            "train/sim_loss_weight": sim_loss_weight,
        }

        log_obj.update(
            get_temperature_statistics(per_sample_temperature_i2t, prefix="train/i2t/")
        )

        log_obj.update(
            get_temperature_statistics(per_sample_temperature_t2i, prefix="train/t2i/")
        )

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss


"""
    https://github.com/goel-shashank/CyCLIP/blob/52d77af2a5f1a4bff01b4c371d6b98e2d0340137/src/train.py
"""


class CyCLIP_Loss(nn.Module):
    def __init__(self, world_size, temperature, cylambda_1=0.25, cylambda_2=0.25):
        super(CyCLIP_Loss, self).__init__()

        self.world_size = world_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.cylambda_1 = cylambda_1
        self.cylambda_2 = cylambda_2

    def forward(self, image_features, text_features):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(image_features)

        logits_text_per_image = (image_features @ text_features.t()) / self.temperature
        logits_image_per_text = logits_text_per_image.t()

        target = torch.arange(batch_size).long().cuda()

        # contrastive loss, the same as CLIP
        contrastive_loss = (
            self.criterion(logits_text_per_image, target)
            + self.criterion(logits_image_per_text, target)
        ) / 2.0

        # inmodal_cyclic_loss
        logits_image_per_image = (
            image_features @ image_features.t()
        ) / self.temperature
        logits_text_per_text = (text_features @ text_features.t()) / self.temperature
        inmodal_cyclic_loss = (
            (logits_image_per_image - logits_text_per_text).square().mean()
            * (self.temperature**2)
            * batch_size
        )

        # crossmodal_cyclic_loss
        crossmodal_cyclic_loss = (
            (logits_text_per_image - logits_image_per_text).square().mean()
            * (self.temperature**2)
            * batch_size
        )

        loss = (
            contrastive_loss
            + self.cylambda_1 * inmodal_cyclic_loss
            + self.cylambda_2 * crossmodal_cyclic_loss
        )

        return loss


"""
    VICReg
    https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
"""


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICReg_Loss(nn.Module):
    def __init__(
        self, world_size, dim_size, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0
    ):
        super(VICReg_Loss, self).__init__()

        self.world_size = world_size
        self.dim_size = dim_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, image_features, text_features):
        if self.world_size > 1:
            x = torch.cat(GatherLayer.apply(image_features), dim=0)
            y = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(x)

        repr_loss = F.mse_loss(x, y)  # invariance term

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = (
            torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        )  # variance term

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.dim_size) + off_diagonal(
            cov_y
        ).pow_(2).sum().div(
            self.dim_size
        )  # covariance term

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss


class iSogCLR_Loss(nn.Module):
    def __init__(
        self, N=2900000, gamma=0.8, tau_init=0.01, world_size=8, bsz=128, rho=8.0
    ):

        # Inputs:
        #   N is number of samples in training set

        super(iSogCLR_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.gamma = gamma
        self.eps = 1e-14

        self.tau_min, self.tau_max = 0.005, 0.05
        self.rho = rho

        self.eta_init = 0.03

        self.beta_u = 0.5
        self.grad_clip = 5.0
        self.tau_I = torch.ones(N).cuda() * tau_init
        self.tau_T = torch.ones(N).cuda() * tau_init
        self.u_I = torch.zeros(N).cuda()
        self.u_T = torch.zeros(N).cuda()

    def forward(
        self, image_features, text_features, image_ids, text_ids, epoch, max_epoch
    ):

        # Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum("i d, j d -> i j", image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]
        mask_neg = (1.0 - torch.eye(batch_size)).cuda()

        # generate temperatures
        tau_image = self.tau_I[image_ids]
        tau_text = self.tau_T[text_ids]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).detach()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).detach()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = (
            torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * mask_neg
        )  # -b to avoid exp operation overflow
        exp_text_diffs = (
            torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * mask_neg
        )

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0 - self.gamma) * self.s_I[image_ids] * torch.exp(
                old_b_I - self.b_I[image_ids]
            ) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T[text_ids] * torch.exp(
                old_b_T - self.b_T[text_ids]
            ) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True)

        total_loss = image_loss.mean() + text_loss.mean()

        temp_weight_image = (
            torch.log(s_I / (batch_size - 1))
            + self.b_I[image_ids][:, None]
            + self.rho
            - torch.sum(weights_image * image_diffs_d_temps, dim=1, keepdim=True)
        )
        temp_weight_text = (
            torch.log(s_T / (batch_size - 1))
            + self.b_T[text_ids][None, :]
            + self.rho
            - torch.sum(weights_text * text_diffs_d_temps, dim=0, keepdim=True)
        )

        self.u_I[image_ids] = (1.0 - self.beta_u) * self.u_I[
            image_ids
        ] + self.beta_u * temp_weight_image.squeeze().clamp_(
            min=-self.grad_clip, max=self.grad_clip
        )
        self.u_T[text_ids] = (1.0 - self.beta_u) * self.u_T[
            text_ids
        ] + self.beta_u * temp_weight_text.squeeze().clamp_(
            min=-self.grad_clip, max=self.grad_clip
        )

        self.tau_I[image_ids] = (
            tau_image - self.eta_init * self.u_I[image_ids]
        ).clamp_(min=self.tau_min, max=self.tau_max)
        self.tau_T[text_ids] = (tau_text - self.eta_init * self.u_T[text_ids]).clamp_(
            min=self.tau_min, max=self.tau_max
        )

        return (
            total_loss,
            tau_image.cpu().detach().numpy(),
            tau_text.cpu().detach().numpy(),
        )


class TempGenerator(torch.nn.Module):
    def __init__(self, feature_dim, M=256, tau_min=0.005, dropout_rate=0.5, rho=6.0):
        super(TempGenerator, self).__init__()

        self.feature_dim = feature_dim
        self.M = M
        self.tau_min = tau_min
        self.tau_max = 0.05
        self.rho = rho

        self.proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.scaler = nn.Parameter(torch.tensor(np.log(0.01)))

        self.prototypes = nn.Parameter(torch.empty((self.M, self.feature_dim)))
        nn.init.normal_(self.prototypes, 0.0, 1.0)

        self.linear_1 = nn.Linear(self.M, 1)
        self.linear_1.weight.data.fill_(1.0)
        self.linear_1.bias.data.fill_(0.0)

        self.dropout = nn.Dropout(dropout_rate)

    def _init_prototypes(self, feats):
        self.prototypes.data.copy_(feats)

    def forward(self, x, return_feats=False):

        x = self.dropout(torch.sigmoid(self.proj(x)))

        if return_feats:
            return x

        normed_protos = F.normalize(self.prototypes, p=2.0, dim=1)

        prods = x @ normed_protos.t()

        weights = nn.Softmax(dim=1)(prods / self.scaler.exp())
        sims = torch.sigmoid(prods)

        tau = self.linear_1((weights - 1.0 / self.M) * sims).squeeze()

        return (self.tau_max - self.tau_min) * torch.sigmoid(tau) + self.tau_min


class iSogCLR_TempNet_Loss(nn.Module):  # using TempGenerator
    def __init__(
        self, N=2900000, gamma=0.8, world_size=8, rho=8.0, feature_dim=256, tau_min=0.01
    ):  # use temperature network

        # Inputs:
        #   N is number of samples in training set

        super(iSogCLR_TempNet_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.gamma = gamma
        self.eps = 1e-14

        self.rho = rho

        self.image_temp_gen = TempGenerator(
            feature_dim=feature_dim, rho=self.rho
        ).cuda()
        self.text_temp_gen = TempGenerator(feature_dim=feature_dim, rho=self.rho).cuda()

    def forward(
        self, image_features, text_features, image_ids, text_ids, epoch, max_epoch
    ):

        # Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum("i d, j d -> i j", image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]
        mask_neg = (1.0 - torch.eye(batch_size)).cuda()

        # generate temperatures
        tau_image = self.image_temp_gen(image_features.detach())
        tau_text = self.text_temp_gen(text_features.detach())

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).detach()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).detach()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = (
            torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * mask_neg
        )  # -b to avoid exp operation overflow
        exp_text_diffs = (
            torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * mask_neg
        )

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0 - self.gamma) * self.s_I[image_ids] * torch.exp(
                old_b_I - self.b_I[image_ids]
            ) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T[text_ids] * torch.exp(
                old_b_T - self.b_T[text_ids]
            ) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True)

        clip_loss = image_loss.mean() + text_loss.mean()

        temp_weight_image = (
            torch.log(s_I / (batch_size - 1))
            + self.b_I[image_ids][:, None]
            + self.rho
            - torch.sum(weights_image * image_diffs_d_temps, dim=1, keepdim=True)
        )
        temp_weight_text = (
            torch.log(s_T / (batch_size - 1))
            + self.b_T[text_ids][None, :]
            + self.rho
            - torch.sum(weights_text * text_diffs_d_temps, dim=0, keepdim=True)
        )

        temp_image_loss = torch.mean(temp_weight_image * tau_image[:, None])
        temp_text_loss = torch.mean(temp_weight_text * tau_text[None, :])

        temp_loss = temp_image_loss + temp_text_loss

        log_obj = {}

        log_obj.update(get_temperature_statistics_1d(tau_image, prefix="train/image/"))
        log_obj.update(get_temperature_statistics_1d(tau_text, prefix="train/text/"))

        if utils.is_main_process():
            wandb.log(log_obj, step=wandb.run.step)

        return (
            (clip_loss, temp_loss),
            tau_image.cpu().detach().numpy(),
            tau_text.cpu().detach().numpy(),
        )


"""
    from dixian
"""


class onlineCLR_Loss(nn.Module):
    def __init__(self, temperature=0.01, world_size=8, gamma=0.5):
        """
        Inputs:
           N is number of samples in training set
        """
        super(onlineCLR_Loss, self).__init__()
        self.world_size = world_size
        self.pT = temperature * 10
        self.nT = temperature

        self.u_p = torch.zeros(1).cuda()
        self.u_n = torch.zeros(1).cuda()
        self.c_p = torch.zeros(1).cuda()
        self.c_n = torch.zeros(1).cuda()

        self.gamma = gamma

    def forward(self, image_features, text_features):
        if self.world_size > 1:
            hidden1 = torch.cat(GatherLayer.apply(image_features), dim=0)
            hidden2 = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = hidden1.shape[0]

        labels = torch.eye(batch_size).cuda()  # identity matrix

        logits_ab = torch.matmul(hidden1, hidden2.T)
        logits_ba = torch.matmul(hidden2, hidden1.T)

        #  online contrastive learning
        neg_mask = 1 - labels

        neg_logits1 = logits_ab * neg_mask
        pos_logits1 = logits_ab * labels
        neg_logits2 = logits_ba * neg_mask
        pos_logits2 = logits_ba * labels

        max_neg_logits = torch.maximum(
            torch.max(neg_logits1), torch.max(neg_logits2)
        ).detach()
        max_pos_logits = torch.maximum(
            torch.max(-pos_logits1), torch.max(-pos_logits2)
        ).detach()

        neg_logits1_exp = torch.exp((neg_logits1 - max_neg_logits) / self.nT) * neg_mask
        pos_logits1_exp = torch.exp((-pos_logits1 - max_pos_logits) / self.pT) * labels
        neg_logits2_exp = torch.exp((neg_logits2 - max_neg_logits) / self.nT) * neg_mask
        pos_logits2_exp = torch.exp((-pos_logits2 - max_pos_logits) / self.pT) * labels

        self.u_n = (1 - self.gamma) * self.u_n.cuda() * torch.exp(
            (self.c_n - max_neg_logits) / self.nT
        ) + self.gamma * torch.sum(neg_logits1_exp + neg_logits2_exp).detach()
        self.u_p = (1 - self.gamma) * self.u_p.cuda() * torch.exp(
            (self.c_p - max_pos_logits) / self.pT
        ) + self.gamma * torch.sum(pos_logits1_exp + pos_logits2_exp).detach()
        self.c_n = max_neg_logits.cuda()
        self.c_p = max_pos_logits.cuda()

        p_neg_weights1 = (neg_logits1_exp / self.u_n).detach()
        p_pos_weights1 = (pos_logits1_exp / self.u_p).detach()
        p_neg_weights2 = (neg_logits2_exp / self.u_n).detach()
        p_pos_weights2 = (pos_logits2_exp / self.u_p).detach()

        def softmax_cross_entropy_with_logits_v2(
            pos_logits, pos_weights, neg_logits, neg_weights
        ):
            expsum_neg_logits = torch.sum(
                neg_weights * neg_logits
            )  # loss on negative pairs
            expsum_pos_logits = torch.sum(
                pos_weights * pos_logits
            )  # loss on positive pairs
            normalized_logits = expsum_neg_logits - expsum_pos_logits
            return normalized_logits

        loss_a = softmax_cross_entropy_with_logits_v2(
            pos_logits1, p_pos_weights1, neg_logits1, p_neg_weights1
        )
        loss_b = softmax_cross_entropy_with_logits_v2(
            pos_logits2, p_pos_weights2, neg_logits2, p_neg_weights2
        )
        loss = (loss_a + loss_b).mean()

        return loss


class Scheduled_SogCLR_Crossmodal_With_Augmentation_Loss(nn.Module):
    def __init__(
        self,
        N=2900000,
        gamma=0.1,
        temperature=0.07,
        world_size=8,
        alpha=0.05,
        total_steps=None,
    ):

        super(Scheduled_SogCLR_Crossmodal_With_Augmentation_Loss, self).__init__()
        self.world_size = world_size

        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()

        self.s_I_scheduled = torch.zeros(N).cuda()
        self.s_T_scheduled = torch.zeros(N).cuda()
        self.b_I_scheduled = torch.zeros(N).cuda()
        self.b_T_scheduled = torch.zeros(N).cuda()

        self.gamma = gamma
        self.temperature = temperature
        self.eps = 1e-14
        self.alpha = alpha
        self.total_steps = total_steps

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_i2i_temperature(self, i2i_temperature):
        pass

    def set_t2t_temperature(self, t2t_temperature):
        pass

    def forward(
        self,
        image_features,
        text_features,
        augmented_image_features,
        augmented_text_features,
        image_ids,
        text_ids,
        epoch,
        current_step,
    ):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
            augmented_image_features = torch.cat(
                GatherLayer.apply(augmented_image_features), dim=0
            )
            augmented_text_features = torch.cat(
                GatherLayer.apply(augmented_text_features), dim=0
            )

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum("i d, j d -> i j", image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]
        mask_neg = (1.0 - torch.eye(batch_size)).to(sim.device)

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = (
            torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * mask_neg
        )  # -b to avoid exp operation overflow
        exp_text_diffs = (
            torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * mask_neg
        )

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size - 1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size - 1)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0 - self.gamma) * self.s_I[image_ids] * torch.exp(
                old_b_I - self.b_I[image_ids]
            ) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T[text_ids] * torch.exp(
                old_b_T - self.b_T[text_ids]
            ) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (
            batch_size - 1
        )
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (
            batch_size - 1
        )

        image_loss = image_loss.mean()
        text_loss = text_loss.mean()

        standard_sogclr_loss = image_loss + text_loss

        # Compute the similarity based loss
        per_sample_temperature_i2t = self.temperature + self.alpha * torch.sqrt(
            (sim + 1.0) / 2.0
        )

        per_sample_temperature_t2i = self.temperature + self.alpha * torch.sqrt(
            (sim.t() + 1.0) / 2.0
        )

        image_diffs_d_temps = (
            (image_diffs / per_sample_temperature_i2t).clone().detach_()
        )
        text_diffs_d_temps = (text_diffs / per_sample_temperature_t2i).clone().detach_()

        # update b
        old_b_I = self.b_I_scheduled[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I_scheduled[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T_scheduled[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T_scheduled[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = (
            torch.exp(image_diffs_d_temps - self.b_I_scheduled[image_ids][:, None])
            * mask_neg
        )  # -b to avoid exp operation overflow
        exp_text_diffs = (
            torch.exp(text_diffs_d_temps - self.b_T_scheduled[text_ids][None, :])
            * mask_neg
        )

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size - 1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size - 1)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0 - self.gamma) * self.s_I_scheduled[image_ids] * torch.exp(
                old_b_I - self.b_I_scheduled[image_ids]
            ) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T_scheduled[text_ids] * torch.exp(
                old_b_T - self.b_T_scheduled[text_ids]
            ) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I_scheduled[image_ids] = s_I.squeeze()
        self.s_T_scheduled[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        sim_image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (
            batch_size - 1
        )

        sim_text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (
            batch_size - 1
        )

        sim_image_loss = sim_image_loss.mean()
        sim_text_loss = sim_text_loss.mean()

        modulated_loss = sim_image_loss + sim_text_loss

        # Compute the image-image and text-text similarity
        labels = torch.arange(image_features.shape[0], device=image_features.device)
        sim_i2i = image_features @ augmented_image_features.T

        per_sample_temp_i2i = self.temperature + self.alpha * torch.sqrt(
            (sim_i2i + 1.0) / 2.0
        )

        modulated_i2i_loss = F.cross_entropy(sim_i2i / per_sample_temp_i2i, labels)

        sim_t2t = text_features @ augmented_text_features.T

        per_sample_temp_t2t = self.temperature + self.alpha * torch.sqrt(
            (sim_t2t + 1.0) / 2.0
        )

        modulated_t2t_loss = F.cross_entropy(sim_t2t / per_sample_temp_t2t, labels)

        modulated_unimodal_loss = modulated_i2i_loss + modulated_t2t_loss

        modulated_loss += modulated_unimodal_loss

        # Compute the weights
        normalized_current_step = current_step / self.total_steps
        sogclr_loss_weight = (normalized_current_step - 1.0) ** 2
        sim_loss_weight = normalized_current_step**2

        # Combine the two losses
        total_loss = (
            sogclr_loss_weight * standard_sogclr_loss + sim_loss_weight * modulated_loss
        )

        log_obj = {
            "train/temperature": self.temperature,
            "train/i2t_loss": image_loss.item(),
            "train/t2i_loss": text_loss.item(),
            "train/sogclr_loss": standard_sogclr_loss.item(),
            "train/sim_i2t_loss": sim_image_loss,
            "train/sim_t2i_loss": sim_text_loss,
            "train/sim_loss": modulated_loss,
            "train/sogclr_loss_weight": sogclr_loss_weight,
            "train/sim_loss_weight": sim_loss_weight,
        }

        log_obj.update(
            get_temperature_statistics(per_sample_temperature_i2t, prefix="train/i2t/")
        )

        log_obj.update(
            get_temperature_statistics(per_sample_temperature_t2i, prefix="train/t2i/")
        )

        log_obj.update(
            get_temperature_statistics(per_sample_temp_i2i, prefix="train/i2i/")
        )

        log_obj.update(
            get_temperature_statistics(per_sample_temp_t2t, prefix="train/t2t/")
        )

        if utils.is_main_process():
            wandb.log(
                log_obj,
                step=wandb.run.step,
            )

        return total_loss
