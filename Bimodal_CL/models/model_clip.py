from functools import partial

import timm
from transformers import AutoModel, RobertaModel

from models.losses import (
    CLIP_Loss,
    CyCLIP_Loss,
    SogCLR_Loss,
    iSogCLR_Loss,
    iSogCLR_TempNet_Loss,
    VICReg_Loss,
    onlineCLR_Loss,
    CLIP_Loss_PCT,
    Sim_Based_CLIP_Loss,
    Scheduled_CLIP_Loss,
    CLIP_MoE_Loss,
)

import torch
from torch import nn
import torch.nn.functional as F


class CLIP(nn.Module):
    def __init__(
        self,
        image_encoder=None,
        text_encoder=None,
        embed_dim=256,
        init_model=True,
        world_size=8,
        ita_type="clip",
        sogclr_gamma=0.9,
        rho=8.0,
        tau_init=0.01,
        temp=0.01,
        learnable_temp=False,
        personalized_tau=False,
        vicreg_sim_coeff=25.0,
        vicreg_std_coeff=25.0,
        N=10000,
        proto_std=10.0,
        proto_num=256,
        upper_rho_plus=0.0,
        proto_weight=1.0,
        sinkhorn_eps=0.05,
        swav_temp=0.1,
        swav_weight=1.0,
        sim_based_loss_alpha=0.1,
        total_steps=None,
    ):
        super().__init__()

        self.temp = temp
        self.learnable_temp = learnable_temp
        self.personalized_tau = personalized_tau

        if self.learnable_temp:
            if not personalized_tau:
                self.temp = nn.Parameter(torch.ones([]) * self.temp)
            else:
                self.image_temp = nn.Parameter(torch.ones(2900000) * self.temp)
                self.text_temp = nn.Parameter(torch.ones(2900000) * self.temp)

        self.visual_encoder = timm.create_model(image_encoder, pretrained=init_model)
        self.visual_encoder.reset_classifier(0)

        if text_encoder == "roberta-large":
            self.text_encoder = RobertaModel.from_pretrained(
                text_encoder, local_files_only=False
            )
            self.text_proj = nn.Linear(1024, embed_dim)
        else:
            self.text_encoder = AutoModel.from_pretrained(
                text_encoder, local_files_only=False
            )
            self.text_proj = nn.Linear(768, embed_dim)

        if not init_model:
            self.text_encoder.init_weights()

        self.vision_proj = nn.Linear(self.visual_encoder.num_features, embed_dim)

        self.ita_type = ita_type

        if self.ita_type == "clip":
            if not personalized_tau:
                self.criterion = CLIP_Loss(
                    world_size=world_size,
                    personalized_tau=personalized_tau,
                    temperature=self.temp,
                )
            else:
                self.criterion = CLIP_Loss(
                    world_size=world_size,
                    personalized_tau=personalized_tau,
                    image_tau=self.image_temp,
                    text_tau=self.text_temp,
                )

        elif self.ita_type == "cyclip":
            self.criterion = CyCLIP_Loss(world_size=world_size, temperature=self.temp)

        elif self.ita_type == "vicreg":
            self.criterion = VICReg_Loss(
                world_size=world_size,
                dim_size=embed_dim,
                sim_coeff=vicreg_sim_coeff,
                std_coeff=vicreg_std_coeff,
            )

        elif self.ita_type == "sogclr":
            self.criterion = SogCLR_Loss(
                N=N, world_size=world_size, gamma=sogclr_gamma, temperature=self.temp
            )

        elif self.ita_type == "isogclr":
            self.criterion = iSogCLR_Loss(
                N=N, world_size=world_size, gamma=sogclr_gamma, rho=rho
            )

        elif self.ita_type == "onlineclr":
            self.criterion = onlineCLR_Loss(
                world_size=world_size, temperature=self.temp, gamma=sogclr_gamma
            )

        elif (
            self.ita_type == "isogclr_tempnet"
        ):  # only use tempnet with new derivation, more deep structures
            self.criterion = iSogCLR_TempNet_Loss(
                N=N,
                world_size=world_size,
                gamma=sogclr_gamma,
                rho=rho,
                feature_dim=embed_dim,
            )

        # Cosine with per class temperature
        elif self.ita_type == "clipPCT":
            print("Using clipPCT as loss")
            self.criterion = CLIP_Loss_PCT(world_size=world_size, temperature=self.temp)

        elif self.ita_type == "sim_based_clip":
            print(f"Using Sim_Based_CLIP_Loss")
            self.criterion = Sim_Based_CLIP_Loss(
                world_size=world_size,
                temperature=self.temp,
                alpha=sim_based_loss_alpha,
            )

        elif self.ita_type == "scheduled_clip_loss":
            print(f"Using Scheduled_CLIP_Loss")
            self.criterion = Scheduled_CLIP_Loss(
                world_size=world_size,
                temperature=self.temp,
                alpha=sim_based_loss_alpha,
                total_steps=total_steps,
            )
        elif self.ita_type == "clip_moe":
            print(f"Using CLIP_MoE")
            self.criterion = CLIP_MoE_Loss(
                world_size=world_size,
                temperature=self.temp,
                alpha=sim_based_loss_alpha,
            )
        else:
            raise NotImplementedError

    def forward(
        self,
        image,
        augmented_image,
        text,
        augmented_text,
        idx,
        text_idx,
        epoch,
        max_epoch,
        args,
        return_feat=False,
        per_sample_temperature=None,
        current_step=None,
        txt_expert_model=None,
        raw_text=None,
    ):
        if self.learnable_temp:
            with torch.no_grad():
                if not self.personalized_tau:
                    self.temp.clamp_(0.001, 0.5)
                else:
                    self.image_temp.clamp_(0.001, 0.5)
                    self.text_temp.clamp_(0.001, 0.5)

        # with torch.no_grad():
        image_embeds = self.visual_encoder(image)
        image_embeds = self.vision_proj(image_embeds)
        image_feat = F.normalize(image_embeds, dim=-1)

        text_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            output_hidden_states=False,
        )
        text_embeds = self.text_proj(text_output.last_hidden_state[:, 0, :])
        text_feat = F.normalize(text_embeds, dim=-1)

        augmented_image_feat = None
        augmented_text_feat = None

        if args.enable_i2i_loss:
            with torch.no_grad():
                augmented_image_embeds = self.visual_encoder(augmented_image)
                augmented_image_embeds = self.vision_proj(augmented_image_embeds)
                augmented_image_feat = F.normalize(augmented_image_embeds, dim=-1)

        if args.enable_t2t_loss:
            with torch.no_grad():
                augmented_text_output = self.text_encoder(
                    augmented_text.input_ids,
                    attention_mask=augmented_text.attention_mask,
                    output_hidden_states=False,
                )
                augmented_text_embeds = self.text_proj(
                    augmented_text_output.last_hidden_state[:, 0, :]
                )
                augmented_text_feat = F.normalize(augmented_text_embeds, dim=-1)

        if txt_expert_model is not None:
            assert self.ita_type == "clip_moe", "txt_expert_model should only be used with clip_moe"
            with torch.no_grad():
                txt_embeds_expert = txt_expert_model.module.encode(
                    raw_text, convert_to_tensor=True, normalize_embeddings=True
                )

        if return_feat:
            return image_feat, text_feat

        info_dict = {}

        if self.ita_type in ["clip", "cyclip"]:
            if self.personalized_tau:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
                loss_ita = self.criterion(
                    image_feat,
                    text_feat,
                    augmented_image_feat,
                    augmented_text_feat,
                    image_ids,
                    text_ids,
                    args,
                )

            else:
                loss_ita = self.criterion(
                    image_features=image_feat,
                    text_features=text_feat,
                    augmented_image_features=augmented_image_feat,
                    augmented_text_features=augmented_text_feat,
                    args=args,
                )

        elif self.ita_type == "vicreg":
            loss_ita = self.criterion(image_embeds, text_embeds)

        elif self.ita_type == "sogclr":
            image_ids = concat_all_gather(idx)
            text_ids = concat_all_gather(text_idx)
            loss_ita = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch)

        elif self.ita_type == "isogclr":
            image_ids = concat_all_gather(idx)
            text_ids = concat_all_gather(text_idx)
            loss_ita, image_tau, text_tau = self.criterion(
                image_feat, text_feat, image_ids, text_ids, epoch, max_epoch
            )
            info_dict = {
                "image_tau": image_tau,
                "text_tau": text_tau,
                "image_ids": image_ids.cpu().numpy(),
                "text_ids": text_ids.cpu().numpy(),
            }

        elif self.ita_type == "isogclr_tempnet":
            image_ids = concat_all_gather(idx)
            text_ids = concat_all_gather(text_idx)
            loss_ita, image_tau, text_tau = self.criterion(
                image_feat, text_feat, image_ids, text_ids, epoch, max_epoch
            )
            info_dict = {
                "image_tau": image_tau,
                "text_tau": text_tau,
                "image_ids": image_ids.cpu().numpy(),
                "text_ids": text_ids.cpu().numpy(),
            }

        elif self.ita_type == "onlineclr":
            loss_ita = self.criterion(image_feat, text_feat)

        elif self.ita_type == "clipPCT":
            per_sample_temperature = concat_all_gather(per_sample_temperature)
            loss_ita = self.criterion(
                image_features=image_feat,
                text_features=text_feat,
                per_sample_temperature=per_sample_temperature,
            )

        elif self.ita_type == "sim_based_clip":
            loss_ita = self.criterion(
                image_features=image_feat,
                text_features=text_feat,
                augmented_image_features=augmented_image_feat,
                augmented_text_features=augmented_text_feat,
                args=args,
            )

        elif self.ita_type == "scheduled_clip_loss":
            loss_ita = self.criterion(
                image_features=image_feat,
                text_features=text_feat,
                current_step=current_step,
            )
        elif self.ita_type == "clip_moe":
            loss_ita = self.criterion(
                image_features=image_feat,
                text_features=text_feat,
                text_expert_features=txt_embeds_expert,
                image_expert_features=None,
                args=args,
            )
        else:
            raise NotImplementedError

        return loss_ita, info_dict


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
