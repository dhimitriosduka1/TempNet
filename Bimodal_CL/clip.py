import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

import wandb
import pickle
import argparse

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "false"
os.environ["CURL_CA_BUNDLE"] = ""

import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision import transforms, datasets

from models.model_clip import CLIP
from transformers import AutoTokenizer, RobertaTokenizer

from itertools import chain

import utils
import shutil

from dataset import (
    create_train_dataset,
    create_val_dataset,
    create_sampler,
    create_train_loader,
    create_val_loader,
)

from dataset import (
    create_train_dataset,
    create_val_dataset,
    create_val_loader,
)
from scheduler import create_scheduler
from optim import create_optimizer

from tqdm import tqdm

from sklearn.cluster import KMeans

from dataset.cc3m_wds import (
    make_dataloader_train,
    get_train_dataset_size,
    get_val_dataset_size,
)
from scheduler.temperature_scheduler import get_next_temperature
from global_step import GlobalStep
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cmlib

import traceback

# =================== Loading configuration files based on the env ========================

from env_config.config_manager import ConfigManager

cm = ConfigManager()

# =================== Done loading configuration files based on the env ===================

# Imports for text expert
from sentence_transformers import SentenceTransformer

# Import for stats evaluator
from mm_stats_evaluator import MMStatsEvaluator
from running_average_tracker import RunningAverageTracker


def train(
    model,
    data_loader,
    optimizer,
    optimizer_tempnet,
    tokenizer,
    epoch,
    max_epoch,
    warmup_steps,
    device,
    scheduler,
    grad_scaler,
    args,
    eval_objects,
    txt_expert_model,
):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "loss_ita", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )

    if optimizer_tempnet is not None:
        metric_logger.add_meter(
            "lr_temp_net", utils.SmoothedValue(window_size=1, fmt="{value:.8f}")
        )

    if args.ita_type == "isogclr_tempnet":
        metric_logger.add_meter(
            "loss_temp", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
        )

    if args.ita_type == "isogclr_protonet":
        metric_logger.add_meter(
            "loss_proto", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
        )
        metric_logger.add_meter(
            "loss_swav", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
        )

    metric_logger.add_meter(
        "avg_image_tau", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "avg_text_tau", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )

    if args.ita_type == "isogclr_protonet":
        image_protos_occurrences = {}
        text_protos_occurrences = {}

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.ita_type == "isogclr_tempnet" and epoch == args.epochs - 1:
        image_tau_array = np.zeros(args.data_number)
        text_tau_array = np.zeros(args.data_number)

    if args.data == "imagenet100" or args.data == "imagenet1k":
        period = (args.epochs * len(data_loader)) / 5.0
    else:
        period = (args.epochs * data_loader.batches_per_epoch) / 5.0

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = batch["image"]
        augmented_images = batch["augmented_image"]

        texts = batch["caption"]
        augmented_texts = batch["augmented_caption"]

        idx = batch["idx"]
        text_idx = batch["text_idx"]

        expert_image_embedding = batch["expert_image_embedding"]

        classes = batch["class_"]
        superclasses = batch["superclass_"]

        if args.data == "imagenet100":
            eval_freq = len(data_loader)
        else:
            eval_freq = 500

        if i % eval_freq == 0:
            model.eval()
            (
                val_result_coco,
                test_result_coco,
                val_result_flickr,
                test_result_flickr,
                val_result_cc3m,
            ) = evaluate(
                model_without_ddp=eval_objects["model_without_ddp"],
                val_coco_loader=eval_objects["val_coco_loader"],
                test_coco_loader=eval_objects["test_coco_loader"],
                val_flickr_loader=eval_objects["val_flickr_loader"],
                test_flickr_loader=eval_objects["test_flickr_loader"],
                val_cc3m_loader=eval_objects["val_cc3m_loader"],
                # val_imagenet100_loader=eval_objects["val_imagenet100_loader"],
                val_imagenet1k_loader=eval_objects["val_imagenet1k_loader"],
                tokenizer=tokenizer,
                device=device,
                args=args,
            )

        model.train()

        if optimizer_tempnet is not None:
            optimizer_tempnet.param_groups[0]["lr"] = (
                args.lr_temp_net * 0.9**epoch
            )  # exp decay

        images = images.to(device, non_blocking=True)
        augmented_images = augmented_images.to(device, non_blocking=True)

        idx = idx.to(device, non_blocking=True)
        text_idx = text_idx.to(device, non_blocking=True)

        text_inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)

        augmented_texts = tokenizer(
            augmented_texts,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)

        if args.enable_vision_expert:
            expert_image_embedding = expert_image_embedding.to(
                device, non_blocking=True
            )

        if grad_scaler is None:
            assert 0
        else:
            with torch.cuda.amp.autocast():
                # Use cos temperature schduler if enabled
                if args.data == "imagenet100" or args.data == "imagenet1k":
                    global_it = epoch * len(data_loader) + i
                else:
                    global_it = epoch * data_loader.batches_per_epoch + i

                if args.temperature_scheduler in ["cos", "cos_aug", "fixed"]:
                    # Get next temperature
                    updated_temperature = get_next_temperature(
                        tau_min=args.tau_min,
                        tau_max=args.tau_max,
                        global_it=global_it,
                        period=period,
                        offset=args.offset,
                    )

                    # Set temperature based on the scheduler
                    if args.temperature_scheduler == "cos":
                        model.module.criterion.set_temperature(updated_temperature)
                        model.module.criterion.set_i2i_temperature(updated_temperature)
                        model.module.criterion.set_t2t_temperature(updated_temperature)

                    elif args.temperature_scheduler == "cos_aug":
                        # I'm only interested in i2i and t2t losses being affected from the cosine scheduler when the cos_aug is used.
                        model.module.criterion.set_i2i_temperature(updated_temperature)
                        model.module.criterion.set_t2t_temperature(updated_temperature)

                    elif args.temperature_scheduler == "fixed":
                        model.module.criterion.set_temperature(args.temp)
                        model.module.criterion.set_i2i_temperature(args.temp)
                        model.module.criterion.set_t2t_temperature(args.temp)

                    else:
                        raise ValueError(
                            f"Unknown temperature scheduler: {args.temperature_scheduler}"
                        )

                loss_term, info_dict = model(
                    image=images,
                    augmented_image=augmented_images,
                    text=text_inputs,
                    augmented_text=augmented_texts,
                    idx=idx,
                    text_idx=text_idx,
                    epoch=epoch,
                    max_epoch=max_epoch,
                    args=args,
                    current_step=global_it,
                    txt_expert_model=txt_expert_model,
                    raw_text=texts,
                    expert_image_embedding=expert_image_embedding,
                )

                if utils.is_main_process():
                    log_obj = {
                        "train/loss": loss_term,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }

                    if args.ita_type == "isogclr_tempnet":
                        clip_loss = loss_term[0]
                        temp_loss = loss_term[1]

                        log_obj["train/loss"] = clip_loss
                        log_obj["train/temp_loss"] = temp_loss

                    if args.enable_cluster_stats_train:
                        train_cluster_stats = train_stats_evaluator.evaluate(
                            image_features=info_dict["image_features"],
                            text_features=info_dict["text_features"],
                            classes=classes,
                            superclasses=superclasses,
                            gather=True,
                        )

                        train_cluster_stats = train_stats_evaluator.format(
                            train_cluster_stats, prefix="cc3m/train"
                        )

                        log_obj.update(train_cluster_stats)

                    wandb.log(log_obj, step=GlobalStep.get())

            if args.ita_type == "isogclr_tempnet" and epoch == args.epochs - 1:
                image_tau_array[info_dict["image_ids"]] = info_dict["image_tau"]
                text_tau_array[info_dict["text_ids"]] = info_dict["text_tau"]

            if args.ita_type == "isogclr_tempnet":
                (clip_loss, temp_loss) = loss_term

                metric_logger.update(loss_ita=clip_loss.item())
                metric_logger.update(loss_temp=temp_loss.item())

                optimizer.zero_grad()
                optimizer_tempnet.zero_grad()

                grad_scaler.scale(clip_loss + temp_loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.step(optimizer_tempnet)

                grad_scaler.update()

            else:
                metric_logger.update(loss_ita=loss_term.item())

                optimizer.zero_grad()

                grad_scaler.scale(loss_term).backward()
                grad_scaler.step(optimizer)

                grad_scaler.update()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if args.ita_type == "isogclr_tempnet":
            metric_logger.update(lr_temp_net=optimizer_tempnet.param_groups[0]["lr"])
            metric_logger.update(avg_image_tau=float(np.mean(info_dict["image_tau"])))
            metric_logger.update(avg_text_tau=float(np.mean(info_dict["text_tau"])))

        elif args.ita_type in ["isogclr"]:
            # metric_logger.update(lr_temp_net=optimizer_tempnet.param_groups[0]["lr"])
            metric_logger.update(avg_image_tau=float(np.mean(info_dict["image_tau"])))
            metric_logger.update(avg_text_tau=float(np.mean(info_dict["text_tau"])))

        else:
            metric_logger.update(avg_image_tau=args.temp)
            metric_logger.update(avg_text_tau=args.temp)

        if (
            epoch == 0
            and i % step_size == 0
            and i <= warmup_iterations
            and args.sched != "midpoint"
        ):
            scheduler.step(i // step_size)

        # Increment the step of wandb
        GlobalStep.increment()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    if args.ita_type == "isogclr_tempnet" and epoch == args.epochs - 1:

        with open(os.path.join(args.output_dir, "tau.pkl"), "wb") as f:
            pickle.dump(
                {"tau_image": image_tau_array, "tau_text": text_tau_array},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print("image tau mean:", np.mean(image_tau_array))
        print("text tau mean:", np.mean(text_tau_array))

    return (
        val_result_coco,
        test_result_coco,
        val_result_flickr,
        test_result_flickr,
        val_result_cc3m,
        {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        },
    )


"""
    zero-shot transfer
    https://github.com/goel-shashank/CyCLIP/blob/52d77af2a5f1a4bff01b4c371d6b98e2d0340137/src/evaluate.py#L42
"""


def create_zeroshot_dataloader(dataset_name, data_folder, image_size, train=False):
    assert dataset_name in ["cifar10", "cifar100", "imagenet"]

    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_folder, download=True, train=train, transform=val_transform
        )
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            root=data_folder, download=True, train=train, transform=val_transform
        )
    else:
        dataset = datasets.ImageFolder(root=data_folder, transform=val_transform)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True
    )

    data_loader.num_samples = len(dataset)

    return data_loader


@torch.no_grad()
def zeroshot_transfer(model, data_loader, dataset_name, tokenizer, device):
    model.eval()

    print(f"===> Loading zeroshot transfer config for {dataset_name}")
    config = eval(open(f"zeroshot_transfer/{dataset_name}_classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]

    text_embeddings = []
    for c in classes:
        texts = [template.format(c) for template in templates]
        text_inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)
        text_outputs = model.text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            output_hidden_states=False,
        )
        text_embeds = F.normalize(
            model.text_proj(text_outputs.last_hidden_state[:, 0, :]), dim=-1
        )
        text_embed = text_embeds.mean(dim=0)
        text_embed /= text_embed.norm()
        text_embeddings.append(text_embed)

    text_embeddings = torch.stack(text_embeddings, dim=1).to(device)

    topk = [1, 3, 5, 10]
    correct = {k: 0 for k in topk}

    for image, label in data_loader:
        image, label = image.to(device), label.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat)
        image_embedding = F.normalize(image_embed, dim=-1)

        logits = image_embedding @ text_embeddings
        ranks = logits.topk(max(topk), 1)[1].T
        predictions = ranks == label

        for k in topk:
            correct[k] += torch.sum(torch.any(predictions[:k], dim=0)).item()

    results = {f"zeroshot_top{k}": correct[k] / data_loader.num_samples for k in topk}

    return results


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, args, dataset_name):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print(f"Computing features for evaluation on {dataset_name}...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)
        text_output = model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            output_hidden_states=False,
        )
        text_embed = F.normalize(
            model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        text_embeds.append(text_embed)
    text_embeds = torch.cat(text_embeds, dim=0)

    image_embeds = []
    classes = []
    superclasses = []

    # for image, img_id in data_loader:
    for batch in data_loader:
        image = batch["image"]
        if dataset_name == "cc3m":
            classes.append(batch["class_"])
            superclasses.append(batch["superclass"])

        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)
        image_embeds.append(image_embed)

    image_embeds = torch.cat(image_embeds, dim=0)

    stats = {}
    if dataset_name == "cc3m" and args.enable_cluster_stats_val:
        classes = torch.cat(classes, dim=0)
        superclasses = torch.cat(superclasses, dim=0)

        stats = stats_evaluator.evaluate(
            image_embeds, text_embeds, classes, superclasses, gather=False
        )
        stats = stats_evaluator.format(stats, prefix="cc3m/val")

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
        score_matrix_i2t[start + i, topk_idx] = topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
        score_matrix_t2i[start + i, topk_idx] = topk_sim

    dist.barrier()
    torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), stats


@torch.no_grad()
def evaluate_modality_gap(model, data_loader, tokenizer, device, args, dataset_name):
    # test
    model.eval()

    print(f"Computing features for modality gap evaluation on {dataset_name}...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)
        text_output = model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            output_hidden_states=False,
        )
        text_embed = F.normalize(
            model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        text_embeds.append(text_embed)

    text_embeds = torch.cat(text_embeds, dim=0)
    print(f"Shape of text_embeds: {text_embeds.shape}")

    image_embeds = []

    for batch in data_loader:
        image = batch["image"]
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)
        image_embeds.append(image_embed)

    image_embeds = torch.cat(image_embeds, dim=0)
    print(f"Shape of image_embeds: {image_embeds.shape}")

    mean_text_embeds = text_embeds.mean(dim=0)
    mean_image_embeds = image_embeds.mean(dim=0)

    modality_gap = torch.norm(mean_text_embeds - mean_image_embeds, p=2)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return {
        "modality_gap": modality_gap,
    }


@torch.no_grad()
def unimodal_tsne_and_pca_plot(model, args, device, save_dir: str = "./emb_viz"):
    """
    Extract val-set embeddings for CIFAR-10 & CIFAR-100, build 2-D PCA and t-SNE
    projections, colour-code by class, and save each figure as a PNG.

    Parameters
    ----------
    model : nn.Module
    args  : namespace  (needs .zs_dataset and .image_res)
    device: torch.device
    save_dir : str     Folder where images will be written.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    for dataset_name, num_class in (("cifar10", 10), ("cifar100", 100)):

        # 1 ─── load validation set ────────────────────────────────────────────
        # print(f"Loading {dataset_name} val dataloader …")
        val_dl = create_zeroshot_dataloader(
            dataset_name=dataset_name,
            data_folder=dataset_name,
            image_size=args.image_res,
            train=False,
        )

        # 2 ─── extract features ──────────────────────────────────────────────
        feats, labels = [], []
        for img, lab in tqdm(val_dl, desc=f"Extracting {dataset_name} feats"):
            img = img.to(device)
            emb = model.vision_proj(model.visual_encoder(img))
            emb = F.normalize(emb, dim=-1)
            feats.append(emb.cpu())
            labels.append(lab)

        feats = torch.cat(feats).numpy()  # [N, D]
        labels = torch.cat(labels).numpy()  # [N]
        print(f"{dataset_name}: features {feats.shape}, labels {labels.shape}")

        # 3 ─── 2-D projections ───────────────────────────────────────────────
        # print(f"Applying PCA on {dataset_name} features")
        pca_2d = PCA(n_components=2, random_state=0).fit_transform(feats)
        # print(f"Applying t-SNE on {dataset_name} features")
        tsne_2d = TSNE(
            n_components=2,
            perplexity=30,
            metric="cosine",
            init="pca",
            random_state=0,
            n_iter=1000,
        ).fit_transform(feats)

        # UMAP also
        # umap_2d = umap.UMAP(n_components=2, random_state=0).fit_transform(feats)

        # 4 ─── plotting & saving ─────────────────────────────────────────────
        cmap = cmlib.get_cmap("tab20", num_class)
        colors = [cmap(c) for c in labels]

        # ── PCA ──────────────────────────────────────────────────────────
        fig_pca, ax_pca = plt.subplots(figsize=(6, 5))
        ax_pca.scatter(pca_2d[:, 0], pca_2d[:, 1], s=7, c=colors, alpha=0.8, lw=0)
        ax_pca.set_title(f"{dataset_name.upper()} – PCA")
        ax_pca.set_xticks([])
        ax_pca.set_yticks([])
        fig_pca.tight_layout()
        pca_path = os.path.join(save_dir, f"{dataset_name}_pca.png")
        fig_pca.savefig(pca_path, dpi=300, bbox_inches="tight")
        plt.close(fig_pca)
        # print(f"Saved → {pca_path}")

        # ── t-SNE ────────────────────────────────────────────────────────
        fig_tsne, ax_tsne = plt.subplots(figsize=(6, 5))
        ax_tsne.scatter(tsne_2d[:, 0], tsne_2d[:, 1], s=7, c=colors, alpha=0.8, lw=0)
        ax_tsne.set_title(f"{dataset_name.upper()} – t-SNE")
        ax_tsne.set_xticks([])
        ax_tsne.set_yticks([])
        fig_tsne.tight_layout()
        tsne_path = os.path.join(save_dir, f"{dataset_name}_tsne.png")
        fig_tsne.savefig(tsne_path, dpi=300, bbox_inches="tight")
        plt.close(fig_tsne)
        # print(f"Saved → {tsne_path}")

        # Send the image to wandb
        wandb.log({f"{dataset_name}_pca_tsne": wandb.Image(pca_path)})
        wandb.log({f"{dataset_name}_tsne_tsne": wandb.Image(tsne_path)})

        # ── SPHERE (3-D PCA projected to the unit sphere) ──────────────────────────
        # 1. 3-D PCA
        pca_3d = PCA(n_components=3, random_state=0).fit_transform(feats)

        # 2. re-project every point to the unit sphere
        sphere_xyz = pca_3d / np.linalg.norm(pca_3d, axis=1, keepdims=True)

        # 3. plot
        fig_sph = plt.figure(figsize=(6, 5))
        ax_sph = fig_sph.add_subplot(111, projection="3d")

        # scatter the points
        ax_sph.scatter(
            sphere_xyz[:, 0],
            sphere_xyz[:, 1],
            sphere_xyz[:, 2],
            c=colors,
            s=7,
            alpha=0.85,
            linewidth=0,
        )

        # optional: light wireframe reference sphere
        u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
        ax_sph.plot_wireframe(
            np.cos(u) * np.sin(v),
            np.sin(u) * np.sin(v),
            np.cos(v),
            color="gray",
            linewidth=0.3,
            alpha=0.25,
        )

        ax_sph.set_title(f"{dataset_name.upper()} – Sphere (3-D PCA)")
        for axis in (ax_sph.xaxis, ax_sph.yaxis, ax_sph.zaxis):
            axis.set_ticks([])

        ax_sph.set_box_aspect([1, 1, 1])
        fig_sph.tight_layout()

        sphere_path = os.path.join(save_dir, f"{dataset_name}_sphere.png")
        fig_sph.savefig(sphere_path, dpi=300, bbox_inches="tight")
        plt.close(fig_sph)

        # 4. log to wandb
        wandb.log({f"{dataset_name}_sphere": wandb.Image(sphere_path)})

        # ── SPHERE (3-D t-SNE ➜ unit sphere) ───────────────────────────────
        tsne_3d = TSNE(
            n_components=3,
            perplexity=30,
            metric="cosine",
            init="pca",
            random_state=0,
            n_iter=1000,
        ).fit_transform(
            feats
        )  # [N, 3]

        sphere_xyz = tsne_3d / np.linalg.norm(tsne_3d, axis=1, keepdims=True)

        fig_sph = plt.figure(figsize=(6, 5))
        ax_sph = fig_sph.add_subplot(111, projection="3d")

        ax_sph.scatter(
            sphere_xyz[:, 0],
            sphere_xyz[:, 1],
            sphere_xyz[:, 2],
            c=colors,
            s=7,
            alpha=0.85,
            linewidth=0,
        )

        # optional wireframe sphere
        u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
        ax_sph.plot_wireframe(
            np.cos(u) * np.sin(v),
            np.sin(u) * np.sin(v),
            np.cos(v),
            color="gray",
            linewidth=0.3,
            alpha=0.25,
        )

        ax_sph.set_title(f"{dataset_name.upper()} – Sphere (3-D t-SNE)")
        for axis in (ax_sph.xaxis, ax_sph.yaxis, ax_sph.zaxis):
            axis.set_ticks([])
        ax_sph.set_box_aspect([1, 1, 1])
        fig_sph.tight_layout()

        sphere_path = os.path.join(save_dir, f"{dataset_name}_sphere_tsne.png")
        fig_sph.savefig(sphere_path, dpi=300, bbox_inches="tight")
        plt.close(fig_sph)

        wandb.log({f"{dataset_name}_sphere_tsne": wandb.Image(sphere_path)})

    exit()


@torch.no_grad()
def compute_temperature_assignments(
    model, data_loader, tokenizer, tau_min, tau_alpha, device, dataset_name
):
    print(f"Computing temperature assignments for {dataset_name}...")
    print(f"Tau min: {tau_min}, Tau alpha: {tau_alpha}")

    model.eval()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    for i in tqdm(range(0, num_text, text_bs), desc="Extracting text features"):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)
        text_output = model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            output_hidden_states=False,
        )
        text_embed = F.normalize(
            model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        text_embeds.append(text_embed)
    text_embeds = torch.cat(text_embeds, dim=0)

    image_embeds = []
    for batch in tqdm(data_loader, desc="Extracting image features"):
        image = batch["image"]
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)
        image_embeds.append(image_embed)

    image_embeds = torch.cat(image_embeds, dim=0)

    print(f"Shape of image_embeds: {image_embeds.shape}")
    print(f"Shape of text_embeds: {text_embeds.shape}")

    i2t_similarity_matrix = image_embeds @ text_embeds.t()
    t2i_similarity_matrix = text_embeds @ image_embeds.t()

    i2t_temp_assignments = tau_min + tau_alpha * torch.sqrt(
        (i2t_similarity_matrix + 1.0) / 2.0
    )

    t2i_temp_assignments = tau_min + tau_alpha * torch.sqrt(
        (t2i_similarity_matrix + 1.0) / 2.0
    )

    i2t_positive_temp_assignments = torch.diagonal(i2t_temp_assignments).cpu().numpy()
    t2i_positive_temp_assignments = torch.diagonal(t2i_temp_assignments).cpu().numpy()

    mask = ~torch.eye(
        i2t_similarity_matrix.size(0),
        dtype=torch.bool,
        device=i2t_similarity_matrix.device,
    )
    i2t_negative_temp_assignments = i2t_temp_assignments[mask].cpu().numpy()
    t2i_negative_temp_assignments = t2i_temp_assignments[mask].cpu().numpy()

    # Save the temperature assignments to a single pickle file
    with open(f"{dataset_name}_temperature_assignments.pkl", "wb") as f:
        pickle.dump(
            {
                "i2t_positive_temp_assignments": i2t_positive_temp_assignments,
                "t2i_positive_temp_assignments": t2i_positive_temp_assignments,
                "i2t_negative_temp_assignments": i2t_negative_temp_assignments,
                "t2i_negative_temp_assignments": t2i_negative_temp_assignments,
                "min_possible_temp": tau_min,
                "max_possible_temp": tau_min + tau_alpha,
            },
            f,
        )


@torch.no_grad()
def evaluate_unimodal_knn(model, args, device):
    model.eval()

    results = {}

    for dataset_name in ["cifar10", "cifar100"]:
        if dataset_name == "cifar10":
            num_class = 10
        elif dataset_name == "cifar100":
            num_class = 100
        else:
            assert 0, f"Dataset {dataset_name} not supported"

        print(f"Loading {dataset_name} train dataloader...")
        train_dataloader = create_zeroshot_dataloader(
            dataset_name=dataset_name,
            data_folder=dataset_name,
            image_size=args.image_res,
            train=True,
        )

        print(f"Loading {dataset_name} val dataloader...")
        val_dataloader = create_zeroshot_dataloader(
            dataset_name=dataset_name,
            data_folder=dataset_name,
            image_size=args.image_res,
            train=False,
        )

        # Extract training features and labelsC
        train_features = []
        train_labels = []
        for image, label in tqdm(
            train_dataloader, desc=f"Extracting {dataset_name} train features"
        ):
            image, label = image.to(device), label.to(device)
            image_feat = model.visual_encoder(image)
            image_embed = model.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)
            train_features.append(image_embed)
            train_labels.append(label)

        train_features = torch.cat(train_features, dim=0).cpu()
        train_labels = torch.cat(train_labels, dim=0).cpu()

        print(f"Shape of {dataset_name} train features: {train_features.shape}")
        print(f"Shape of {dataset_name} train labels: {train_labels.shape}")

        # Extract validation features and labels
        val_features = []
        val_labels = []
        for image, label in tqdm(
            val_dataloader, desc=f"Extracting {dataset_name} val features"
        ):
            image, label = image.to(device), label.to(device)
            image_feat = model.visual_encoder(image)
            image_embed = model.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)
            val_features.append(image_embed)
            val_labels.append(label)

        val_features = torch.cat(val_features, dim=0).cpu()
        val_labels = torch.cat(val_labels, dim=0).cpu()
        print(f"Shape of {dataset_name} val features: {val_features.shape}")
        print(f"Shape of {dataset_name} val labels: {val_labels.shape}")

        dist_tmp = torch.cdist(val_features, train_features)
        print(f"Shape of dist_tmp: {dist_tmp.shape}")

        predicted = torch.argsort(dist_tmp, dim=1).numpy()

        # KNN@1
        class_acc = []
        for cl in range(num_class):
            class_mask = val_labels == cl
            predictions = predicted[class_mask, 0]
            result = (train_labels[predictions] == cl).sum() / class_mask.sum()
            class_acc.append(result)

        class_acc.append(np.mean(class_acc))
        print(f"{dataset_name} KNN@1: {class_acc[-1]*100:.2f}")
        knn1_acc = class_acc[-1] * 100

        # KNN@10
        class_acc = []
        for cl in range(num_class):
            class_mask = val_labels == cl
            predictions = predicted[class_mask, :10]

            predict_mat = np.zeros(((class_mask.sum(), num_class)))
            for cl2 in range(num_class):
                result2 = (train_labels[predictions] == cl2).sum(1)
                predict_mat[:, cl2] = result2

            result = (np.argmax(predict_mat, 1) == cl).sum() / class_mask.sum()
            class_acc.append(result)

        class_acc.append(np.mean(class_acc))
        print(f"{dataset_name} KNN@10: {class_acc[-1]*100:.2f}")
        knn10_acc = class_acc[-1] * 100

        results[f"{dataset_name}_knn1"] = knn1_acc
        results[f"{dataset_name}_knn10"] = knn10_acc

    return results


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt, img2supercls=[]):

    # Images->Text
    i2t_ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        i2t_ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(i2t_ranks < 1)[0]) / len(i2t_ranks)
    tr5 = 100.0 * len(np.where(i2t_ranks < 5)[0]) / len(i2t_ranks)
    tr10 = 100.0 * len(np.where(i2t_ranks < 10)[0]) / len(i2t_ranks)

    # Text->Images
    t2i_ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        t2i_ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(t2i_ranks < 1)[0]) / len(t2i_ranks)
    ir5 = 100.0 * len(np.where(t2i_ranks < 5)[0]) / len(t2i_ranks)
    ir10 = 100.0 * len(np.where(t2i_ranks < 10)[0]) / len(t2i_ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    # Compute metrics for head, middle and tail classes
    supercls_stats = {}
    if len(img2supercls) > 0:
        # Compute i2t
        img2supercls = np.array(img2supercls)

        superclass_to_label = {
            0: "head",
            1: "mid",
            2: "tail",
        }

        for i in range(len(superclass_to_label)):
            ranks_i2t = i2t_ranks[img2supercls == i]

            print(
                f"Number of samples in {superclass_to_label[i]} i2t class: {len(ranks_i2t)}"
            )

            tr1_i = 100.0 * len(np.where(ranks_i2t < 1)[0]) / len(ranks_i2t)
            tr5_i = 100.0 * len(np.where(ranks_i2t < 5)[0]) / len(ranks_i2t)
            tr10_i = 100.0 * len(np.where(ranks_i2t < 10)[0]) / len(ranks_i2t)

            label = superclass_to_label[i]
            supercls_stats[f"{label}_txt1"] = tr1_i
            supercls_stats[f"{label}_txt5"] = tr5_i
            supercls_stats[f"{label}_txt10"] = tr10_i
            supercls_stats[f"{label}_txt_mean"] = (tr1_i + tr5_i + tr10_i) / 3

            ranks_t2i = t2i_ranks[img2supercls == i]

            print(
                f"Number of samples in {superclass_to_label[i]} t2i class: {len(ranks_t2i)}"
            )

            ir1_i = 100.0 * len(np.where(ranks_t2i < 1)[0]) / len(ranks_t2i)
            ir5_i = 100.0 * len(np.where(ranks_t2i < 5)[0]) / len(ranks_t2i)
            ir10_i = 100.0 * len(np.where(ranks_t2i < 10)[0]) / len(ranks_t2i)

            label = superclass_to_label[i]
            supercls_stats[f"{label}_img1"] = ir1_i
            supercls_stats[f"{label}_img5"] = ir5_i
            supercls_stats[f"{label}_img10"] = ir10_i
            supercls_stats[f"{label}_img_mean"] = (ir1_i + ir5_i + ir10_i) / 3

    return {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
    } | supercls_stats


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


def main(args):
    utils.init_distributed_mode(args)

    if utils.is_main_process():
        # Check if a run exists for the current experiment
        run_id = None
        wandb_run_id_path = os.path.join(args.output_dir, "wandb_id.json")
        if os.path.exists(wandb_run_id_path):
            with open(wandb_run_id_path, "r") as f:
                run_id = json.load(f)["wandb_id"]

        wandb.init(
            project="Bimodal_CL_CC3M",
            name=args.run_name,
            resume="allow",
            id=run_id,
            config=args,
            entity="dduka-max-planck-society",
        )

        # Save wandb run id to a file
        with open(wandb_run_id_path, "w") as f:
            json.dump({"wandb_id": wandb.run.id}, f)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating retrieval dataset")

    if args.data in ["sbu", "cc3m", "cc12m"]:
        train_dataset = create_train_dataset("re", args)
    elif args.data == "imagenet100":
        train_dataset = create_train_dataset("imagenet100", args)
    elif args.data == "imagenet1k":
        train_dataset = create_train_dataset("imagenet1k", args)
    else:
        assert 0, args.data + " is not supported."

    if args.data == "imagenet100" or args.data == "imagenet1k":
        args.data_number = len(train_dataset)
    else:
        args.data_number = get_train_dataset_size()

    val_coco_dataset, test_coco_dataset = create_val_dataset(
        "re", args, args.val_coco_file, args.coco_image_root, args.test_coco_file
    )

    val_flickr_dataset, test_flickr_dataset = create_val_dataset(
        "re", args, args.val_flickr_file, args.flickr_image_root, args.test_flickr_file
    )

    # For loading cc3m, we need to set the load_cc3m_val flag to True
    val_cc3m_dataset = create_val_dataset(
        dataset=None,
        args=args,
        val_file=None,
        val_image_root=None,
        load_cc3m_val=True,
    )

    # val_imagenet100_dataset = create_val_dataset(
    #     dataset="imagenet100",
    #     args=args,
    #     val_file=None,
    #     val_image_root=args.imagenet100_val_root,
    # )

    # val_imagenet1k_dataset = create_val_dataset(
    #     dataset="imagenet1k",
    #     args=args,
    #     val_file=None,
    #     val_image_root=args.imagenet1k_val_root,
    # )

    print("len of train_dataset:", args.data_number)
    print("len of coco val/test:", len(val_coco_dataset), len(test_coco_dataset))
    print("len of flickr val/test:", len(val_flickr_dataset), len(test_flickr_dataset))
    print("len of cc3m val:", get_val_dataset_size())
    # print("len of imagenet100 val:", len(val_imagenet100_dataset))
    # print("len of imagenet1k val:", len(val_imagenet1k_dataset))

    if args.extract_data:
        idx_list = []
        data_dir = os.path.join(args.output_dir, "high_images")
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        for idx in tqdm(idx_list):
            image, text, _, _ = train_dataset.__getitem__(idx, enable_transform=False)
            torchvision.utils.save_image(
                image, fp=os.path.join(data_dir, str(idx) + ":" + text + ".png")
            )

        shutil.make_archive(data_dir, "zip", data_dir)

        assert 0

    if args.data == "imagenet100" or args.data == "imagenet1k":
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler(
                [train_dataset], [True], num_tasks, global_rank
            ) + [
                None,
                None,
            ]
        else:
            samplers = [None, None, None]

        train_loader = create_train_loader(
            train_dataset, samplers[0], args.batch_size_train, 8, None, drop_last=False
        )
    else:
        train_loader = make_dataloader_train(
            trainset=train_dataset, batch_size=args.batch_size_train, num_workers=8
        )

    val_coco_loader, test_coco_loader = create_val_loader(
        [val_coco_dataset, test_coco_dataset],
        [None, None],
        [args.batch_size_test] * 2,
        [8] * 2,
        [None] * 2,
    )

    val_flickr_loader, test_flickr_loader = create_val_loader(
        [val_flickr_dataset, test_flickr_dataset],
        [None, None],
        [args.batch_size_test] * 2,
        [8] * 2,
        [None] * 2,
    )

    # val_imagenet100_loader = create_val_loader(
    #     [val_imagenet100_dataset],
    #     [None],
    #     [args.batch_size_test],
    #     [8],
    #     [None],
    # )[0]

    # val_imagenet1k_loader = create_val_loader(
    #     [val_imagenet1k_dataset],
    #     [None],
    #     [args.batch_size_test],
    #     [8],
    #     [None],
    # )[0]

    val_cc3m_loader = create_val_loader(
        [val_cc3m_dataset],
        [None],
        [args.batch_size_test],
        [8],
        [None],
    )[0]

    if args.text_encoder == "roberta-large":
        tokenizer = RobertaTokenizer.from_pretrained(
            args.text_encoder, local_files_only=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.text_encoder, local_files_only=False
        )

    #### Zero-shot transfer ####
    # DD
    print(f"===> Creating zeroshot dataloader for {args.zs_dataset}")

    zs_dataset = args.zs_dataset
    zeroshot_dataloader = None

    zsh_eval = args.zsh_eval

    if zsh_eval:
        zeroshot_dataloader = create_zeroshot_dataloader(
            dataset_name=zs_dataset,
            data_folder=zs_dataset,
            image_size=args.image_res,
        )

    if args.data == "imagenet100" or args.data == "imagenet1k":
        total_steps = args.epochs * len(train_dataset)
    else:
        total_steps = args.epochs * train_loader.batches_per_epoch

    #### Model ####
    print("Creating model")
    model = CLIP(
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        embed_dim=args.embed_dim,
        init_model=args.init_model,
        world_size=args.world_size,
        ita_type=args.ita_type,
        sogclr_gamma=args.sogclr_gamma,
        rho=args.rho,
        tau_init=args.tau_init,
        temp=args.temp,
        learnable_temp=args.learnable_temp,
        personalized_tau=args.personalized_tau,
        vicreg_sim_coeff=args.vicreg_sim_coeff,
        vicreg_std_coeff=args.vicreg_std_coeff,
        N=args.data_number,
        proto_num=args.proto_num,
        proto_std=args.proto_std,
        upper_rho_plus=args.upper_rho_plus,
        proto_weight=args.proto_weight,
        sinkhorn_eps=args.sinkhorn_eps,
        swav_temp=args.swav_temp,
        swav_weight=args.swav_weight,
        total_steps=total_steps,
        sim_based_loss_alpha=args.sim_based_loss_alpha,
        sim_blend_ratio=args.sim_blend_ratio,
        clip_scheduled_loss_type=args.clip_scheduled_loss_type,
        use_per_sample_temp=args.use_per_sample_temp,
        include_unimodal_loss=args.include_unimodal_loss,
        disable_temo_modulation=args.disable_temo_modulation,
        disable_crossmodal_minfonce=args.disable_crossmodal_minfonce,
        disable_i2i_temo_loss=args.disable_i2i_temo_loss,
        disable_t2t_temo_loss=args.disable_t2t_temo_loss,
        reversed_scheduler=args.reversed_scheduler,
    )

    model = model.to(device)

    # use kmeans to find several clusters from the dataset
    if args.find_clusters:
        model.eval()

        text_feats = []
        keys = []

        print("generating features...")
        for i, batch in tqdm(enumerate(train_loader)):
            image = batch["image"]
            text = batch["caption"]
            idx = batch["idx"]
            text_idx = batch["text_idx"]

            key = batch["key"]

            image = image.to(device, non_blocking=True)
            text_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=30,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                image_feat, text_feat = model(
                    image=image,
                    augmented_image=None,
                    text=text_input,
                    augmented_text=None,
                    idx=idx,
                    text_idx=text_idx,
                    epoch=0,
                    max_epoch=0,
                    args=args,
                    return_feat=True,
                )
                text_feat = concat_all_gather(text_feat)

                gathered_keys = [[] for _ in range(torch.distributed.get_world_size())]

                torch.distributed.all_gather_object(gathered_keys, key)
                key = list(chain(*gathered_keys))

            text_feats.append(text_feat.cpu())
            keys.extend(key)

        text_feats = torch.cat(text_feats, dim=0).numpy()

        print(f"Text features mean: {np.mean(text_feats)}")

        print("Input shapes:", text_feats.shape)

        for num_clusters in [18, 200]:
            args.num_clusters = num_clusters
            print(f"Number of clusters: {args.num_clusters}")
            kmeans_txt = KMeans(n_clusters=args.num_clusters, random_state=0)

            print("KMeans clustering for txt feats...")
            kmeans_txt.fit(text_feats)
            labels = kmeans_txt.labels_

            print(f"Keys length: {len(keys)}, labels length: {len(labels)}")

            key_class_mapping = {}
            for i, key in enumerate(keys):
                key_class_mapping[key] = labels[i]

            print(f"Length of unique keys: {len(set(key_class_mapping.keys()))}")

            with open(
                f"/BS/dduka/work/projects/TempNet/Bimodal_CL/pickle/key_class_mapping_training_{args.num_clusters}_test.pkl",
                "wb",
            ) as f:
                pickle.dump(key_class_mapping, f)

            print("Saved key_class_mapping")

        return

    if len(args.checkpoint) > 0:
        checkpoint_path = args.checkpoint or saved_checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict, strict=False)
        print("Load checkpoint from %s" % args.checkpoint)
        print(f"Keys in checkpoint model: {checkpoint.keys()}")

    else:
        # pass
        if args.ita_type == "isogclr_tempnet":
            with torch.no_grad():
                batch = next(iter(train_loader))
                image = batch["image"]
                text = batch["caption"]

                image = image.to(device)
                image_embeds = model.vision_proj(model.visual_encoder(image))

                text = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=30,
                    return_tensors="pt",
                ).to(device)
                text_output = model.text_encoder(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    output_hidden_states=False,
                )
                text_embeds = model.text_proj(text_output.last_hidden_state[:, 0, :])

                model.criterion.image_temp_gen._init_prototypes(
                    text_embeds[: args.batch_size_train, :]
                )
                model.criterion.text_temp_gen._init_prototypes(
                    image_embeds[: args.batch_size_train, :]
                )

    if args.vis_prototypes:
        model.eval()

        print("Using checkpoint:", args.checkpoint)
        ckpt_idx = str(args.checkpoint).split("_")[-1].split(".")[0]

        print("get the image prototypes and text prototypes first...")
        img_protos = (
            F.normalize(model.criterion.image_temp_gen.prototypes, dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        txt_protos = (
            F.normalize(model.criterion.text_temp_gen.prototypes, dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        M = img_protos.shape[0]

        print("img_protos:", img_protos.shape)
        print("txt_protos:", txt_protos.shape)

        N = 2000
        print("get feature from " + str(N) + " image-text pairs...")
        image_feat_list = []
        text_feat_list = []
        with torch.no_grad():
            for item_idx in range(N):
                image, text, _, _ = train_dataset.__getitem__(item_idx)
                image = image.to(device).unsqueeze(0)
                image_feat = F.normalize(
                    model.vision_proj(model.visual_encoder(image)), dim=-1
                )  # .cpu().numpy()
                image_feat = (
                    model.criterion.image_temp_gen(image_feat, return_feats=True)
                    .cpu()
                    .numpy()
                )
                image_feat_list.append(image_feat)

                text = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=30,
                    return_tensors="pt",
                ).to(device)
                text_output = model.text_encoder(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    output_hidden_states=False,
                )
                text_feat = F.normalize(
                    model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
                )  # .cpu().numpy()
                text_feat = (
                    model.criterion.text_temp_gen(text_feat, return_feats=True)
                    .cpu()
                    .numpy()
                )
                text_feat_list.append(text_feat)

        image_feats = np.concatenate(image_feat_list, axis=0)
        print("image_feats:", image_feats.shape)

        text_feats = np.concatenate(text_feat_list, axis=0)
        print("text_feats:", text_feats.shape)

        print("perform visualization...")
        all_feats = np.concatenate(
            (img_protos, txt_protos, image_feats, text_feats), axis=0
        )
        # all_feats = np.concatenate((image_feats, text_feats), axis=0)

        # tsne = TSNE(n_components=2)
        # tsne_results = tsne.fit_transform(all_feats)
        # print("tsne_results:", tsne_results.shape)

        # pca = PCA(n_components=2)
        # pca_results = pca.fit_transform(all_feats)
        # print("pca_results:", pca_results.shape)

        """
            fit new umap model
        """
        # umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        # umap_results = umap_model.fit_transform(all_feats)

        # save the fitted model
        # with open('umap_model.pkl', 'wb') as f:
        #    pickle.dump(umap_model, f)

        """
            load fitted umap model
        """
        # with open('umap_model.pkl', 'rb') as f:
        #    umap_model = pickle.load(f)

        # umap_results = umap_model.transform(all_feats)

        labels = ["ip"] * M + ["tp"] * M + ["if"] * N + ["tf"] * N
        # labels = ['if'] * N + ['tf'] * N

        # with open(os.path.join(args.output_dir, "tsne_feats_"+ ckpt_idx +".pkl"), "wb") as f:
        #    pickle.dump({"results":tsne_results, "labels":labels}, f, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(os.path.join(args.output_dir, "pca_feats_"+ ckpt_idx +".pkl"), "wb") as f:
        #    pickle.dump({"results":pca_results, "labels":labels}, f, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(os.path.join(args.output_dir, "umap_feats_"+ ckpt_idx +".pkl"), "wb") as f:
        #    pickle.dump({"results":umap_results, "labels":labels}, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(
            os.path.join(args.output_dir, "umap_feats_" + ckpt_idx + ".pkl"), "wb"
        ) as f:
            pickle.dump(
                {"feats": all_feats, "labels": labels},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print("finished...")

        assert 0

    if args.check_tempnet_params:

        info_dict = {}

        print("check some learnable parameters in the tempnet...")

        image_b = model.criterion.image_temp_gen.linear_1.bias.data.cpu().item()
        text_b = model.criterion.text_temp_gen.linear_1.bias.data.cpu().item()

        print("linear_1, bias, image:", image_b)
        print("linear_1, bias, text:", text_b)

        info_dict["image_b"] = image_b
        info_dict["text_b"] = text_b

        taus_image = (
            model.criterion.image_temp_gen.taus.exp().detach().cpu().squeeze().tolist()
        )
        taus_text = (
            model.criterion.text_temp_gen.taus.exp().detach().cpu().squeeze().tolist()
        )

        print("taus image:", np.mean(taus_image), taus_image)
        print("taus text:", np.mean(taus_text), taus_text)

        info_dict["taus_image_mean"] = np.mean(taus_image)
        info_dict["taus_image"] = taus_image
        info_dict["taus_text_mean"] = np.mean(taus_text)
        info_dict["taus_text"] = taus_text

        # keep track of two representative samples
        item_idx_list = [26, 65492]
        with torch.no_grad():
            for item_idx in item_idx_list:
                image, text, _, _ = train_dataset.__getitem__(item_idx)
                image = image.to(device).unsqueeze(0)
                image_feat = F.normalize(
                    model.vision_proj(model.visual_encoder(image)), dim=-1
                )
                tau_image, weights_image = model.criterion.image_temp_gen(
                    image_feat, return_weights=True
                )
                print(
                    "learned tau and weights for ",
                    item_idx,
                    ":",
                    text,
                    tau_image.cpu().squeeze(),
                    weights_image.detach().cpu().squeeze().tolist(),
                )

                if item_idx == 26:
                    info_dict["rare_image_tau"] = tau_image.cpu().squeeze().item()
                    info_dict["rare_image_weights"] = (
                        weights_image.detach().cpu().squeeze().tolist()
                    )
                else:
                    info_dict["freq_image_tau"] = tau_image.cpu().squeeze().item()
                    info_dict["freq_image_weights"] = (
                        weights_image.detach().cpu().squeeze().tolist()
                    )

        json.dump(info_dict, open(args.checkpoint + ".json", "w"), indent=2)

        assert 0

    if args.check_samples_tau:

        image_tau_array = []
        text_tau_array = []

        model.eval()

        ckpt_idx = str(args.checkpoint).split("_")[-1].split(".")[0]

        item_idx_list = [26, 65492]
        with torch.no_grad():
            for item_idx in item_idx_list:
                image, text, _, _ = train_dataset.__getitem__(item_idx)
                image = image.to(device).unsqueeze(0)
                image_feat = F.normalize(
                    model.vision_proj(model.visual_encoder(image)), dim=-1
                )
                tau_image = (
                    model.criterion.image_temp_gen(image_feat).cpu().squeeze().numpy()
                )
                print("*" * 10, text, tau_image)

        with torch.no_grad():
            for batch in tqdm(train_loader):
                image = batch["image"]
                text = batch["caption"]
                idx = batch["idx"]
                text_idx = batch["text_idx"]

                image = image.to(device)
                text = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=30,
                    return_tensors="pt",
                ).to(device)

                image_feat = F.normalize(
                    model.vision_proj(model.visual_encoder(image)), dim=-1
                )
                text_output = model.text_encoder(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    output_hidden_states=False,
                )
                text_feat = F.normalize(
                    model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
                )

                tau_image = (
                    model.criterion.image_temp_gen(image_feat).cpu().squeeze().numpy()
                )
                tau_text = (
                    model.criterion.text_temp_gen(text_feat).cpu().squeeze().numpy()
                )

                image_tau_array.append(tau_image)
                text_tau_array.append(tau_text)

            image_tau_array = np.concatenate(image_tau_array)
            text_tau_array = np.concatenate(text_tau_array)

        with open(os.path.join(args.output_dir, "tau_" + ckpt_idx + ".pkl"), "wb") as f:
            pickle.dump(
                {"tau_image": image_tau_array, "tau_text": text_tau_array},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print("image tau mean:", np.mean(image_tau_array))
        assert 0

    optimizer, optimizer_tempnet = create_optimizer(args, model)  # clip model optimizer

    if args.ita_type in ["isogclr_tempnet"]:  # , 'isogclr_protonet']:
        assert optimizer_tempnet is not None, "we need a optimizer for isogclr_tempnet"
    else:
        assert optimizer_tempnet is None

    lr_scheduler, _ = create_scheduler(args, optimizer)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        find_unused_parameters=args.text_encoder == "roberta-large",
    )
    model_without_ddp = model.module

    text_expert_model = None
    if args.enable_txt_expert:
        print(f"Loading text expert model {args.txt_expert_model}...")
        text_expert_model = SentenceTransformer(args.txt_expert_model)
        text_expert_model = text_expert_model.to(device)

        text_expert_model = torch.nn.parallel.DistributedDataParallel(
            text_expert_model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )

        text_expert_model.eval()
        print("Text expert model loaded and set to eval mode.")

    if args.use_amp:
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        grad_scaler = None

    max_epoch = args.epochs
    warmup_steps = args.warmup_epochs
    best = 0
    best_epoch = 0

    if len(args.checkpoint) > 0:
        print(f"========== Loading states from {args.checkpoint} ==========")
        # Load optimizer state if it exists in the checkpoint
        if "optimizer" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded optimizer state")

        # Load scheduler state if it exists
        if "lr_scheduler" in checkpoint and lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print("Loaded lr_scheduler state")

        # Load old args
        if "args" in checkpoint:
            # Load the pre-loaded args
            merged_dict = {**vars(args), **vars(checkpoint["args"])}
            args = argparse.Namespace(**merged_dict)

            print(f"Loaded args from checkpoint: {args}")

        # Resume from the next epoch
        if "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1

            if args.data == "imagenet100" or args.data == "imagenet1k":
                step = args.start_epoch * len(train_loader)
            else:
                step = args.start_epoch * train_loader.batches_per_epoch

            GlobalStep.set(step)

            print(f"Epoch stored in checkpoint: {checkpoint['epoch']}")
            print(f"Training will start from epoch {args.start_epoch}")

            if utils.is_main_process():
                print(f"Starting step in run: {wandb.run.step}")

            print(f"Training will start from step: {GlobalStep.get()}")

        print(f"========== Loaded states from {args.checkpoint} ==========")

    if args.compute_temperature_assignments:
        for loader in [val_cc3m_loader]:
            compute_temperature_assignments(
                model_without_ddp,
                loader,
                tokenizer,
                args.temp,
                args.sim_based_loss_alpha,
                device,
                args.data,
            )
        exit()

    if args.unimodal_tsne_and_pca_eval:
        unimodal_tsne_and_pca_plot(model_without_ddp, args, device)
        exit()

    if args.knn_eval:
        results = evaluate_unimodal_knn(model_without_ddp, args, device)
        wandb.log(results)
        print(results)
        exit()

    if zsh_eval:
        zsh_results = zeroshot_transfer(
            model_without_ddp, zeroshot_dataloader, zs_dataset, tokenizer, device
        )
        print("finished zeroshot transfer")
        print(zsh_results)

        # Write also to a file in a directory called zsh_results
        os.makedirs("zsh_results", exist_ok=True)
        with open(f"zsh_results/{args.run_name}.json", "w") as f:
            json.dump(zsh_results, f)

        wandb.log({"zsh_results": zsh_results})
        wandb.finish()
        exit()

    print("Start training")
    start_time = time.time()

    epoch_times = np.array([])

    for epoch in range(args.start_epoch, max_epoch):
        print(f"Epoch {epoch} of {max_epoch}")

        if (
            args.data == "imagenet100" or args.data == "imagenet1k"
        ) and args.distributed:
            train_loader.sampler.set_epoch(epoch)

        start_epoch_time = time.time()

        eval_objects = {
            "model_without_ddp": model_without_ddp,
            "val_coco_loader": val_coco_loader,
            "test_coco_loader": test_coco_loader,
            "val_flickr_loader": val_flickr_loader,
            "test_flickr_loader": test_flickr_loader,
            "val_cc3m_loader": val_cc3m_loader,
            # "val_imagenet100_loader": val_imagenet100_loader,
            # "val_imagenet1k_loader": val_imagenet1k_loader,
        }

        (
            val_result_coco,
            test_result_coco,
            val_result_flickr,
            test_result_flickr,
            val_result_cc3m,
            train_stats,
        ) = train(
            model,
            train_loader,
            optimizer,
            optimizer_tempnet,
            tokenizer,
            epoch,
            max_epoch,
            warmup_steps,
            device,
            lr_scheduler,
            grad_scaler,
            args,
            eval_objects=eval_objects,
            txt_expert_model=text_expert_model,
        )

        # save tau for visualization
        if not args.evaluate and args.store_tau and (epoch + 1) % 10 == 0:
            print("saving tau...")
            tau_image = model_without_ddp.criterion.tau_I.clone().cpu().numpy()
            tau_text = model_without_ddp.criterion.tau_T.clone().cpu().numpy()

            with open(
                os.path.join(args.output_dir, "tau_" + str(epoch) + ".pkl"), "wb"
            ) as f:
                pickle.dump(
                    {"tau_image": tau_image, "tau_text": tau_text},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_result_coco.items()},
            **{f"test_{k}": v for k, v in test_result_coco.items()},
            "epoch": epoch,
            "data": "coco",
        }
        with open(os.path.join(args.output_dir, "coco_log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        if args.sched == "midpoint" or args.sched == "linear":
            # This scheduler just needs the number of epochs and nothing else
            lr_scheduler.step(epoch)
        else:
            lr_scheduler.step(epoch + warmup_steps + 1)

        save_obj = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": args,
            "epoch": epoch,
        }

        if val_result_coco["r_mean"] > best:
            torch.save(save_obj, os.path.join(args.output_dir, "checkpoint_best.pth"))
            best = val_result_coco["r_mean"]
            best_epoch = epoch

        # Save multiple checkpoints if needed
        if args.save_multiple_ckpt:
            torch.save(
                save_obj,
                os.path.join(args.output_dir, f"checkpoint_{epoch}.pth"),
            )

        torch.save(
            save_obj,
            os.path.join(args.output_dir, "checkpoint_last.pth"),
        )

        dist.barrier()
        torch.cuda.empty_cache()

        epoch_time = time.time() - start_epoch_time
        epoch_times = np.append(epoch_times, epoch_time)

        print(f"Mean epoch time: {np.mean(epoch_times)}")

        # If there isn't enough remaining time for another epoch, just end the run
        # threshold = 600
        # if get_remaining_slurm_time() - threshold <= np.mean(epoch_times):
        #     print(
        #         f"Remaining time {get_remaining_slurm_time()}s, mean epoch time {np.mean(epoch_times)}s. Breaking."
        #     )
        #     break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "coco_log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)

        wandb.finish()


def evaluate(
    model_without_ddp,
    val_coco_loader,
    test_coco_loader,
    val_flickr_loader,
    test_flickr_loader,
    val_cc3m_loader,
    val_imagenet100_loader,
    val_imagenet1k_loader,
    tokenizer,
    device,
    args,
):
    score_val_i2t_coco, score_val_t2i_coco, _ = evaluation(
        model_without_ddp, val_coco_loader, tokenizer, device, args, "coco"
    )
    score_test_i2t_coco, score_test_t2i_coco, _ = evaluation(
        model_without_ddp, test_coco_loader, tokenizer, device, args, "coco"
    )

    score_val_i2t_flickr, score_val_t2i_flickr, _ = evaluation(
        model_without_ddp, val_flickr_loader, tokenizer, device, args, "flickr"
    )
    score_test_i2t_flickr, score_test_t2i_flickr, _ = evaluation(
        model_without_ddp, test_flickr_loader, tokenizer, device, args, "flickr"
    )

    score_val_i2t_imagenet100, score_val_t2i_imagenet100, _ = evaluation(
        model_without_ddp,
        val_imagenet100_loader,
        tokenizer,
        device,
        args,
        "imagenet100",
    )

    # score_val_i2t_imagenet1k, score_val_t2i_imagenet1k, _ = evaluation(
    #     model_without_ddp,
    #     val_imagenet1k_loader,
    #     tokenizer,
    #     device,
    #     args,
    #     "imagenet1k",
    # )

    score_val_i2t_cc3m, score_val_t2i_cc3m, cc3m_stats = evaluation(
        model_without_ddp, val_cc3m_loader, tokenizer, device, args, "cc3m"
    )

    val_result_cc3m = itm_eval(
        score_val_i2t_cc3m,
        score_val_t2i_cc3m,
        val_cc3m_loader.dataset.txt2img,
        val_cc3m_loader.dataset.img2txt,
        val_cc3m_loader.dataset.img2superclass,
    )
    print("cc3m val:", val_result_cc3m)
    val_result_cc3m_wandb = {
        "cc3m/val/" + key: value for key, value in val_result_cc3m.items()
    }

    val_result_coco = itm_eval(
        score_val_i2t_coco,
        score_val_t2i_coco,
        val_coco_loader.dataset.txt2img,
        val_coco_loader.dataset.img2txt,
    )
    print("coco val:", val_result_coco)
    val_result_coco_wandb = {
        "coco/val/" + key: value for key, value in val_result_coco.items()
    }

    test_result_coco = itm_eval(
        score_test_i2t_coco,
        score_test_t2i_coco,
        test_coco_loader.dataset.txt2img,
        test_coco_loader.dataset.img2txt,
    )
    print("coco test:", test_result_coco)
    test_result_coco_wandb = {
        "coco/test/" + key: value for key, value in test_result_coco.items()
    }

    val_result_flickr = itm_eval(
        score_val_i2t_flickr,
        score_val_t2i_flickr,
        val_flickr_loader.dataset.txt2img,
        val_flickr_loader.dataset.img2txt,
    )
    print("flickr val:", val_result_flickr)
    val_result_flickr_wandb = {
        "flickr/val/" + key: value for key, value in val_result_flickr.items()
    }

    test_result_flickr = itm_eval(
        score_test_i2t_flickr,
        score_test_t2i_flickr,
        test_flickr_loader.dataset.txt2img,
        test_flickr_loader.dataset.img2txt,
    )

    print("flickr test:", test_result_flickr)
    test_result_flickr_wandb = {
        "flickr/test/" + key: value for key, value in test_result_flickr.items()
    }

    # val_result_imagenet100 = itm_eval(
    #     score_val_i2t_imagenet100,
    #     score_val_t2i_imagenet100,
    #     val_imagenet100_loader.dataset.txt2img,
    #     val_imagenet100_loader.dataset.img2txt,
    # )
    # print("imagenet100 val:", val_result_imagenet100)
    # val_result_imagenet100_wandb = {
    #     "imagenet100/val/" + key: value for key, value in val_result_imagenet100.items()
    # }

    # val_result_imagenet1k = itm_eval(
    #     score_val_i2t_imagenet1k,
    #     score_val_t2i_imagenet1k,
    #     val_imagenet1k_loader.dataset.txt2img,
    #     val_imagenet1k_loader.dataset.img2txt,
    # )
    # print("imagenet1k val:", val_result_imagenet1k)
    # val_result_imagenet1k_wandb = {
    #     "imagenet1k/val/" + key: value for key, value in val_result_imagenet1k.items()
    # }

    overall_stats = (
        val_result_coco_wandb
        | test_result_coco_wandb
        | val_result_flickr_wandb
        | test_result_flickr_wandb
        | val_result_cc3m_wandb
        | cc3m_stats
        # | val_result_imagenet100_wandb
    )

    # Modality gap
    if args.modality_gap_evaluation:
        modality_gap_flickr30k = evaluate_modality_gap(
            model_without_ddp, val_flickr_loader, tokenizer, device, args, "flickr"
        )

        modality_gap_flickr30k_wandb = {
            "flickr30k/val/modality_gap": modality_gap_flickr30k,
        }

        modality_gap_coco = evaluate_modality_gap(
            model_without_ddp, val_coco_loader, tokenizer, device, args, "coco"
        )
        modality_gap_coco_wandb = {
            "coco/val/modality_gap": modality_gap_coco,
        }

        modality_gap_cc3m = evaluate_modality_gap(
            model_without_ddp, val_cc3m_loader, tokenizer, device, args, "cc3m"
        )

        modality_gap_cc3m_wandb = {
            "cc3m/val/modality_gap": modality_gap_cc3m,
        }

        overall_stats = (
            overall_stats
            | modality_gap_flickr30k_wandb
            | modality_gap_coco_wandb
            | modality_gap_cc3m_wandb
        )

    if utils.is_main_process():
        wandb.log(data=overall_stats, step=wandb.run.step)

    if args.modality_gap_evaluation:
        print(f"Exiting...")
        exit()

    return (
        val_result_coco,
        test_result_coco,
        val_result_flickr,
        test_result_flickr,
        val_result_cc3m,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument(
        "--data",
        required=True,
        choices=["sbu", "cc3m", "cc12m", "imagenet100", "imagenet1k"],
    )
    parser.add_argument("--data_path", default=cm.get_config_for("data_path"))

    # DD
    # parser.add_argument("--train_file", default="downstream/cc3m_train_new.json")
    parser.add_argument("--train_image_root", default="cc3m")

    # model config
    parser.add_argument("--bert_config", default="configs/config_bert.json")
    parser.add_argument("--image_encoder", default="resnet50")
    parser.add_argument("--text_encoder", default="distilbert-base-uncased")
    parser.add_argument("--image_res", default=256, type=int)
    parser.add_argument("--vision_width", default=768, type=int)
    parser.add_argument("--embed_dim", default=256, type=int)

    # optimizer and schedular
    parser.add_argument("--opt", default="adamW")
    parser.add_argument(
        "--sched", default="cosine", choices=["cosine", "midpoint", "linear"]
    )
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--lr_temp_net", default=6e-6, type=float)
    parser.add_argument("--warmup", default=True, type=bool)
    parser.add_argument("--warmup_lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=0.02, type=float)
    parser.add_argument("--decay_rate", default=1, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--warmup_epochs", default=20, type=int)
    parser.add_argument("--cooldown_epochs", default=0, type=int)

    # For midpoint lr schedule
    parser.add_argument("--lr_start", default=3e-4, type=float)
    parser.add_argument("--lr_end", default=2e-4, type=float)

    # training & test settings
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--init_model", action="store_true")
    parser.add_argument("--batch_size_train", default=512, type=int)
    parser.add_argument("--batch_size_test", default=512, type=int)
    parser.add_argument("--k_test", default=256, type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--evaluate_cc3m", action="store_true")

    # output path
    parser.add_argument("--output_dir", default="./output/clip_test")
    parser.add_argument("--save_multiple_ckpt", action="store_true")

    # loss config
    parser.add_argument(
        "--ita_type",
        required=True,
        choices=[
            "clip",
            "cyclip",
            "vicreg",
            "sogclr",
            "isogclr",
            "isogclr_tempnet",
            "onlineclr",
            "clipPCT",
            "sim_based_clip",
            "scheduled_clip_loss",
            "clip_moe_text",
            "clip_moe_vision",
            "clip_moe_blend",
            "clip_meo_text_supervision",
            "scheduled_crossmodal_clip_loss",
            "clip_moe_vision_and_text",
            "scheduled_clip_moe_text",
            "scheduled_crossmodal_with_augmentations_and_unimodal_clip_loss",
            "scheduled_crossmodal_with_augmentations_clip_loss",
            "scheduled_sogclr_crossmodal",
            "scheduled_sogclr_crossmodal_with_augmentations",
            "scheduled_crossmodal_cosine_clip_with_augmentations_and_unimodal_loss",
            "sogclr_with_cosine_and_unimodal_loss",
        ],
    )
    parser.add_argument("--vicreg_sim_coeff", default=25.0, type=float)
    parser.add_argument("--vicreg_std_coeff", default=25.0, type=float)
    parser.add_argument("--sogclr_gamma", default=0.8, type=float)
    parser.add_argument("--rho", default=8.0, type=float)
    parser.add_argument("--tau_init", default=0.01, type=float)
    parser.add_argument("--temp", default=0.01, type=float)
    parser.add_argument("--learnable_temp", action="store_true")
    parser.add_argument("--personalized_tau", action="store_true")
    parser.add_argument("--store_tau", action="store_true")
    parser.add_argument("--grad_accum", default=10, type=int)

    # save learned taus for isogclr_tempnet objective
    parser.add_argument("--save_learned_taus", action="store_true")

    # set the noise level for imagenet100 dataset
    parser.add_argument("--noise_level", default=0.0, type=float)

    # set the fraction of data used for training
    parser.add_argument("--train_frac", default=1.0, type=float)

    # check samples with high/low temperature values
    parser.add_argument("--check_samples_tau", action="store_true")

    # extract data from the cc3m dataset
    parser.add_argument("--extract_data", action="store_true")

    # check some learnable parameters in TempNet
    parser.add_argument("--check_tempnet_params", action="store_true")

    # visualize the prototypes
    parser.add_argument("--vis_prototypes", action="store_true")

    # zero-shot transfer
    parser.add_argument("--zs_dataset", choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument(
        "--zs_datafolder", default="./datasets", type=str
    )  # I don't use this.

    # arguments for bilevel tempnet
    parser.add_argument("--proto_std", default=10.0, type=float)
    parser.add_argument("--proto_num", default=256, type=int)
    parser.add_argument("--upper_rho_plus", default=0.0, type=float)
    parser.add_argument("--proto_weight", default=1.0, type=float)
    parser.add_argument("--sinkhorn_eps", default=0.05, type=float)
    parser.add_argument("--swav_temp", default=0.1, type=float)
    parser.add_argument("--swav_weight", default=1.0, type=float)

    # find clusters in a dataset
    parser.add_argument("--find_clusters", action="store_true")
    parser.add_argument("--num_clusters", default=200, type=int)

    # Wandb
    parser.add_argument("--run_name", required=True)

    # Temperature
    # Explanation:
    #  cosPCT: Cosine scheduler with per class temperature
    #  cos_aug: Temperature schedulter is only applied to self losses: i2i and/or t2t
    parser.add_argument(
        "--temperature_scheduler",
        default="none",
        choices=["none", "cos", "cosPCT", "cos_aug", "fixed"],
    )
    parser.add_argument("--tau_min", default=0.01, type=float)
    parser.add_argument("--tau_max", default=0.02, type=float)
    parser.add_argument("--pct_tau_min", default=0.01, type=float)
    parser.add_argument("--pct_tau_max", default=0.02, type=float)
    parser.add_argument("--offset", default=0.0, type=float)

    parser.add_argument(
        "--per_sample_temp_similarity",
        default="t2i",
        choices=["i2t", "t2i", "t2t", "i2i"],
    )
    parser.add_argument(
        "--per_sample_temp_mapping",
        default="adaptive_with_base",
        choices=["adaptive_with_base", "adaptive_without_base", "cosine"],
    )

    parser.add_argument(
        "--use_per_sample_temp",
        action="store_true",
    )

    # cc3m
    parser.add_argument(
        "--cc3m_ann_file",
        default=cm.get_config_for("cc3m_ann_file"),
    )
    parser.add_argument(
        "--cc3m_img2cls_file_train",
        default=cm.get_config_for("cc3m_img2cls_file_train"),
    )
    parser.add_argument(
        "--cc3m_img2cls_file_val",
        default=cm.get_config_for("cc3m_img2cls_file_val"),
    )

    parser.add_argument("--cc3m_val_root", default=cm.get_config_for("cc3m_val_root"))
    parser.add_argument("--captions_path", default=cm.get_config_for("captions_path"))
    parser.add_argument("--cc3m_extended_captions_path", default="")

    # Losses
    parser.add_argument("--enable_i2i_loss", action="store_true")
    parser.add_argument("--enable_t2t_loss", action="store_true")
    parser.add_argument("--i2i_loss_weight", default=1.0, type=float)
    parser.add_argument("--t2t_loss_weight", default=1.0, type=float)
    parser.add_argument("--exclude_modulated_info_nce_loss", action="store_true")

    # SimBasedCLIP loss params
    parser.add_argument("--sim_based_loss_alpha", default=0.1, type=float)

    # For Scheduled_CLIP_Loss
    parser.add_argument(
        "--clip_scheduled_loss_type",
        default="none",
        choices=["none", "linear", "quadratic", "fixed"],
    )

    # Experts
    parser.add_argument("--enable_txt_expert", action="store_true")
    parser.add_argument(
        "--txt_expert_model", default="sentence-transformers/all-roberta-large-v1"
    )

    parser.add_argument("--sim_blend_ratio", default=0.0, type=float)

    parser.add_argument("--enable_vision_expert", action="store_true")
    parser.add_argument(
        "--vision_embeddings_base_path",
        default=cm.get_config_for("vision_embeddings_base_path"),
    )

    parser.add_argument("--include_alignment_loss", action="store_true")
    parser.add_argument("--include_unimodal_loss", action="store_true")

    parser.add_argument("--clip_loss_weight", type=float)
    parser.add_argument("--sim_loss_weight", type=float)

    parser.add_argument("--zsh_eval", action="store_true")

    parser.add_argument("--number_of_classes", default=18, type=int)
    parser.add_argument("--number_of_superclasses", default=3, type=int)

    # For cluster stats
    parser.add_argument("--enable_cluster_stats_train", action="store_true")
    parser.add_argument("--enable_cluster_stats_val", action="store_true")

    # ImageNet100
    parser.add_argument(
        "--imagenet100_val_root", default="/ptmp/dduka/work/data/imagenet100/"
    )

    # ImageNet1k
    parser.add_argument(
        "--imagenet1k_val_root", default="/ptmp/dduka/work/data/imagenet/"
    )

    # Enabling or not temperature modulation for the loss
    parser.add_argument("--disable_temo_modulation", action="store_true")
    parser.add_argument("--disable_crossmodal_minfonce", action="store_true")
    parser.add_argument("--disable_i2i_temo_loss", action="store_true")
    parser.add_argument("--disable_t2t_temo_loss", action="store_true")

    # Reversed scheduler
    parser.add_argument("--reversed_scheduler", action="store_true")

    # Modality gap evaluation
    parser.add_argument("--modality_gap_evaluation", action="store_true")

    # KNN evaluation
    parser.add_argument("--knn_eval", action="store_true")

    # Unimodal t-SNE and PCA evaluation
    parser.add_argument("--unimodal_tsne_and_pca_eval", action="store_true")

    # Compute temperature assignments
    parser.add_argument("--compute_temperature_assignments", action="store_true")

    args = parser.parse_args()

    # Validation
    validation_errors = []

    if args.temperature_scheduler == "cos_aug":
        if not args.enable_i2i_loss and not args.enable_t2t_loss:
            validation_errors.append(
                "If --temperature_scheduler is 'cos_aug', either --enable_i2i_loss and/or --enable_t2t_loss must be set."
            )

    if args.enable_t2t_loss and args.data == "cc3m":
        if not args.cc3m_extended_captions_path:
            validation_errors.append(
                "--cc3m_extended_captions_path must be provided when --enable_t2t_loss is set."
            )

    if (
        args.ita_type
        == "scheduled_crossmodal_with_augmentations_and_unimodal_clip_loss"
    ):
        if not args.enable_i2i_loss and not args.enable_t2t_loss:
            validation_errors.append(
                "If --ita_type is 'scheduled_crossmodal_with_augmentations_and_unimodal_clip_loss', --enable_i2i_loss and --enable_t2t_loss must be set."
            )

    # /ptmp/dduka/work/data/cc3m/validation/cc3m_validation_key_class_mapping_18.pk !!!! The format of the file must be like this
    if args.number_of_classes != int(
        args.cc3m_img2cls_file_val.split("/")[-1].split(".")[0].split("_")[-1]
    ):
        validation_errors.append(
            "The number of classes in the cc3m_img2cls_file_val must match the number of classes in the pickle file."
        )

    if args.number_of_classes != int(
        args.cc3m_img2cls_file_train.split("/")[-1].split(".")[0].split("_")[-1]
    ):
        validation_errors.append(
            "The number of classes in the cc3m_img2cls_file_train must match the number of classes in the pickle file."
        )

    # TODO: Maybe add some more validations here
    if validation_errors:
        parser.error("\n".join(validation_errors))

    if args.check_samples_tau:
        args.evaluate = True

    # DD
    # args.train_file = os.path.join(args.data_path, args.train_file)
    # args.train_image_root = os.path.join(args.data_path, args.train_image_root)

    args.val_coco_file = os.path.join(args.data_path, "clip_train/coco_val_new.json")
    args.test_coco_file = os.path.join(args.data_path, "clip_train/coco_test_new.json")
    args.coco_image_root = os.path.join(args.data_path, "coco")
    args.val_flickr_file = os.path.join(args.data_path, "clip_train/flickr30k_val.json")
    args.test_flickr_file = os.path.join(
        args.data_path, "clip_train/flickr30k_test.json"
    )
    args.flickr_image_root = os.path.join(args.data_path, "flickr30k")

    args.cc3m_train_base_path = os.path.join(args.data_path, "cc3m/training/")
    args.cc3m_val_base_path = os.path.join(args.data_path, "cc3m/validation/")

    # DD
    # args.sbu_file = os.path.join(args.data_path, "clip_train/sbu_train_new.json")
    # args.sbu_image_root = os.path.join(args.data_path, "sbu")
    # Add timestamp to output_dir so that every run is unique

    # Add the slurm job id
    args.job_id = os.environ.get("SLURM_JOB_ID")
    saved_checkpoint_path = os.path.join(args.output_dir, "checkpoint_last.pth")

    if os.path.exists(saved_checkpoint_path):
        args.checkpoint = saved_checkpoint_path

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    json.dump(
        args.__dict__, open(os.path.join(args.output_dir, "args.json"), "w"), indent=2
    )

    average_tracker_template = {
        "gap": RunningAverageTracker(name="gap", alpha=0.9),
        "average_pairwise_distance_image": RunningAverageTracker(
            name="average_pairwise_distance_image", alpha=0.9
        ),
        "average_pairwise_distance_text": RunningAverageTracker(
            name="average_pairwise_distance_text", alpha=0.9
        ),
        "temp_pos": RunningAverageTracker(name="temp_pos", alpha=0.9),
        "temp_neg": RunningAverageTracker(name="temp_neg", alpha=0.9),
        "temp_avg": RunningAverageTracker(name="temp_avg", alpha=0.9),
        "temp_min": RunningAverageTracker(name="temp_min", alpha=0.9),
        "temp_max": RunningAverageTracker(name="temp_max", alpha=0.9),
    }

    # Initialize the stats evaluator
    running_average_trackers = {
        "modality": average_tracker_template,
    }

    for i in range(args.number_of_classes):
        running_average_trackers[f"class_{i}"] = average_tracker_template

    for i in range(args.number_of_superclasses):
        running_average_trackers[f"superclass_{i}"] = average_tracker_template

    stats_evaluator = MMStatsEvaluator(
        world_size=args.world_size,
        running_average_trackers=running_average_trackers,
        temperature=args.temp,
        alpha=args.sim_based_loss_alpha,
    )

    train_stats_evaluator = MMStatsEvaluator(
        world_size=args.world_size,
        running_average_trackers=running_average_trackers,
        temperature=args.temp,
        alpha=args.sim_based_loss_alpha,
    )

    try:
        main(args)
    except Exception as e:
        wandb.alert(title="Training Failed", text=str(e))  # Send an alert if enabled
        print(traceback.format_exc())
        print(f"Error occurred: {e}")
    finally:
        wandb.finish()  # Ensure W&B run is properly closed
