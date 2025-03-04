import io
import os
import torch
import webdataset as wds
from dataset.utils import pre_caption
import PIL.Image as Image
import torch.distributed as dist

import pickle
from scheduler.temperature_scheduler import get_per_class_temperature


# Copied from ../utils.py
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# Copied from ../utils.py
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_dataset_size():
    # https://github.com/AILab-CVC/SEED/blob/93b3cf408196735ec4820ad2eb4d9dc4a670003d/MultiModalLLM/src/data/data.py#L73C1-L74C1
    # After counting the files inside each tar folder, we have 2799842 images.
    return 2799842


def get_shard_list(
    start_index=0, end_index=110, base_path="/BS/databases23/CC3M_tar/training/"
):
    shard_list = []
    for i in range(start_index, end_index + 1):
        tar_file_path = os.path.join(base_path, f"{i}.tar")
        if os.path.exists(tar_file_path):
            shard_list.append(tar_file_path)
    return shard_list


def decoder_pth(key, value):
    if "image.pth" in key:
        image = torch.load(io.BytesIO(value), weights_only=True)
        image = Image.fromarray(image.numpy()).convert("RGB")
        return image


def make_dataset_train(
    transform,
    max_words=30,
    batch_size=128,
    tau_min=0.01,
    tau_max=0.02,
):
    
    # Remove
    seen_keys = {}

    def load_precomputed_classes(
        path="/BS/dduka/work/projects/TempNet/Bimodal_CL/key_class_mapping.pkl",
    ):
        with open(path, "rb") as file:
            return pickle.load(file)

    # TODO: Change `torch.tensor(-1.0)` to correct values when using different loss than CLIP
    def make_sample(sample, precomputed_classes, per_class_temperature):
        image = sample["image.pth"]
        caption = sample["metadata.pyd"]["caption"]
        key = sample["__key__"]

        if key in seen_keys:
            print(f"Duplicate retrieved in make_sample: {key}")
        
        seen_keys[key] = 1

        # class_ = precomputed_classes[key]
        # temperature = per_class_temperature[class_]

        class_ = -1.0
        temperature = 0.0

        return (
            transform(image),
            pre_caption(caption=caption, max_words=max_words),
            torch.tensor(-1.0),
            torch.tensor(-1.0),
            class_,
            temperature,
            key,
        )

    precomputed_classes = load_precomputed_classes()
    per_class_temperature = get_per_class_temperature(
        classes_=precomputed_classes, tau_min=tau_min, tau_max=tau_max
    )

    train_set = (
        wds.WebDataset(
            urls=get_shard_list(),
            shardshuffle=1000,
            resampled=True,
            nodesplitter=None,
        )
        .shuffle(1000)
        .decode(decoder_pth)
        .map(
            lambda sample: make_sample(
                sample, precomputed_classes, per_class_temperature
            )
        )
        .batched(batch_size, partial=False)
    )

    return train_set


def make_dataloader_train(trainset, batch_size=128, num_workers=4, resample=True):
    loader = wds.WebLoader(
        trainset, batch_size=None, shuffle=False, num_workers=num_workers
    )

    nbatches = max(1, get_dataset_size() // (batch_size * get_world_size()))
    return loader.with_epoch(nbatches)
