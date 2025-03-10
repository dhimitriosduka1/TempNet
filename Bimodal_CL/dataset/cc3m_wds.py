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
    return 2905954


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
    cache_dir=None,
    batch_size=128,
    tau_min=0.01,
    tau_max=0.02,
):
    def load_precomputed_classes(
        path="/BS/dduka/work/projects/TempNet/Bimodal_CL/pickle/caption_features_without_tensors.pkl",
    ):
        with open(path, "rb") as file:
            return pickle.load(file)

    # TODO: Change `torch.tensor(-1.0)` to correct values when using different loss than CLIP
    def make_sample(sample, precomputed_classes, per_class_temperature):
        image = sample["image.pth"]
        caption = sample["metadata.pyd"]["caption"]
        key = sample["__key__"]
        class_ = precomputed_classes[key]["class_"]
        temperature = per_class_temperature[class_]

        return (
            transform(image),
            pre_caption(caption=caption, max_words=max_words),
            torch.tensor(-1.0),
            torch.tensor(-1.0),
            torch.tensor(class_),
            torch.tensor(temperature),
            key,
        )

    train_set = wds.WebDataset(
        urls=get_shard_list(),
        resampled=True,
        shardshuffle=True,
        cache_dir=cache_dir,
        nodesplitter=wds.split_by_worker,
    )

    precomputed_classes = load_precomputed_classes()
    per_class_temperature = get_per_class_temperature(
        classes_=precomputed_classes["metadata"]["classes"], tau_min=tau_min, tau_max=tau_max
    )

    print(f"Using per-class-temperatures: {per_class_temperature}")

    train_set = (
        train_set.shuffle(1000)
        .decode(decoder_pth)
        .map(
            lambda sample: make_sample(
                sample, precomputed_classes, per_class_temperature
            )
        )
    )
    train_set = train_set.batched(batch_size)
    return train_set


def make_dataloader_train(trainset, batch_size=128, num_workers=4):
    trainloader = wds.WebLoader(trainset, batch_size=None, num_workers=num_workers)

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    trainloader = trainloader.unbatched().shuffle(1000).batched(batch_size)

    # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
    world_size = get_world_size()
    batches_per_epoch = get_dataset_size() // (batch_size * world_size)
    trainloader.batches_per_epoch = batches_per_epoch
    trainloader = trainloader.with_epoch(batches_per_epoch)

    return trainloader