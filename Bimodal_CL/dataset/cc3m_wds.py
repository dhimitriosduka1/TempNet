import io
import os
import json
import torch
import pickle
import webdataset as wds
from dataset.utils import pre_caption
import PIL.Image as Image
import torch.distributed as dist
from torch.utils.data import Dataset

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


def get_train_dataset_size():
    # Value we got after counting the number of samples in the training dataset
    return 2905954


def get_val_dataset_size():
    # Value we got after counting the number of samples in the validation dataset
    return 13358


def get_train_shards():
    return _get_shard_list(
        start_index=0, end_index=110, base_path="/BS/databases23/CC3M_tar/training/"
    )


def get_val_shards():
    # We only have on validation shard. Therefore, start_index and end_index are the same.
    return _get_shard_list(
        start_index=0, end_index=0, base_path="/BS/databases23/CC3M_tar/validation"
    )


def _get_shard_list(start_index, end_index, base_path):
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
    def make_sample_train(sample, precomputed_classes, per_class_temperature):
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
        urls=get_train_shards(),
        resampled=True,
        shardshuffle=True,
        cache_dir=cache_dir,
        nodesplitter=wds.split_by_worker,
    )

    precomputed_classes = load_precomputed_classes()
    per_class_temperature = get_per_class_temperature(
        classes_=precomputed_classes["metadata"]["classes"],
        tau_min=tau_min,
        tau_max=tau_max,
    )

    train_set = (
        train_set.shuffle(1000)
        .decode(decoder_pth)
        .map(
            lambda sample: make_sample_train(
                sample, precomputed_classes, per_class_temperature
            )
        )
    )
    train_set = train_set.batched(batch_size)
    return train_set


def make_dataset_val(transform, max_words=30, batch_size=128):
    def make_sample_val(sample):
        image = sample["image.pth"]
        caption = sample["metadata.pyd"]["caption"]
        key = sample["__key__"]

        return (
            transform(image),
            pre_caption(caption=caption, max_words=max_words),
            key,
            # Placeholder for class
            torch.tensor(-1.0),
        )

    return (
        wds.WebDataset(
            urls=get_val_shards(),
            nodesplitter=wds.split_by_worker,
        )
        .decode(decoder_pth)
        .map(make_sample_val)
        .batched(batch_size)
    )


def make_dataloader_train(trainset, batch_size=128, num_workers=4):
    trainloader = wds.WebLoader(trainset, batch_size=None, num_workers=num_workers)

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    trainloader = trainloader.unbatched().shuffle(1000).batched(batch_size)

    # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
    world_size = get_world_size()
    batches_per_epoch = get_train_dataset_size() // (batch_size * world_size)   
    trainloader.batches_per_epoch = batches_per_epoch
    trainloader = trainloader.with_epoch(batches_per_epoch)

    return trainloader


def make_dataloader_val(valset, batch_size=128):
    # In a IterableDataset, the batch creation is done in the dataset
    dataloader = wds.WebLoader(valset, batch_size=None, num_workers=4, shuffle=False)
    
    world_size = get_world_size()
    batches_per_epoch = get_val_dataset_size() // (batch_size * world_size)
    dataloader = dataloader.with_epoch(batches_per_epoch)

    # Add mappings for evaluation
    size = get_val_dataset_size()
    dataloader.dataset = {
        "txt2img": {i: [i] for i in range(size)},
        "img2txt": {i: [i] for i in range(size)},
    }

    return dataloader

# A non-wds version of the validation dataset
class CC3M_Val_Dataset(Dataset):
    def __init__(self, ann_file, transform, root, max_words=30):
        self.ann = json.load(open(ann_file, "r"))
        self.transform = transform
        self.root = root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []

            # Preprocess caption
            caption_path = os.path.join(self.root, ann["caption"])
            caption = pickle.load(open(caption_path, "rb"))["caption"]
            caption = pre_caption(caption, self.max_words)

            self.text.append(caption)

            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.ann[index]["image"])
        image = torch.load(image_path, map_location="cpu", weights_only=True)
        image = Image.fromarray(image.numpy()).convert("RGB")
        image = self.transform(image)
        return image, index
