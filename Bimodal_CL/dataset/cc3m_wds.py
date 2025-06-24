import io
import os
import json
import torch
import pickle
import random
import numpy as np
import webdataset as wds
from dataset.utils import pre_caption
import PIL.Image as Image
import torch.distributed as dist
from torch.utils.data import Dataset
from collections import Counter


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


def get_train_shards(base_path):
    return _get_shard_list(start_index=0, end_index=110, base_path=base_path)


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


def load_image_embedding(vision_embeddings_base_path, key):
    # First check if the file exists
    embedding_path = os.path.join(vision_embeddings_base_path, f"{key}.npz")

    if not os.path.exists(embedding_path):
        print(f"File {embedding_path} does not exist.")
        return None

    # Load the embeddings
    return np.load(embedding_path)["embedding"].astype(np.float32)


def make_dataset_train(
    transform,
    args,
    max_words=30,
):

    def create_image_indexer(captions):
        print(f"===> Creating image indexer...")
        img2idx = {}
        idx = 0

        for image_id, _ in captions.items():
            # image_id is in the format shardindex_samplexxx_yyy. We need it in the format: samplexxx_yyy
            image_id = image_id.split("_", 1)[1]
            if image_id not in img2idx:
                img2idx[image_id] = idx
                idx += 1

        return img2idx

    def load_captions():
        with open(args.captions_path, "r") as f:
            return json.load(f)

    def make_sample_train(
        sample,
    ):
        key = sample["__key__"]
        image = sample["image.pth"]
        caption = sample["metadata.pyd"]["caption"]

        base_caption = pre_caption(caption=caption, max_words=max_words)
        base_image = transform(image)
        idx = text_idx = img2idx[key]

        return {
            "image": base_image,
            "caption": base_caption,
            "idx": idx,
            "text_idx": text_idx,
        }

    captions = load_captions()
    img2idx = create_image_indexer(captions)

    train_set = wds.WebDataset(
        urls=get_train_shards(base_path=args.cc3m_train_base_path),
        resampled=True,
        shardshuffle=True,
        cache_dir=None,
        nodesplitter=wds.split_by_worker,
    )

    train_set = (
        train_set.shuffle(1000)
        .decode(decoder_pth)
        .map(lambda sample: make_sample_train(sample))
        .select(lambda x: x is not None)
    )
    train_set = train_set.batched(args.batch_size_train)
    return train_set


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
        image_id = self.ann[index]["image"]
        image_path = os.path.join(self.root, image_id)
        image = torch.load(image_path, map_location="cpu", weights_only=True)
        image = Image.fromarray(image.numpy()).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "caption": self.text[index],
            "index": index,
        }
