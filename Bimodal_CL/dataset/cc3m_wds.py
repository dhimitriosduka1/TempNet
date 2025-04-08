import io
import os
import json
import torch
import pickle
import random
import webdataset as wds
from dataset.utils import pre_caption
import PIL.Image as Image
import torch.distributed as dist
from torch.utils.data import Dataset
from collections import Counter

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
    args,
    max_words=30,
):
    
    def create_image_indexer(captions):
        img2idx = {}
        idx = 0

        for image_id, _ in captions.items():
            if image_id not in img2idx:
                img2idx[image_id] = idx
                idx += 1
        
        return img2idx
    
    def load_captions():
        with open(args.captions_path, "r") as f:
            return json.load(f)
            

    def load_extended_captions():
        with open(args.cc3m_extended_captions_path, "r") as f:
            data = json.load(f)

            # Key is in the format shardindex_samplexxx_yyy => samplexxx_yyy
            # We need to remove the shardindex_ part
            return {k.split("_", 1)[1]: v for k, v in data.items()}

    def make_sample_train(
        sample, extended_captions=None, enable_i2i_loss=False, enable_t2t_loss=False
    ):
        key = sample["__key__"]
        image = sample["image.pth"]
        caption = sample["metadata.pyd"]["caption"]

        if extended_captions is not None:
            caption = random.sample(
                [caption, *extended_captions[key]["paraphrases"]], 1
            )[0]

        base_caption = pre_caption(caption=caption, max_words=max_words)
        augmented_caption = ""
        if enable_t2t_loss:
            assert extended_captions is not None, "Extended captions are not available"
            random_caption = random.sample(extended_captions[key]["paraphrases"], 1)[0]
            augmented_caption = pre_caption(caption=random_caption, max_words=max_words)

        base_image = transform(image)
        augmented_image = torch.empty(0)
        if enable_i2i_loss:
            augmented_image = transform(image)

        idx = text_idx = img2idx[key]

        return {
            "image": base_image,
            "augmented_image": augmented_image,
            "caption": base_caption,
            "augmented_caption": augmented_caption,
            "key": key,
            "idx": idx,
            "text_idx": text_idx,
        }
    
    captions = load_captions()
    img2idx = create_image_indexer(captions)

    train_set = wds.WebDataset(
        urls=get_train_shards(),
        resampled=True,
        shardshuffle=True,
        cache_dir=None,
        nodesplitter=wds.split_by_worker,
    )

    extended_captions = None
    if args.cc3m_extended_captions_path != "":
        extended_captions = load_extended_captions()

    train_set = (
        train_set.shuffle(1000)
        .decode(decoder_pth)
        .map(lambda sample: make_sample_train(sample, extended_captions, args.enable_i2i_loss, args.enable_t2t_loss))
    )
    train_set = train_set.batched(args.batch_size_train)
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
    def __init__(self, ann_file, img2cls_file, transform, root, max_words=30):
        self.ann = json.load(open(ann_file, "r"))
        self.transform = transform
        self.root = root
        self.max_words = max_words
        self.img2cls_file = img2cls_file

        self.img2cls = pickle.load(open(self.img2cls_file, "rb"))
        self.cls2supercls = self._setup_superclasses(self.img2cls)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.img2superclass = []

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            self.img2superclass.append(self.cls2supercls[self.img2cls[ann["image"]]])

            # Preprocess caption
            caption_path = os.path.join(self.root, ann["caption"])
            caption = pickle.load(open(caption_path, "rb"))["caption"]
            caption = pre_caption(caption, self.max_words)

            self.text.append(caption)

            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1

        print(
            f"Number of samples for head classes: {len([i for i in self.img2superclass if i == 0])}"
        )
        print(
            f"Number of samples for mid classes: {len([i for i in self.img2superclass if i == 1])}"
        )
        print(
            f"Number of samples for tail classes: {len([i for i in self.img2superclass if i == 2])}"
        )

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        image_id = self.ann[index]["image"]
        image_path = os.path.join(self.root, image_id)
        image = torch.load(image_path, map_location="cpu", weights_only=True)
        image = Image.fromarray(image.numpy()).convert("RGB")
        image = self.transform(image)

        class_ = torch.tensor((self.img2cls[image_id]))

        return {
            "image": image,
            "caption": self.text[index],
            "class_": class_,
            "index": index,
            "key": image_id,
        }

    def _setup_superclasses(self, img2cls):
        # Create a mapping from class to superclass
        # 0 -> head class
        # 1 -> mid class
        # 2 -> tail class
        counter = Counter(img2cls.values()).most_common()
        classes = [int(class_) for class_, _ in counter]

        assert len(classes) % 3 == 0, "Number of classes should be divisible by 3"

        interval = len(classes) // 3
        superclasses = {}
        for i, class_ in enumerate(classes):
            if i < interval:
                superclasses[class_] = 0
            elif i < 2 * interval:
                superclasses[class_] = 1
            else:
                superclasses[class_] = 2

        return superclasses
