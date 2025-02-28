import torch
import webdataset as wds
from dataset.utils import pre_caption
import PIL.Image as Image

from multiprocessing import Value

def get_dataset_size():
    # https://github.com/AILab-CVC/SEED/blob/93b3cf408196735ec4820ad2eb4d9dc4a670003d/MultiModalLLM/src/data/data.py#L73C1-L74C1
    return 2905954

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    if "No images in sample" in str(exn) or "Only one image in sample" in str(exn):  # Avoid spamming logs with these
        return True
    print(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


def make_dataset_train(input_shards, transform, max_words=30, cache_dir=None, batch_size=128):

    # TODO: Change `torch.tensor(-1.0)` to correct values when using different loss than CLIP
    def make_sample(sample):
        image = sample["image.pth"]
        if len(image.shape) == 2:
            # Grayscale
            image = image.unsqueeze(0)  # Shape becomes [1, H, W]
        elif len(image.shape) == 3:
            # 3D images, permute to get [C, H, W]
            image = image.permute(2, 0, 1)

        caption = sample["metadata.pyd"]["caption"]
        return transform(image), pre_caption(caption=caption, max_words=max_words), torch.tensor(-1.0), torch.tensor(-1.0)
    
    train_set = wds.WebDataset(
        urls=input_shards,
        resampled=True,
        shardshuffle=True,
        cache_dir=cache_dir,
        nodesplitter=wds.split_by_node
    )

    train_set = train_set.shuffle(1000).decode("pil").map(make_sample)
    train_set = train_set.batched(batch_size)

    return train_set

def make_dataloader_train(trainset, batch_size=128, num_workers=4):
    trainloader = wds.WebLoader(trainset, batch_size=None, num_workers=num_workers)

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    trainloader = trainloader.unbatched().shuffle(1000).batched(batch_size)

    # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
    trainloader = trainloader.with_epoch(get_dataset_size() // batch_size)

    return trainloader