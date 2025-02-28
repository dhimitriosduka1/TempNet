import io
import torch
import webdataset as wds
from dataset.utils import pre_caption
import PIL.Image as Image

def get_dataset_size():
    # https://github.com/AILab-CVC/SEED/blob/93b3cf408196735ec4820ad2eb4d9dc4a670003d/MultiModalLLM/src/data/data.py#L73C1-L74C1
    return 2905954

def decoder_pth(key, value):
    if "image.pth" in key:
        image = torch.load(io.BytesIO(value))
        image = Image.fromarray(image.numpy()).convert("RGB")
        return image

def make_dataset_train(input_shards, transform, max_words=30, cache_dir=None, batch_size=128):
    # TODO: Change `torch.tensor(-1.0)` to correct values when using different loss than CLIP
    def make_sample(sample):
        image = sample["image.pth"]
        caption = sample["metadata.pyd"]["caption"]
        return transform(image), pre_caption(caption=caption, max_words=max_words), torch.tensor(-1.0), torch.tensor(-1.0)
       
    train_set = wds.WebDataset(
        urls=input_shards,
        resampled=True,
        shardshuffle=True,
        cache_dir=cache_dir,
        nodesplitter=wds.split_by_node
    )

    train_set = train_set.shuffle(1000).decode(decoder_pth).map(make_sample)
    train_set = train_set.batched(batch_size)

    return train_set

def make_dataloader_train(trainset, batch_size=128, num_workers=4):
    trainloader = wds.WebLoader(trainset, batch_size=None, num_workers=num_workers)

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    trainloader = trainloader.unbatched().shuffle(1000).batched(batch_size)

    # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
    trainloader = trainloader.with_epoch(get_dataset_size() // batch_size)

    return trainloader