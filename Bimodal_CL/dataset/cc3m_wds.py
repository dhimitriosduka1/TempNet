import torch
import webdataset as wds
from dataset.utils import pre_caption
import PIL.Image as Image

def make_dataset_train(input_shards, transform, max_words=30, cache_dir=None, batch_size=128):

    # TODO: Change `torch.tensor(-1.0)` to correct values when using different loss than CLIP
    def make_sample(sample):
        image = sample["image.pth"].permute(2, 1, 0)
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

    for sample in train_set:
        print(sample)
        break

    return train_set

def make_dataloader_train():
    pass

# CC3M (train): 2905954