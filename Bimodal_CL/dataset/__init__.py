import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
import random

from dataset.caption_dataset import (
    ImageNet100ValDataset,
    ImageNet1kDataset,
    ImageNet1kValDataset,
    re_eval_dataset,
)
from dataset.randaugment import RandomAugment
from dataset.cc3m_wds import make_dataset_train, CC3M_Val_Dataset
from dataset.caption_dataset import ImageNet100Dataset


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def create_train_dataset(dataset, args, use_test_transform=False):

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    if dataset == "imagenet100":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.image_res, scale=(0.5, 1.0), interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2,
                7,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Equalize",
                    "Brightness",
                    "Sharpness",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if dataset == "imagenet100":
        return ImageNet100Dataset(
            root=args.train_image_root,
            transform=train_transform,
            noise_level=args.noise_level,
        )

    if dataset == "imagenet1k":
        return ImageNet1kDataset(
            root=args.train_image_root,
            transform=train_transform,
        )

    return make_dataset_train(transform=train_transform, args=args)


def create_val_dataset(
    dataset, args, val_file, val_image_root, test_file=None, load_cc3m_val=False
):

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    if dataset == "imagenet100":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (args.image_res, args.image_res), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if load_cc3m_val:
        return CC3M_Val_Dataset(
            ann_file=args.cc3m_ann_file,
            img2cls_file=args.cc3m_img2cls_file_val,
            transform=test_transform,
            root=args.cc3m_val_root,
        )

    if dataset == "re":
        val_dataset = re_eval_dataset(val_file, test_transform, val_image_root)

        if test_file is not None:
            test_dataset = re_eval_dataset(test_file, test_transform, val_image_root)
            return val_dataset, test_dataset
        else:
            return val_dataset

    elif dataset == "imagenet100":
        return ImageNet100ValDataset(root=val_image_root, transform=test_transform)
    elif dataset == "imagenet1k":
        return ImageNet1kValDataset(root=val_image_root, transform=test_transform)
    else:
        assert 0, dataset + " is not supported."


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_train_loader(
    dataset, sampler, batch_size, num_workers, collate_fn, drop_last
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        drop_last=drop_last,
        prefetch_factor=4,
    )


def create_val_loader(datasets, samplers, batch_size, num_workers, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, collate_fns
    ):
        shuffle = False
        drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            prefetch_factor=12,
        )
        loaders.append(loader)
    return loaders
