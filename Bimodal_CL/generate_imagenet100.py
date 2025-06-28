# Source: https://github.com/danielchyeh/ImageNet-100-Pytorch/blob/main/generate_IN100.py
import os
import shutil
import argparse
import json
from tqdm import tqdm


def parse_option():
    parser = argparse.ArgumentParser("argument for generating ImageNet-100")

    parser.add_argument(
        "--source_folder", type=str, default="", help="folder of ImageNet-1K dataset"
    )
    parser.add_argument(
        "--target_folder", type=str, default="", help="folder of ImageNet-100 dataset"
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default="/BS/dduka/work/projects/TempNet/Bimodal_CL/dataset/Labels.json",
        help="class file of ImageNet-100",
    )

    opt = parser.parse_args()

    return opt


def generate_data(source_folder, target_folder, target_class):
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(target_class, "r") as f:
        labels = json.load(f)

    f = []
    for key, _ in labels.items():
        f.append(f"{key}.tar")

    for ids, dirs in enumerate(tqdm(os.listdir(source_folder))):
        # Check if the directory is in the labels
        if dirs in f:
            print("{} is transferred".format(dirs))
            shutil.copytree(
                os.path.join(source_folder, dirs), os.path.join(target_folder, dirs)
            )


opt = parse_option()
generate_data(opt.source_folder, opt.target_folder, opt.target_class)
