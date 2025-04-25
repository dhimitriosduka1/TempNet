from glob import glob
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor, logging as transformers_logging
import numpy as np
from accelerate import Accelerator, logging as accelerate_logging
import webdataset as wds
from tqdm.auto import tqdm
import logging
from PIL import Image
import argparse

# Configure argument parser
parser = argparse.ArgumentParser(
    description="Extract DINOv2 embeddings from WebDataset"
)
parser.add_argument(
    "--model-name",
    type=str,
    default="facebook/dinov2-large",
    help="HuggingFace model name",
)
parser.add_argument(
    "--tar-files",
    type=str,
    required=True,
    help="Path to directory containing tar files",
)
parser.add_argument(
    "--output-dir",
    type=str,
    required=True,
)
parser.add_argument(
    "--batch-size", type=int, default=512, help="Batch size for processing"
)
parser.add_argument(
    "--num-workers", type=int, default=8, help="Number of workers for dataloader"
)
parser.add_argument(
    "--image-key", type=str, default="image.pth", help="Key for image in WebDataset"
)
parser.add_argument(
    "--key_key", type=str, default="__key__", help="Key for key in WebDataset"
)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

model_name = args.model_name
tar_files = args.tar_files
image_key = args.image_key
key_key = args.key_key
output_dir = args.output_dir

logger.info(f"Finding tar files in {tar_files}")
tar_paths = sorted(glob(os.path.join(tar_files, "*.tar")))
logger.info(f"Found {len(tar_paths)} tar files")

os.makedirs(output_dir, exist_ok=True)
logger.info(f"Embeddings will be saved to: {output_dir}")

accelerator = Accelerator()
logger.info(f"Using device: {accelerator.device}")

unprocessed_samples = []

try:
    processor = AutoProcessor.from_pretrained(model_name)

    # Set image mean and std for DINOv2 based on CC3M dataset
    processor.image_mean = (0.48145466, 0.4578275, 0.40821073)
    processor.image_std = (0.26862954, 0.26130258, 0.27577711)

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model = accelerator.prepare(model)
    logger.info(f"Model '{model_name}' loaded successfully.")
except Exception as e:
    logger.error(f"Error loading processor or model: {e}")
    exit(1)


def preprocess(sample):
    try:
        image = sample[image_key]
        key = sample[key_key]

        image = Image.fromarray(image.numpy()).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        return {"pixel_values": inputs.pixel_values[0], "key": key}
    except Exception as e:
        logger.error(f"Error processing sample {sample.get('__key__')}: {e}")
        unprocessed_samples.append(sample)
        return None


try:
    dataset = (
        wds.WebDataset(tar_paths, nodesplitter=wds.split_by_node)
        .shuffle(0)
        .decode("pil")
        .map(preprocess)
        .select(lambda x: x is not None)
        .batched(args.batch_size)
    )
    logger.info(f"WebDataset created from: {tar_paths}")
except Exception as e:
    logger.error(f"Error creating WebDataset: {e}")
    exit(1)

try:
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
    )
    dataloader = accelerator.prepare(dataloader)
    logger.info(f"DataLoader created with {args.num_workers} workers")

except Exception as e:
    logger.error(f"Error creating DataLoader: {e}")
    exit(1)

logger.info("Starting embedding extraction...")

for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting Embeddings")):
    pixel_values = batch["pixel_values"].to(accelerator.device)
    keys = batch["key"]

    with torch.no_grad():
        try:
            outputs = model(pixel_values)
            embeddings = outputs.last_hidden_state[:, 0, :]
        except Exception as e:
            logger.error(f"Error during model inference for batch {batch_idx}: {e}")
            continue

    embeddings_cpu = embeddings.cpu().numpy()

    for i, (key, embedding) in enumerate(zip(keys, embeddings_cpu)):
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        output_file = os.path.join(output_dir, f"{safe_key}.npz")
        try:
            np.savez_compressed(output_file, embedding=embedding)
        except Exception as e:
            logger.error(f"Error saving embedding for key {key} to {output_file}: {e}")

if accelerator.is_main_process:
    logger.info(f"Embedding extraction complete. Embeddings saved to {output_dir}")

if len(unprocessed_samples) > 0:
    logger.info(
        f"Some samples could not be processed. {len(unprocessed_samples)} unprocessed samples."
    )
    import json

    with open(os.path.join(output_dir, "unprocessed_samples.json"), "w") as f:
        json.dump(unprocessed_samples, f)
