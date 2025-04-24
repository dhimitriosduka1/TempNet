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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

model_name = "facebook/dinov2-large"

tar_files = "/BS/dduka/work/data/cc3m/validation/"

logger.info(f"Finding tar files in {tar_files}")
tar_paths = sorted(glob.glob(os.path.join(tar_files, "*.tar")))
logger.info(f"Found {len(tar_paths)} tar files")

image_key = "image.pth"
key_key = "__key__"

output_dir = "dinov2_large_embeddings"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"Embeddings will be saved to: {output_dir}")

accelerator = Accelerator()
logger.info(f"Using device: {accelerator.device}")

try:
    processor = AutoProcessor.from_pretrained(model_name)
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
        logger.error(f"Error processing sample {sample.get('__url__')}: {e}")
        return None


batch_size = 32
try:
    dataset = (
        wds.WebDataset(tar_files)
        .decode("pil")
        .map(preprocess)
        .select(lambda x: x is not None)
        .batched(batch_size)
    )
    logger.info(f"WebDataset created from: {tar_files}")
except Exception as e:
    logger.error(f"Error creating WebDataset: {e}")
    exit(1)

num_workers = 4
try:
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    dataloader = accelerator.prepare(dataloader)
    logger.info(f"DataLoader created with {num_workers} workers")

except Exception as e:
    logger.error(f"Error creating DataLoader: {e}")
    exit(1)

logger.info("Starting embedding extraction...")
processed_keys = set()

for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting Embeddings")):
    if not batch or "pixel_values" not in batch or len(batch["pixel_values"]) == 0:
        logger.warning(f"Skipping empty batch {batch_idx}")
        continue

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
        if key in processed_keys:
            logger.warning(f"Duplicate key found: {key}, skipping...")
            continue

        processed_keys.add(key)
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        output_file = os.path.join(output_dir, f"{safe_key}.npz")
        try:
            np.savez_compressed(output_file, embedding=embedding)
        except Exception as e:
            logger.error(f"Error saving embedding for key {key} to {output_file}: {e}")

if accelerator.is_main_process:
    logger.info(f"Embedding extraction complete. Embeddings saved to {output_dir}")
    logger.info(f"Total processed keys: {len(processed_keys)}")
