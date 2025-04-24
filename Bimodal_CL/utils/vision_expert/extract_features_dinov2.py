import os
import argparse
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import glob
from tqdm import tqdm
import gc
import logging
import time
import webdataset as wds
from io import BytesIO
import torchvision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("embedding_extraction.log")],
)
logger = logging.getLogger(__name__)


def preprocess_image(sample, processor):
    """Process a WebDataset sample to prepare it for the model"""
    try:
        image_bytes = sample["image.pth"]
        image = torch.load(BytesIO(image_bytes), weights_only=True)
        image = Image.fromarray(image.numpy()).convert("RGB")

        # Write image to a temporary file
        # temp_file = "temp_image.jpg"
        # image.save(temp_file)

        # Process with the model's processor
        inputs = processor(images=image, return_tensors="pt")

        # # Save the processed image to a temporary file
        # temp_file = "temp_image_processed.jpg"

        # # Unnormalize
        # unnorm = transforms.Normalize(
        #     mean=[-m / s for m, s in zip(processor.image_mean, processor.image_std)],
        #     std=[1 / s for s in processor.image_std],
        # )
        # img_tensor = unnorm(inputs["pixel_values"][0]).clamp(0, 1)
        # torchvision.transforms.ToPILImage()(img_tensor).save(temp_file)

        # Remove batch dimensionf
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        # Return the processed inputs and the key (if available)
        key = sample["__key__"]
        return inputs, key
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None


def extract_embeddings(
    model, loader, device, batch_size, save_dir, save_interval=10000, embedding_dim=1024
):
    """Extract embeddings and save them periodically to manage memory"""
    model.eval()
    model.to(device)

    total_processed = 0
    current_batch = 0
    embeddings_dict = {}

    # Estimate memory usage
    single_embedding_bytes = np.dtype(np.float32).itemsize * embedding_dim
    logger.info(
        f"Each embedding vector uses {single_embedding_bytes} bytes (dimension: {embedding_dim})"
    )

    # REMOVE
    unique_keys = set()

    with torch.no_grad():
        for batch_idx, (inputs, keys) in enumerate(
            tqdm(loader, desc="Extracting embeddings")
        ):
            unique_keys.update(keys)

            # if inputs is None:
            #     logger.warning("Empty batch encountered, skipping...")
            #     continue

            # # Move inputs to device
            # for k, v in inputs.items():
            #     inputs[k] = v.to(device)

            # # Get embeddings
            # outputs = model(**inputs, output_hidden_states=True)

            # # Use the [CLS] token embedding from the last hidden state
            # batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # # Store embeddings with their keys
            # for i, key in enumerate(keys):
            #     embeddings_dict[key] = batch_embeddings[i]

            # total_processed += len(keys)
            # current_batch += len(keys)

            # # Save periodically to manage memory
            # if current_batch >= save_interval:
            #     part_num = total_processed // save_interval
            #     save_path = os.path.join(save_dir, f"embeddings_part_{part_num}.npz")
            #     np.savez_compressed(save_path, **embeddings_dict)

            #     # Clear memory
            #     embeddings_dict = {}
            #     current_batch = 0

            #     # Force garbage collection
            #     gc.collect()
            #     if device == "cuda":
            #         torch.cuda.empty_cache()

    # # Save any remaining embeddings
    # if embeddings_dict:
    #     part_num = (total_processed // save_interval) + 1
    #     save_path = os.path.join(save_dir, f"embeddings_part_{part_num}.npz")
    #     np.savez_compressed(save_path, **embeddings_dict)

    #     # Log file size
    #     file_size = os.path.getsize(save_path)

    # REMOVE
    total_processed = len(unique_keys)
    logger.info(f"Processed {total_processed} unique keys")
    return total_processed


def merge_npz_files(save_dir, final_output_path):
    """Merge all part files into a single NPZ file"""
    logger.info("Merging NPZ files...")

    # Get all part files
    part_files = sorted(glob.glob(os.path.join(save_dir, "embeddings_part_*.npz")))

    if not part_files:
        logger.error("No part files found to merge")
        return

    # Get total size of part files before merging
    total_parts_size = sum(os.path.getsize(f) for f in part_files)

    # Merge all embeddings into a single dictionary
    all_embeddings = {}
    for part_file in tqdm(part_files, desc="Merging files"):
        with np.load(part_file) as data:
            for key in data.files:
                all_embeddings[key] = data[key]

    # Save the merged dictionary
    np.savez_compressed(final_output_path, **all_embeddings)
    merged_size = os.path.getsize(final_output_path)

    logger.info(
        f"Merged {len(part_files)} files containing {len(all_embeddings)} embeddings"
    )
    compression_ratio = total_parts_size / merged_size if merged_size > 0 else 0
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")

    # Optionally, remove part files to save space
    if args.remove_parts:
        for part_file in part_files:
            os.remove(part_file)
        logger.info("Removed part files")


def create_webdataset_loader(tar_paths, processor, batch_size, num_workers):
    """Create a WebDataset data loader for tar files"""

    # Create the dataset pipeline
    dataset = (
        wds.WebDataset(tar_paths)
        .map(lambda sample: preprocess_image(sample, processor))
        .batched(batch_size)
    )

    # Create the loader
    loader = wds.WebLoader(
        dataset,
        batch_size=None,  # Already batched in the dataset
        num_workers=num_workers,
    )

    return loader


def main(args):
    start_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load DINOv2 model and processor
    logger.info(f"Loading DINOv2 model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name)
    processor = AutoImageProcessor.from_pretrained(args.model_name)

    # Set processor parameters
    processor.image_mean = (0.48145466, 0.4578275, 0.40821073)
    processor.image_std = (0.26862954, 0.26130258, 0.27577711)

    # Get all tar file paths
    logger.info(f"Finding tar files in {args.tar_dir}")
    tar_paths = sorted(glob.glob(os.path.join(args.tar_dir, "*.tar")))
    logger.info(f"Found {len(tar_paths)} tar files")

    if len(tar_paths) == 0:
        logger.error("No tar files found. Exiting.")
        return

    # Create WebDataset loader
    logger.info("Creating WebDataset loader")
    loader = create_webdataset_loader(
        tar_paths, processor, args.batch_size, args.num_workers
    )

    # Extract embeddings
    logger.info(f"Starting embedding extraction using device: {args.device}")
    total_processed = extract_embeddings(
        model,
        loader,
        args.device,
        args.batch_size,
        args.output_dir,
        args.save_interval,
        embedding_dim=1024,
    )

    # Merge all part files into a single NPZ file if requested
    if args.merge_output:
        final_output_path = os.path.join(args.output_dir, "dinov2_embeddings_all.npz")
        merge_npz_files(args.output_dir, final_output_path)

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(f"Completed processing {total_processed} images")
    logger.info(f"Total processing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Average time per image: {elapsed_time/total_processed:.4f}s")

    # Print memory usage
    memory_usage = ""
    if args.device == "cuda" and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        memory_usage = f", GPU memory: {memory_allocated:.2f} GB"

    process_memory = 0
    try:
        import psutil

        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / (1024**3)  # GB
        memory_usage += f", RAM: {process_memory:.2f} GB"
    except ImportError:
        pass

    logger.info(f"Peak memory usage{memory_usage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 embeddings from images in tar files"
    )
    parser.add_argument("--tar_dir", type=str, help="Directory containing tar files")
    parser.add_argument("--output_dir", type=str, help="Directory to save embeddings")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov2-large",
        help="DINOv2 model name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for processing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use for computation",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10000,
        help="Save embeddings after processing this many images",
    )
    parser.add_argument(
        "--merge_output",
        action="store_true",
        help="Merge all part files into a single NPZ file",
    )
    parser.add_argument(
        "--remove_parts", action="store_true", help="Remove part files after merging"
    )

    args = parser.parse_args()

    main(args)
