import argparse
import json
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from caption_dataset import CaptionDataset

# --- Configuration ---
DEFAULT_BATCH_SIZE = 64
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_NUM_AUGMENTATIONS = 5
DEFAULT_MAX_NEW_TOKENS = 75
DEFAULT_DATA_PATH = "/BS/dduka/work/databases/cc3m/train/captions.json"
OUTPUT_DIR = "."

# PROMPT_TEMPLATE = """
# Generate {num_augmentations} diverse paraphrases for the following image caption. Ensure each paraphrase is DESCRIPTIVE, focusing on the visual elements rather than telling a story.

# Avoid narrative paraphrases that interpret actions sequentially or add character motivations not in the original caption.

# Do not include the original sentence in your output.

# Examples:

# Original Caption: Honey buttermilk biscuits on a cooling rack being drizzled with honey
# Paraphrase: A warm stack of freshly baked honey buttermilk biscuits sits on a cooling rack, being drizzled with golden honey.

# Original Caption: Happy corgi time
# Paraphrase: A delighted corgi stands in a hallway, looking towards its owner.

# Original Caption: Pineapple Wearing Headphones Art Print by Philip Haynes
# Paraphrase: An art print by Philip Haynes depicts a pineapple wearing headphones.

# Output only the {num_augmentations} paraphrased sentences, each on a new line, starting with the number.

# Original Sentence: "{caption}"

# Descriptive Paraphrases:
# 1. """

PROMPT_TEMPLATE = """
Generate {num_augmentations} diverse paraphrases for the following image caption. Follow these requirements:

1. Each paraphrase must preserve the original meaning completely
2. Use different vocabulary and sentence structure for each paraphrase
3. Ensure no two paraphrases follow the same grammatical pattern
4. Keep each paraphrase between 10-20 words
5. Use descriptive language that states facts/observations rather than telling a story
6. Do not include the original sentence in your output
7. Do not start multiple paraphrases with the same word or phrase

Original Caption: "{caption}"

Paraphrases:
1.
"""


def parse_generated_text(text, num_expected):
    """Parses the model's output to extract individual augmentations."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned_lines = []
    for line in lines:
        # There are some cases where the model outputs \"
        line = line.replace('"', "")

        # Basic cleaning: remove potential numbering/bullets
        if line.startswith(tuple(f"{i}." for i in range(1, 10))) or line.startswith(
            ("- ", "* ")
        ):
            # Check if there's content after the numbering/bullet before splitting
            parts = line.split(" ", 1)
            if len(parts) > 1:
                cleaned_lines.append(parts[1].strip())
            # else: # Handle cases like "1." with no text after (ignore)
            #    pass
        else:
            cleaned_lines.append(line)

    # Return up to num_expected, filtering any short/empty lines again
    return [line for line in cleaned_lines if len(line) > 5][:num_expected]


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse augmentations using a Llama model and PyTorch Dataset."
    )
    # --- Adjusted Arguments ---
    parser.add_argument(
        "--data_path",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the data file needed by the custom Dataset (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--output_file",
        help="Path to save the results (JSON format).",
        default="output.json",
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model name (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=DEFAULT_NUM_AUGMENTATIONS,
        help=f"Number of augmentations per caption (default: {DEFAULT_NUM_AUGMENTATIONS}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch size (per device) (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max new tokens to generate *per augmentation* (approx) (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus sampling probability."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for model loading.",
    )

    args = parser.parse_args()

    accelerator = Accelerator()

    dataset = CaptionDataset(data_path=args.data_path)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    prepared_data_loader = accelerator.prepare(data_loader)

    accelerator.print(f"Loading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        accelerator.print("Set tokenizer pad_token to eos_token.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16,
    )
    accelerator.print(f"Model loaded on device(s): {model.hf_device_map}")
    model.eval()

    all_results_list = []

    pbar = tqdm(
        total=len(prepared_data_loader),
        disable=not accelerator.is_main_process,
        desc="Generating Augmentations",
    )

    i = 0

    print(f"Starting generation with batch size {args.batch_size} per device...")
    for batch in prepared_data_loader:
        batch_captions = batch["caption"]
        batch_image_ids = batch["image_id"]

        # Construct prompts for the batch
        prompts = [
            PROMPT_TEMPLATE.format(
                num_augmentations=args.num_augmentations, caption=caption
            )
            for caption in batch_captions
        ]

        # Tokenize batch
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs.to(
                    model.device
                ),  # Ensure inputs are on the correct device segment
                max_new_tokens=args.max_new_tokens * args.num_augmentations
                + 50,  # Add buffer
                num_return_sequences=1,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and Parse
        generated_texts = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        gathered_image_ids = gather_object(batch_image_ids)
        gathered_captions = gather_object(batch_captions)
        gathered_generated_texts = gather_object(generated_texts)

        if accelerator.is_main_process:
            batch_results = {}
            for original_caption, image_id, generated_text in zip(
                gathered_captions, gathered_image_ids, gathered_generated_texts
            ):
                parsed_augmentations = parse_generated_text(
                    generated_text, args.num_augmentations
                )

                batch_results[image_id] = {
                    "original": original_caption,
                    "paraphrases": parsed_augmentations,
                }

            all_results_list.append(batch_results)
            pbar.update(1)

        # Save a checkpoint every 200 batches
        if i % 10 == 0 and i !=0 and accelerator.is_main_process:
            with open(
                f"/BS/dduka/work/projects/TempNet/Bimodal_CL/submit/dhimitrios/llama_rewrite/checkpoint_{i}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(all_results_list, f, indent=4, ensure_ascii=False)
            print(f"Checkpoint saved at batch {i}.")
        
        i += 1

    # --- Finalize and Save Results (Main Process Only) ---
    if accelerator.is_main_process:
        pbar.close()

        # Combine results from all gathered dictionaries
        final_results = {}
        for res_dict in all_results_list:
            final_results.update(res_dict)  # Combine dicts from all batches/processes

        # Check for potential data loss due to non-unique keys (original captions)
        original_caption_count = sum(
            len(d) for d in all_results_list
        )  # Count items across all gathered dicts
        if len(final_results) < original_caption_count:
            print(
                "\nWarning: Some results might have been overwritten due to duplicate original captions."
            )
            print(f"  Processed items (approx): {original_caption_count}")
            print(f"  Unique keys in output: {len(final_results)}")
        elif len(final_results) < len(dataset):
            print(
                f"\nWarning: Generated results for {len(final_results)} captions, but dataset had {len(dataset)}. Some items might have been skipped or failed."
            )
        else:
            print(
                f"\nGenerated augmentations for {len(final_results)} unique captions."
            )

        # Save results
        try:
            with open(
                "/BS/dduka/work/projects/TempNet/Bimodal_CL/submit/dhimitrios/llama_rewrite/"
                + args.output_file,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(final_results, f, indent=4, ensure_ascii=False)
            print(f"Results saved to {args.output_file}")
        except Exception as e:
            print(f"Error saving results to {args.output_file}: {e}")

    accelerator.wait_for_everyone()
    print("Processing finished.")


if __name__ == "__main__":
    main()
