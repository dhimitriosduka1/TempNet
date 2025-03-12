# Source: https://github.com/LijieFan/LaCLIP/blob/main/rewrite/utils.py

import os
import sys
import math
import fire
import time
import json
import torch
import random

from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from fairscale.nn.model_parallel.initialize import initialize_model_parallel


# Used for In-Context Learning
human_source_caption_list = [
    "Honey buttermilk biscuits on a cooling rack being drizzled with honey",
    "happy corgi time",
    "<PERSON> dog looking at dirt from the ground",
    "navy vintage pants - lime green bag - ivory Maison Simons t-shirt - Zara clogs",
    "Ooak Barbie City Shine",
    "Real Wedding on a NYC Rooftop",
    "the proud of my beloved italian bracco after leg amputation due to a tumor.",
    "Pineapple Wearing Headphones Art Print by Philip Haynes",
    "Ominous thunderclouds behind the Capitol Building",
    "Steampunk woman with gun",
    "a new watch with some old friends",
    "Particularly important to Africa is the East African Highland Banana (EAHB), a staple food for 80 million people. Uganda alone has about 120 varieties of this type of banana.",
    "Electric Blue Guitar There Goes My Hero, Rock The Vote, <PERSON>, <PERSON>, Music Photo, Red Eyes, Photo Quotes, Electric Blue, Music Lyrics",
    "Advanced Bicycle Skills Video - Valuable Video for Safe Cycl",
    "grilled turkey pesto sandwich",
    "Actress <PERSON> during the launch of international fashion brand Forever 21 store at a mall in Mumbai on Saturday, October 12th, 2013.",
]

# Used for In-Context Learning
human_target_caption_list = [
    "A warm stack of freshly baked honey buttermilk biscuits, sit on a cooling rack as they are drizzled with golden honey",
    "Delighted corgi stands in the hallway, looking at its owner",
    "<Person>'s dog, lying on the ground, looks at the dirt",
    "A young beautiful lady wearing navy vintage pants and ivory Maison Simons t-shirt, is holding a lime green bag.",
    "A custom-made Barbie doll with a city-inspired look shines brightly",
    "a couple is kissing each other during their rooftop wedding in NYC",
    "my italian bracco lied down proudly under the sunshile, despite of leg amputation due to a tumor.",
    "An art from Philip Haynes depicts a pineapple that wears headphones",
    "Thunderclouds loom over the Capitol Building, casting a dark shadow",
    "A fierce and stylish steampunk woman holds a toy revolver in her hands",
    "The watch sits besides a cartoon picture, evoking memories of cherished times shared with long-time friends",
    "An African man holds a bunch of bananas, which is particularly important to Africa",
    "<PERSON> is playing an electric blue guitar, eyes bloodshot from the stage lights",
    "A Cyclist is demonstrating advanced bicycle skills in a video that will help people stay safe.",
    "A grilled turkey pesto sandwich with melted cheese and fresh arugula is served on a plate.",
    "The young beautiful actress attended the launch of fashion brand Forever 21 at a mall.",
]


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    prompt_filename: str = "text/source.txt",
    output_filename: str = "text/target.txt",
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    new_prompt_filename = output_filename

    # Change parent directory to output
    with open(prompt_filename, "r") as f:
        original_prompts = f.readlines()

    new_prompts = []
    num_batches = math.ceil(len(original_prompts) / max_batch_size)

    for batch_idx in tqdm(range(num_batches)):
        prompts = []
        current_batch = original_prompts[
            batch_idx * max_batch_size : (batch_idx + 1) * max_batch_size
        ]

        for _, original_prompt in enumerate(current_batch):
            chosen_source_caption_list = []
            chosen_target_caption_list = []
            num_caps = len(human_source_caption_list)
            chosen_idx = random.sample(range(num_caps), 3)
            for idx in chosen_idx:
                chosen_source_caption_list.append(human_source_caption_list[idx])
                chosen_target_caption_list.append(human_target_caption_list[idx])

            current_prompt = """Write image captions differently,

            {} => {}

            {} => {}

            {} => {}

            {} =>""".format(
                chosen_source_caption_list[0],
                chosen_target_caption_list[0],
                chosen_source_caption_list[1],
                chosen_target_caption_list[1],
                chosen_source_caption_list[2],
                chosen_target_caption_list[2],
                original_prompt.replace("\n", ""),
            )
            prompt_tokens = generator.tokenizer.encode(
                current_prompt, bos=True, eos=False
            )
            if len(prompt_tokens) <= max_seq_len - 5:
                prompts.append(current_prompt)
            else:
                cut_len = max_seq_len - 10
                prompt_tokens = prompt_tokens[:cut_len]
                current_prompt = generator.tokenizer.decode(prompt_tokens) + " =>"
                prompts.append(current_prompt)

        results = generator.generate(
            prompts, max_gen_len=77, temperature=temperature, top_p=top_p
        )

        for result in results:
            prompt_line = result.split("\n")[8].strip()
            new_prompt = prompt_line.split("=>")[1].strip()
            new_prompts.append(new_prompt)

        if local_rank == 0:
            with open(new_prompt_filename, "w") as f:
                f.writelines([p.strip().replace("\n", " ") + "\n" for p in new_prompts])


if __name__ == "__main__":
    fire.Fire(main)
