import os
import json
import torch
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from captions_dataset import CaptionsDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# Constants
BATCH_SIZE = 64
MODEL_NAME = "tuner007/pegasus_paraphrase"
TEMPORARY_SAVE_PATH = (
    "/BS/dduka/work/projects/TempNet/Bimodal_CL/cc3m/merged_checkpoint.json"
)
NUM_BEAMS = 10
NUM_RETURN_SEQUENCES = 5


# Initialize distributed process group
init_process_group(backend="nccl")  # NCCL is optimized for multi-GPU training

# Get GPU information for each process
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)


def get_response(input_text, num_return_sequences, num_beams):
    batch = tokenizer(
        input_text,
        truncation=True,
        padding="longest",
        max_length=60,
        return_tensors="pt",
    ).to(torch_device)

    output = model.module.generate(
        **batch,
        max_length=60,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        temperature=0.7,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
        early_stopping=True,
    )
    tgt_text = tokenizer.batch_decode(output, skip_special_tokens=True)

    # Split the senteces into lists of num_return_sequences
    tgt_text = [
        tgt_text[i : i + num_return_sequences]
        for i in range(0, len(tgt_text), num_return_sequences)
    ]

    return tgt_text


torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(torch_device)
model = DDP(model, device_ids=[local_rank])
model.eval()

caption_dataset = CaptionsDataset(
    path="/BS/dduka/work/projects/TempNet/Bimodal_CL/cc3m/merged.json"
)

dataloader = torch.utils.data.DataLoader(
    caption_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    sampler=DistributedSampler(caption_dataset),
)

results = {}

for i, sample in enumerate(dataloader):
    if rank == 0:
        print(f"Processing batch {i}/{len(dataloader)}")

    sample_ids, captions = sample["sample_id"], sample["caption"]
    output = get_response(captions, NUM_RETURN_SEQUENCES, NUM_BEAMS)

    # Gather the outputs from all processes. Output is a list of lists of strings
    gathered_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_outputs, output)
    gatherted_outputs = [item for sublist in gathered_outputs for item in sublist]

    gathered_sample_ids = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_sample_ids, sample_ids)
    gatherted_sample_ids = [item for sublist in gathered_sample_ids for item in sublist]

    gathered_caption = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_caption, captions)
    gatherted_caption = [item for sublist in gathered_caption for item in sublist]

    if rank == 0:
        for _, (sample_id, caption, output) in enumerate(
            zip(gatherted_sample_ids, gatherted_caption, gatherted_outputs)
        ):
            results[sample_id] = {"caption": caption, "paraphrases": output}

        if i % 100 == 0 and i > 0:
            print(f"Saving temporary results at iteration {i}")
            with open(TEMPORARY_SAVE_PATH, "w") as f:
                json.dump(results, f)

# Save the final results
if rank == 0:
    print("Saving final results")
    with open(TEMPORARY_SAVE_PATH, "w") as f:
        json.dump(results, f)

destroy_process_group()
