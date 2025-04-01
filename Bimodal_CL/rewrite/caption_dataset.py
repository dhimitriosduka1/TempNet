import torch
import json
import os


class CaptionDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading captions."""

    def __init__(self, data_path):
        self.data_path = data_path

        # Each caption is in the following format:
        # image_id: caption_text
        self.captions = [
            {"image_id": image_id, "caption": caption}
            for image_id, caption in self.load_captions().items()
        ]

    def load_captions(self):
        """Load captions from a JSON file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} not found.")
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]
