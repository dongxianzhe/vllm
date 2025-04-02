import os
import base64
import random
from datasets import load_dataset
from dataclasses import dataclass
from PIL import Image
from io import BytesIO


def encode_base64_content_from_image(image: Image.Image) -> str:
    buffered = BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_string


@dataclass
class SyntheticDataEntry:
    prompt: str
    image: str
    ttft_slo: float
    tpot_slo: float

class SyntheticDataset:
    def __init__(
        self, 
        num_requests: int,
        textcaps: int = 1, 
    ):
        self.dataset = load_dataset("lmms-lab/TextCaps", split="test")
        self.entries: list[SyntheticDataEntry] = []
        for data in self.dataset:
            entry = SyntheticDataEntry(
                prompt = data['question'], 
                image = encode_base64_content_from_image(data['image']), 
                ttft_slo = 2.,
                tpot_slo = 0.16, 
            )
            self.entries.append(entry)
            if len(self.entries) == num_requests:
                break
        self.num_requests = num_requests
        self.max_index = len(self.dataset)
        self.test = os.getenv("TEST", "0") == '1'

    def __len__(self):
        return self.num_requests

    def __getitem__(self, i: int) -> SyntheticDataEntry:
        return self.entries[i]