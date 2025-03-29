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
        self.num_requests = num_requests
        self.max_index = len(self.dataset)
        self.requests_made = 0

    def __iter__(self):
        return self

    def __next__(self) -> SyntheticDataEntry:
        if self.requests_made >= self.num_requests:
            raise StopIteration
        
        index = random.randint(0, self.max_index - 1)
        data = self.dataset[index]
        self.requests_made += 1
        entry = SyntheticDataEntry(
            prompt = data['question'], 
            image = encode_base64_content_from_image(data['image']), 
            ttft_slo = 2.,
            tpot_slo = 0.08, 
        )
        return entry

    def __len__(self):
        return self.num_requests