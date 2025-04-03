import os
import base64
import random
from datasets import load_dataset
from dataclasses import dataclass
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor


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
        tpot_slo: float = 2., 
        ttft_slo: float = 0.16, 
        textcaps: int = 1, 
        pope: int = 0, 
        mme: int = 0, 
        text_vqa: int = 0,
        vizwiz_vqa: int = 0, 
    ):
        self.test = os.getenv("TEST", "0") == '1'
        self.num_requests = num_requests
        datasets = []
        dataset_iters = []
        datasets_name: list[str] = []
        weights: list[int] = []

        flags = [
            textcaps, 
            pope, 
            mme, 
            text_vqa, 
            vizwiz_vqa, 
        ]
        names = [
            "lmms-lab/TextCaps", 
            "lmms-lab/POPE", 
            "lmms-lab/MME", 
            "lmms-lab/textvqa", 
            "lmms-lab/VizWiz-VQA", 
        ]
        for flag, name in zip(flags, names):
            if flag > 0:
                dataset = load_dataset(name, split="test")
                datasets.append(dataset)
                dataset_iters.append(iter(dataset))
                datasets_name.append(name)
                weights.append(flag)

        chosen_datasets = random.choices(population=range(len(weights)), weights=weights, k=num_requests)

        self.entries: list[SyntheticDataEntry] = []
        tasks = []
        for i in chosen_datasets:
            dataset = datasets[i]
            dataset_iter = dataset_iters[i]
            name = datasets_name[i]
            if self.test:
                data = dataset[0]
            else:
                data = next(dataset_iter)
            tasks.append(data)

        def create_entry(task):
            data = task
            if name in [
                "lmms-lab/TextCaps", 
                "lmms-lab/POPE", 
                "lmms-lab/MME", 
                "lmms-lab/textvqa", 
                "lmms-lab/VizWiz-VQA", 
            ]: 
                entry = SyntheticDataEntry(
                    prompt = data['question'], 
                    image = encode_base64_content_from_image(data['image']), 
                    ttft_slo = 0,
                    tpot_slo = 0, 
                )
            else:
                raise Exception('invalid dataset')
            return entry

        with ThreadPoolExecutor(max_workers=32) as executor:
            self.entries = list(executor.map(create_entry, tasks))
        for entry in self.entries:
            entry.tpot_slo = tpot_slo
            entry.ttft_slo = ttft_slo

    def __len__(self):
        return self.num_requests

    def __getitem__(self, i: int) -> SyntheticDataEntry:
        return self.entries[i]

if __name__ == '__main__':
    import time
    start = time.perf_counter()
    dataset = SyntheticDataset(
        num_requests=100, 
        tpot_slo = 2., 
        ttft_slo = 0.16, 
        textcaps = 0, 
        pope = 1, 
        mme = 0, 
        text_vqa = 0,
        vizwiz_vqa = 0, 
    )
    end = time.perf_counter()
    print(f'dur {end - start}')
    for i in range(10):
        print(dataset[i].prompt)