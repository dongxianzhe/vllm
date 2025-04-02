import numpy as np
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tabulate import tabulate

# model = "llava-hf/llava-1.5-7b-hf"
model = "/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf"
llm = LLM(model=model)
names = [
    "lmms-lab/TextCaps", 
    "lmms-lab/POPE", 
    "lmms-lab/MME", 
    "lmms-lab/textvqa", 
    "lmms-lab/VizWiz-VQA", 
]

headers = ["dataset", "avg prefill tokens", "avg decode tokens"]
table_data = []

for name in names:
    print(f'----------------------------- {name} -------------------------------')
    dataset = load_dataset(name, split="test")
    requests = []
    for entry in dataset:
        requests.append({
            "prompt": f"USER: <image>\n{entry['question']}\nASSISTANT:",
            "multi_modal_data": {"image": entry['image']},
        })
    sampling_params = SamplingParams(max_tokens=1024)
    outputs = llm.generate(requests, sampling_params=sampling_params)
    decode_workload = []
    prefill_workload = []
    for o in outputs:
        prefill_workload.append(len(o.prompt_token_ids))
        decode_workload.append(len(o.outputs[0].token_ids))
    table_data.append((name, np.mean(prefill_workload), np.mean(decode_workload)))

    for i in range(10):
        o = outputs[i]
        print(o.outputs[0].text)

workload_table = tabulate(table_data, headers, tablefmt="plain")
with open("result/dataset_analysis.log", "w") as f:
    f.write(workload_table)