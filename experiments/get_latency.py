import re
import argparse
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument('--log-path', type=str, help='apiserver log file path')
args = parser.parse_args()

latency_process_data = [
    ("step_latency"  , re.compile(r"step latency (\d+\.\d+)")    , []), 
    ("encode_latency", re.compile(r"encode latency (\d+\.\d+)")  , []), 
    ("decode_latency", re.compile(r"language latency (\d+\.\d+)"), []), 
]

for name, pattern, data_list in latency_process_data:
    with open(args.log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                data_list.append(match.group(1))


headers = [
    "Step_Latency(s)", 
    "Encode_Latency(s)", 
    "Decode_Latency(s)", 
    ]
data = []

for i in range(3):
    latency_process_data[i][2].sort()

num_latencies = min([
    len(latency_process_data[0][2]), 
    len(latency_process_data[1][2]), 
    len(latency_process_data[2][2])]
)

for i in range(num_latencies):
    data.append((latency_process_data[0][2][i], latency_process_data[1][2][i], latency_process_data[2][2][i]))

data_table = tabulate(data, headers, tablefmt="plain")
print(data_table)