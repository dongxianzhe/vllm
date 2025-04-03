import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--input-name', type=str)
parser.add_argument('--output-name', type=str)
args = parser.parse_args()

log_file = f"{args.dir}/{args.input_name}_api_server.log"

patterns = [
    re.compile(r"step latency (\d+\.\d+)"),
    re.compile(r"encode latency (\d+\.\d+)"),
    re.compile(r"language latency (\d+\.\d+)"),
]

output_files = [
    f"{args.dir}/{args.output_name}_step_latency.log",
    f"{args.dir}/{args.output_name}_encode latency.log",
    f"{args.dir}/{args.output_name}_language latency.log",
]

for pattern, output_file in zip(patterns, output_files):
    latencies = []
    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                latencies.append(match.group(1))
    latencies.sort(reverse=True)
    with open(output_file, "w") as out:
        for latency in latencies:
            out.write(f"{latency}\n")