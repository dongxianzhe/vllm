import re

log_file = "result/vllm_api_server.log"

patterns = [
    re.compile(r"step latency (\d+\.\d+)"),
    re.compile(r"encode latency (\d+\.\d+)"),
    re.compile(r"language latency (\d+\.\d+)"),
]

output_files = [
    "result/step_latency.log",
    "result/encode latency.log",
    "result/language latency.log",
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