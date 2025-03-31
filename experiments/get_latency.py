import re

log_file = "result/vllm_api_server.log"

pattern = re.compile(r"step latency (\d+\.\d+)")

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            print(match.group(1))
