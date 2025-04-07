import re
import argparse

def extract_metrics(file_path):
    patterns = {
        "Mean TTFT (ms)": r"Mean TTFT \(ms\):\s+([0-9]*\.?[0-9]+)",
        "P99 TPOT (ms)":  r"P99 TPOT \(ms\):\s+([0-9]*\.?[0-9]+)"
    }

    with open(file_path, 'r') as file:
        content = file.read()

        for name, pattern in patterns.items():
            matches = re.findall(pattern, content)
            for match in matches:
                print(f"{name}: {match}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Mean TTFT and P99 TPOT values from a file.")
    parser.add_argument("filename", help="Path to the input file.")
    args = parser.parse_args()

    extract_metrics(args.filename)
