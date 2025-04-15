import argparse
import pickle
import json
from dataclasses import dataclass, asdict
from typing import List
from benchmark_metric import BenchmarkMetricsAnalyzer, MethodResults, BenchmarkMetricsAnalysisResult
import statistics

def main(args: argparse.Namespace):
    analyszer = BenchmarkMetricsAnalyzer(
        ttft_slo = [1, 2, 4, 5, 10, 15, 20, 25, 30, 40, 50], 
        tpot_slo = [ms / 1000 for ms in list(range(80, 500, 10))]
    )
    for method_results_path in args.method_results_paths:
        analyszer.add_method(data=method_results_path)

    results: list[BenchmarkMetricsAnalysisResult] = analyszer.analysis() 
    results = sorted(results, key=lambda x: statistics.variance(x.methods_score))
    
    for result in results:
        result.print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load multiple result files.")
    parser.add_argument(
        '--method-results-paths', 
        type=str, 
        nargs='+', 
        required=True,
        help='A list of paths to method result files (.pkl)'
    )
    args = parser.parse_args()
    main(args)