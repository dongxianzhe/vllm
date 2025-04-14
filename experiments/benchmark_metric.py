from tabulate import tabulate
import pickle
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Union
from synthetic_dataset import SyntheticDataset, SyntheticDataEntry


@dataclass
class OnlineRequestOutput:
    entry: SyntheticDataEntry
    success: bool = False
    output_text: str = ""
    start_time: float = 0.
    token_times: list[float] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    request_rate: float = 0
    start_time: float = 0
    end_time: float = 0
    outputs: list[OnlineRequestOutput] = field(default_factory=list)


@dataclass
class MethodResults:
    method_name: str
    results: list[BenchmarkResult] = field(default_factory=list)


@dataclass
class BenchmarkMetrics:
    benchmark_duration: float
    completed: int
    total_input_tokens: int
    total_output_tokens: int
    request_throughput: float
    input_token_throughput: float
    output_token_throughput: float

    mean_input_len: int
    median_input_len: int
    max_input_len: int

    mean_output_len: int
    median_output_len: int
    max_output_len: int

    mean_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float

    mean_ttft_ms: float
    median_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float

    mean_tpot_ms: float
    median_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float

    ttft_slo_attainment: float
    tpot_slo_attainment: float
    slo_attainment: float

    def print(self):
        print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
        print("{:<40} {:<10}".format("Successful requests:", self.completed))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):", self.benchmark_duration))
        print("{:<40} {:<10}".format("Total input tokens:", self.total_input_tokens))
        print("{:<40} {:<10}".format("Total generated tokens:", self.total_output_tokens))
        print("{:<40} {:<10}".format("Mean input len:", self.mean_input_len))
        print("{:<40} {:<10}".format("Median input len:", self.median_input_len))
        print("{:<40} {:<10}".format("Max input len:", self.max_input_len))
        print("{:<40} {:<10}".format("Mean generated tokens:", self.mean_output_len))
        print("{:<40} {:<10}".format("Median generated tokens:", self.median_output_len))
        print("{:<40} {:<10}".format("Max generated tokens:", self.max_output_len))
        print("{s:{c}^{n}}".format(s='Throughput', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):", self.request_throughput))
        print("{:<40} {:<10.2f}".format("Input token throughput (char/s):", self.input_token_throughput))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", self.output_token_throughput))
        print("{s:{c}^{n}}".format(s='Time to Latency', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean Latency (ms):", self.mean_latency_ms))
        print("{:<40} {:<10.2f}".format("Median Latency (ms):", self.median_latency_ms))
        print("{:<40} {:<10.2f}".format("P90 Latency (ms):", self.p90_latency_ms))
        print("{:<40} {:<10.2f}".format("P99 Latency (ms):", self.p99_latency_ms))
        print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", self.mean_ttft_ms))
        print("{:<40} {:<10.2f}".format("Median TTFT (ms):", self.median_ttft_ms))
        print("{:<40} {:<10.2f}".format("P90 TTFT (ms):", self.p90_ttft_ms))
        print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", self.p99_ttft_ms))
        print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", self.mean_tpot_ms))
        print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                        self.median_tpot_ms))
        print("{:<40} {:<10.2f}".format("P90 TPOT (ms):", self.p90_tpot_ms))
        print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", self.p99_tpot_ms))
        print("{s:{c}^{n}}".format(s='SLO Attainment', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("TTFT SLO Attainment:", self.ttft_slo_attainment))
        print("{:<40} {:<10.2f}".format("TPOT SLO Attainment:", self.tpot_slo_attainment))
        print("{:<40} {:<10.2f}".format("SLO Attainment:", self.slo_attainment))
        print("=" * 50)
    
class BenchmarkMetricsBuilder:
    def __init__(self, 
        start_time: float, 
        end_time  : float, 
    ):
        self.total_requests: int = 0
        self.completed = 0
        self.input_lens: list[int] = [] # num of prompt char
        self.output_lens: list[int] = [] # num of output tokens
        self.latencies: list[float] = []
        self.ttfts: list[float] = []
        self.tpots: list[float] = []
        self.start_time = start_time
        self.end_time = end_time
        self.ttft_slo = 0.
        self.tpot_slo = 0.
        self.ttft_slo_cnt: int = 0 # num of requests that satisfied ttft
        self.tpot_slo_cnt: int = 0 # num of requests that satisfied tpot
        self.slo_cnt: int = 0 # num of requests that satisfied both ttft and tpot
        
    def set_ttft_slo(self, ttft_slo: float):
        # You can set the slo for different requests
        self.ttft_slo = ttft_slo

    def set_tpot_slo(self, tpot_slo: float):
        self.tpot_slo = tpot_slo

    def _append(
        self, 
        input_len: int, # number of prompt char
        success: bool, # wheather request is success
        output_len: int, # number of output tokens
        arrival_time: float, # request arrival time
        finished_time: float, # request finished time
        token_times: list[float] # request each token finish time
    ):
        assert self.ttft_slo > 0., 'set ttft_slo first and append request'
        assert self.tpot_slo > 0., 'set tpot_slo first and append request'
        self.total_requests += 1
        self.input_lens.append(input_len)
        if not success:
            return

        self.completed += 1
        self.output_lens.append(output_len)
        self.latencies.append(finished_time - arrival_time)

        is_ttft_satisfied: bool = True
        is_tpot_satisfied: bool = True
        tpot_above_slo_cnt: int = 0
        for i in range(len(token_times)):
            if i == 0:
                ttft = token_times[i] - arrival_time
                self.ttfts.append(ttft) 
                is_ttft_satisfied = ttft < self.ttft_slo
            else:
                tpot = token_times[i] - token_times[i - 1]
                self.tpots.append(tpot)
                tpot_above_slo_cnt += tpot > self.tpot_slo
                is_tpot_satisfied = is_tpot_satisfied and tpot_above_slo_cnt <= 0
        self.ttft_slo_cnt += is_ttft_satisfied
        self.tpot_slo_cnt += is_tpot_satisfied
        self.slo_cnt += is_ttft_satisfied and is_tpot_satisfied 

    def append(self, data: Union[OnlineRequestOutput, list[OnlineRequestOutput]]):
        if isinstance(data, list):
            for output in data:
                self.append(output)
            return
        if isinstance(data, OnlineRequestOutput):
            self._append(
                input_len = len(data.entry.prompt),
                success = data.success, 
                output_len = len(data.token_times), 
                arrival_time = data.start_time, 
                finished_time = data.token_times[-1], 
                token_times = data.token_times, 
            )
            return
        raise Exception(f'invalid data dtype {type(data)}')

    def get_metrics(self) ->  BenchmarkMetrics:
        duration = self.end_time - self.start_time
        metrics = BenchmarkMetrics(
            benchmark_duration=duration, 
            completed=self.completed,
            total_input_tokens=sum(self.input_lens),
            total_output_tokens=sum(self.output_lens),
            mean_input_len=np.mean(self.input_lens),
            median_input_len=np.median(self.input_lens),
            max_input_len=max(self.input_lens),
            mean_output_len=np.mean(self.output_lens),
            median_output_len=np.median(self.output_lens),
            max_output_len=max(self.output_lens),
            request_throughput=self.completed / duration,
            input_token_throughput=sum(self.input_lens) / duration,
            output_token_throughput=sum(self.output_lens) / duration,
            mean_latency_ms=np.mean(self.latencies) * 1000,
            median_latency_ms=np.median(self.latencies) * 1000,
            p90_latency_ms=np.percentile(self.latencies, 90) * 1000,
            p99_latency_ms=np.percentile(self.latencies, 99) * 1000,
            mean_ttft_ms=np.mean(self.ttfts or 0) * 1000,
            median_ttft_ms=np.median(self.ttfts or 0) * 1000,
            p90_ttft_ms=np.percentile(self.ttfts or 0, 90) * 1000,
            p99_ttft_ms=np.percentile(self.ttfts or 0, 99) * 1000,
            mean_tpot_ms=np.mean(self.tpots) * 1000,
            median_tpot_ms=np.median(self.tpots) * 1000,
            p90_tpot_ms=np.percentile(self.tpots, 90) * 1000 if len(self.tpots) > 0 else np.nan,
            p99_tpot_ms=np.percentile(self.tpots, 99) * 1000 if len(self.tpots) > 0 else np.nan,
            ttft_slo_attainment = self.ttft_slo_cnt / self.total_requests,
            tpot_slo_attainment = self.tpot_slo_cnt / self.total_requests,
            slo_attainment = self.slo_cnt / self.total_requests, 
        )
        return metrics


@dataclass
class BenchmarkMetricsAnalysisResult:
    ttft_slo: float
    tpot_slo: float
    methods_results: list[MethodResults] # each method result list, each result in list is coressponding to one request rate
    methods_metrics: list[list[BenchmarkMetrics]] # each method result list, each metric in list is coressponding to one request rate
    methods_score: list[float] # each method score, the more the better, used to find a suitable slo settings

    def print(self):
        print(f'TTFT_SLO: {self.ttft_slo}')
        print(f'TPOT_SLO: {self.tpot_slo}')

        headers = [
            "Method", 
            "Request_Rate(Req/s)", 
            "TTFT_SLO_Attainment", 
            "TPOT_SLO_Attainment", 
            "SLO_Attainment", 
            "Request_Throughput(Req/s)", 
            "Token_Throughput(token/s)", 
            "Avg_Latency(ms)", 
            "Median_Latency(ms)", 
            "P90_Latency(ms)", 
            "P99_Latency(ms)", 
            "Avg_TTFT(ms)", 
            "Median_TTFT(ms)", 
            "P90_TTFT(ms)", 
            "P99_TTFT(ms)", 
            "Avg_TPOT(ms)", 
            "Median_TPOT(ms)", 
            "P90_TPOT(ms)", 
            "P99_TPOT(ms)", 
            ]

        data = []
        for method_metrics, method_results in zip(self.methods_metrics, self.methods_results):
            method_name: str = method_results.method_name
            results_all_rate: list[BenchmarkResult] = method_results.results
            metrics_all_rate: list[BenchmarkMetrics] = method_metrics
            for metrics, results in zip(metrics_all_rate, results_all_rate):
                data.append((
                    method_name, 
                    results.request_rate, 
                    metrics.ttft_slo_attainment, 
                    metrics.tpot_slo_attainment, 
                    metrics.slo_attainment,
                    metrics.request_throughput, 
                    metrics.output_token_throughput, 
                    metrics.mean_latency_ms,
                    metrics.median_latency_ms, 
                    metrics.p90_latency_ms, 
                    metrics.p99_latency_ms, 
                    metrics.mean_ttft_ms, 
                    metrics.median_ttft_ms, 
                    metrics.p90_ttft_ms, 
                    metrics.p99_ttft_ms, 
                    metrics.mean_tpot_ms, 
                    metrics.median_tpot_ms, 
                    metrics.p90_tpot_ms, 
                    metrics.p99_tpot_ms, 
                    ))
        slo_table = tabulate(data, headers, tablefmt="plain")
        print(slo_table)


class BenchmarkMetricsAnalyzer:
    def __init__(
        self, 
        tpot_slo: list[float], # analysis tpot slo
        ttft_slo: list[float], # analysis ttft slo
    ):
        self.tpot_slo_list = tpot_slo
        self.ttft_slo_list = ttft_slo
        # how many request rate each baseline test, all baseline should have same number of request rates
        self.num_request_rate: int = 0
        self.methods_results: list[MethodResults] = []

    def add_method(self, data: Union[str, MethodResults]):
        # 1. read online request output
        if isinstance(data, str):
            path = data
            with open(path, "rb") as f:
                data: MethodResults = pickle.load(f)
        self.methods_results.append(data)
        if self.num_request_rate == 0:
            self.num_request_rate = len(data.results)
        else:
            assert self.num_request_rate == len(data.results), 'all results should have same number of request rate'

    def _analysis_slo_combination(self, ttft_slo: float, tpot_slo: float, method_results: MethodResults) -> list[BenchmarkMetrics]:
        ret: list[BenchmarkMetrics] = []
        for result_each_rate in method_results.results:
            start_time = result_each_rate.start_time
            end_time = result_each_rate.end_time
            request_rate = result_each_rate.request_rate
            outputs = result_each_rate.outputs
            builder = BenchmarkMetricsBuilder(start_time=start_time, end_time=end_time)
            builder.set_ttft_slo(ttft_slo)
            builder.set_tpot_slo(tpot_slo)
            builder.append(outputs)
            metrics = builder.get_metrics()
            ret.append(metrics)
        return ret

    def _compare_metric(self, metrics_list: list[BenchmarkMetrics]) -> float:
        sum: float = 0
        for metrics in metrics_list:
            sum += metrics.slo_attainment
        return sum / len(metrics_list)

    def analysis(self) -> list[BenchmarkMetricsAnalysisResult]:
        """
        data: data path or list[BenchmarkResult]
        """
        analysis_results: list[BenchmarkMetricsAnalysisResult] = []
        # 2. iterate ttft slo list
        for ttft_slo in self.ttft_slo_list:
            # 3. iterate tpot slo list
            for tpot_slo in self.tpot_slo_list:
                # 4. iterate baseline and our dataset
                methods_metrics: list[list[BenchmarkMetrics]] = [] # first dim is methods, second dim is request rate
                methods_score: list[float] = [] 
                for method_results in self.methods_results:
                    # 5. caculate metric
                    metrics_list = self._analysis_slo_combination(ttft_slo, tpot_slo, method_results)
                    methods_metrics.append(metrics_list)
                    # 6. caculate metric score
                    score = self._compare_metric(metrics_list)
                    methods_score.append(score)
                # 7. record score to metric map
                analysis_results.append(BenchmarkMetricsAnalysisResult(
                    ttft_slo = ttft_slo, 
                    tpot_slo = tpot_slo, 
                    methods_results = self.methods_results, 
                    methods_metrics = methods_metrics, 
                    methods_score = methods_score
                ))
        return analysis_results