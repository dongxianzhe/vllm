import os
import time
import random
import asyncio
import argparse
from tabulate import tabulate
import numpy as np
from tqdm import tqdm
from openai import AsyncOpenAI
from typing import AsyncGenerator
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from benchmark_metric import BenchmarkMetrics, BenchmarkMetricsBuilder
from synthetic_dataset import SyntheticDataset, SyntheticDataEntry


@dataclass
class BenchmarkResult:
    metric: BenchmarkMetrics
    output_text: list[str] = field(default_factory=list)


def log_result(args: argparse.Namespace, dataset: SyntheticDataset, results: list[BenchmarkResult]):
    if args.test_correctness:
        for i, result in enumerate(results):
            print(f'==================== correctness test {i} ====================')
            for i, output_text in enumerate(result.output_text):
                print(f'{i}: {output_text}')

        headers = [
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
        for request_rate, result in zip(args.request_rate, results):
            data.append((
                request_rate, 
                result.metric.ttft_slo_attainment, 
                result.metric.tpot_slo_attainment, 
                result.metric.slo_attainment,
                result.metric.request_throughput, 
                result.metric.output_token_throughput, 
                result.metric.mean_latency_ms,
                result.metric.median_latency_ms, 
                result.metric.p90_latency_ms, 
                result.metric.p99_latency_ms, 
                result.metric.mean_ttft_ms, 
                result.metric.median_ttft_ms, 
                result.metric.p90_ttft_ms, 
                result.metric.p99_ttft_ms, 
                result.metric.mean_tpot_ms, 
                result.metric.median_tpot_ms, 
                result.metric.p90_tpot_ms, 
                result.metric.p99_tpot_ms, 
                ))
        slo_table = tabulate(data, headers, tablefmt="plain")
        print(slo_table)


async def poisson_process_request_generator(
    dataset,
    request_rate: float,
) -> AsyncGenerator[tuple[int, SyntheticDataEntry], None]:
    for i, request in enumerate(dataset):
        yield i, request
        if request_rate == float('inf'):
            continue
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


def async_wrapper(func):
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


@dataclass
class OnlineRequestOutput:
    entry: SyntheticDataEntry
    prompt: str = ""
    success: bool = False
    output_text: str = ""
    start_time: float = 0.
    token_times: list[float] = field(default_factory=list)


async def server_proxy(args: argparse.Namespace, entry: SyntheticDataEntry, send_pbar: tqdm, recv_pbar: tqdm, client: AsyncOpenAI) -> OnlineRequestOutput:
    send_pbar.update(1)
    output = OnlineRequestOutput(entry=entry)
    output.start_time = time.perf_counter()
    response = await client.chat.completions.create(
        messages = [{
            "role":"user",
            "content": [
                {
                    "type": "text",
                    "text": entry.prompt, 
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{entry.image}"
                    },
                },
            ],
        }], 
        max_tokens=128, 
        model = args.model,
        temperature=0., 
        stream=True, 
    )
    output.success = True
    async for chunk in response:
        context = chunk.choices[0].delta.content
        output.output_text += context
        output.token_times.append(time.perf_counter())
    output.prompt = entry.prompt
    recv_pbar.update(1)
    return output

async def benchmark(args: argparse.Namespace, dataset: SyntheticDataset, client: AsyncOpenAI, request_rate: float) -> BenchmarkResult:
    send_pbar = tqdm(total = len(dataset), desc='send')
    recv_pbar = tqdm(total = len(dataset), desc='recv')
    metric_builder = BenchmarkMetricsBuilder()

    start = time.perf_counter()
    metric_builder.start()
    tasks = []
    async for (i, entry) in poisson_process_request_generator(dataset=dataset, request_rate=request_rate):
        tasks.append(asyncio.create_task(server_proxy(args, entry, send_pbar=send_pbar, recv_pbar=recv_pbar, client=client)))
    outputs: list[OnlineRequestOutput] = await asyncio.gather(*tasks)

    metric_builder.end()
    recv_pbar.close()
    assert len(outputs) > 0
    for output in outputs:
        metric_builder.append(
            input_len = len(output.entry.prompt), # todo
            success = output.success, 
            output_len = len(output.token_times), 
            arrival_time = output.start_time, 
            finished_time = output.token_times[-1], 
            token_times = output.token_times, 
            ttft_slo = output.entry.ttft_slo,
            tpot_slo = output.entry.tpot_slo,
        )
        
    return BenchmarkResult(
        metric = metric_builder.get_metrics(), 
        output_text = [output.output_text for output in outputs]
    )

@async_wrapper
async def benchmarks(args: argparse.Namespace, dataset: SyntheticDataset) -> list[BenchmarkResult]:
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{args.host}:{args.port}/v1"
    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    results: list[BenchmarkResult] = []
    for request_rate in args.request_rate:
        print(f'start test request rate {request_rate}')
        result = await benchmark(args, dataset, client, request_rate)
        results.append(result)
    return results


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = SyntheticDataset(
        num_requests = args.num_requests, 
        ttft_slo     = args.ttft_slo, 
        tpot_slo     = args.tpot_slo, 
        textcaps     = args.textcaps, 
        pope         = args.pope, 
        mme          = args.mme, 
        text_vqa     = args.text_vqa, 
        vizwiz_vqa   = args.vizwiz_vqa, 
    )                 
    results = benchmarks(args, dataset)
    log_result(args, dataset, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarking script for inference system', conflict_handler='resolve')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(f'--model', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the model.')
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of requests to process.",
    )
    parser.add_argument(
        '--request-rate', 
        type=float, 
        nargs='*', 
        default=[float('inf')], 
        metavar='rate',
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        '--test-correctness',
        action='store_true',
        default=False,
        help='test correctness'
    ) 
    parser.add_argument("--tpot-slo", type=float, default=float(os.environ.get("TPOT_SLO", 0.16)))
    parser.add_argument("--ttft-slo", type=float, default=float(os.environ.get("TTFT_SLO", 2.0)))
    parser.add_argument("--textcaps", type=int, default=int(os.environ.get("TEXTCAPS", 0)))
    parser.add_argument("--pope", type=int, default=int(os.environ.get("POPE", 0)))
    parser.add_argument("--mme", type=int, default=int(os.environ.get("MME", 0)))
    parser.add_argument("--text_vqa", type=int, default=int(os.environ.get("TEXT_VQA", 0)))
    parser.add_argument("--vizwiz_vqa", type=int, default=int(os.environ.get("VIZWIZ_VQA", 0)))
    args, remain_args = parser.parse_known_args()
    print(f'benchmark args {args}')
    main(args)