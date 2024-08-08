import math
import pandas as pd
import time
import torch

from sarathi.benchmark.config import BenchmarkConfig
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.engine.llm_engine import BaseLLMEngine
from sarathi.config.config import BaseEndpointConfig, ReplicaConfig, RollingPreemptionProfilingSchedulerConfig
from dataclasses import dataclass

CACHE_SIZE_PER_TOKEN = 131072 # Determined by the model
TOKEN_SIZE_LOG_MAX = 17  # Determined by number of GPU blocks (~ GPU HBM size)
MAX_MODEL_TOKENS = 8192
NUM_PASSES = 4

chunk_size_logs = [11, 10, 9, 8, 7, 6]

benchmark_config = BenchmarkConfig.create_from_cli_args()
replica_config = ReplicaConfig(
    0,
    benchmark_config.output_dir,
    [('node:172.19.128.82', 0)]
)

@dataclass
class BenchmarkDim:
    max_seq_len: int
    batch_size: int
    chunk_size: int

benchmark_dims = []

for chunk_size_log in chunk_size_logs:
    assert chunk_size_log <= TOKEN_SIZE_LOG_MAX

    chunk_size = int(2 ** chunk_size_log)

    token_count_logs = torch.linspace(start=chunk_size_log, end=TOKEN_SIZE_LOG_MAX, steps=TOKEN_SIZE_LOG_MAX-chunk_size_log+1, dtype=int).tolist()
    token_count_logs = reversed(token_count_logs)

    for token_count_log in token_count_logs:
        token_count = 2 ** token_count_log

        batch_size_log_lo = max(token_count_log - int(math.log2(MAX_MODEL_TOKENS)), 0)
        batch_size_log_hi = token_count_log - chunk_size_log
        batch_sizes = torch.logspace(start=batch_size_log_lo, end=batch_size_log_hi, steps=batch_size_log_hi-batch_size_log_lo+1, base=2, dtype=int).tolist()

        for batch_size in batch_sizes:
            max_seq_len = token_count // batch_size
            benchmark_dims.append(BenchmarkDim(max_seq_len, batch_size, chunk_size))

for benchmark_dim in benchmark_dims:
    print(benchmark_dim)


benchmark_results = []
for benchmark_dim in benchmark_dims:
    print(f"Benchmarking: {benchmark_dim}")

    assert benchmark_dim.max_seq_len % benchmark_dim.chunk_size == 0
    num_chunked_prefill_iters = benchmark_dim.max_seq_len // benchmark_dim.chunk_size

    scheduler_config = RollingPreemptionProfilingSchedulerConfig(
        max_num_seqs=benchmark_dim.batch_size,
        chunk_size=benchmark_dim.chunk_size,
        max_num_batched_tokens=benchmark_dim.max_seq_len * benchmark_dim.batch_size,
    )

    # NOTE: max_model_len doesn't have an effect on prefill time (DUH, since we're using PAGED attention haha)
    system_config = BaseEndpointConfig(
        scheduler_config=scheduler_config,
    ).create_system_config(replica_config)
    engine = BaseLLMEngine(system_config)

    print(f"Creating {2 * benchmark_dim.batch_size} sequences of length {benchmark_dim.max_seq_len}...")
    for _ in range(2 * benchmark_dim.batch_size):
        sampling_params = SamplingParams(temperature=0, max_tokens=benchmark_dim.max_seq_len)
        engine.add_request(None, sampling_params, prompt_token_ids=list(range(benchmark_dim.max_seq_len)))

    # Warmup
    for _ in range(num_chunked_prefill_iters):
        engine.step()

    latencies = []
    start_all = time.perf_counter_ns()
    start = time.perf_counter_ns()
    # engine.start_profiling()
    for i in range(NUM_PASSES *  num_chunked_prefill_iters):
        outputs = engine.step()
        # print("State:")
        # for seq in engine.seq_manager.seq_map.values():
        #     print(seq.seq_id, seq.prompt_tokens_processed)

        if i % num_chunked_prefill_iters == num_chunked_prefill_iters - 1:
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1e6)
            print(f"Recomputation of whole batch took: {latencies[-1]} ms")
            start = time.perf_counter_ns()
    end_all = time.perf_counter_ns()
    mean_latency_all_div = (end_all - start_all) / (1e6 * NUM_PASSES)

    # engine.pull_worker_metrics()
    # engine.plot_metrics()
    # engine.stop_profiling()
    
    mean_latency = torch.mean(torch.tensor(latencies)).item()
    std_latency = torch.std(torch.tensor(latencies)).item()
    print(f"Mean latency: {mean_latency} ms, std latency: {std_latency} ms, mean latency (all divided by num passes): {mean_latency_all_div}")

    engine.terminate()

    benchmark_results.append({
        'chunk_size': benchmark_dim.chunk_size,
        'token_count': benchmark_dim.max_seq_len * benchmark_dim.batch_size,
        'batch_size': benchmark_dim.batch_size,
        'max_seq_len': benchmark_dim.max_seq_len,
        'mean_latency': mean_latency,
        'std_latency': std_latency,
        'mean_latency_all_div': mean_latency_all_div,
        'kv_cache_size': CACHE_SIZE_PER_TOKEN * benchmark_dim.max_seq_len * benchmark_dim.batch_size
    })
    
df = pd.DataFrame(benchmark_results)
df.to_csv("prefill_latency_profiling.csv", index=False)