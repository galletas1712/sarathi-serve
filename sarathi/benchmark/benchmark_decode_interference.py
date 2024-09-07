import math
import pandas as pd
import time
import torch
import random
import threading

import torch.distributed
from torch.profiler import profile, record_function, ProfilerActivity

from sarathi.benchmark.config import BenchmarkConfig
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.sequence import Sequence, SequenceMetadata
from sarathi.engine.llm_engine import BaseLLMEngine
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.worker.cache_engine import CacheEngine
from sarathi.config.config import BaseEndpointConfig, ReplicaConfig, RollingPreemptionProfilingSchedulerConfig
from sarathi.model_executor.attention import get_attention_wrapper, set_attention_backend
from sarathi.model_executor.parallel_utils.parallel_state import initialize_model_parallel
from dataclasses import dataclass

from sarathi.model_executor.model_runner import ModelRunner

# CACHE_SIZE_PER_TOKEN = 131072 # Determined by the model
# CHUNK_SIZE_LOG_MIN = 8 # Arbitrary
# TOKEN_SIZE_LOG_MIN = 8 # Arbitrary, but should be at least chunk size min
# TOKEN_SIZE_LOG_MAX = 17  # Determined by number of GPU blocks (~ GPU HBM size).
# MAX_MODEL_TOKENS = 65536 # Should have been 131072 but we truncate to 65536 otherwise it throws a CUDA error
NUM_WARMUP_PASSES = 4
NUM_PASSES = 10

benchmark_config = BenchmarkConfig.create_from_cli_args()
replica_config = ReplicaConfig(
    0,
    benchmark_config.output_dir,
    [('node:172.19.128.82', 0)]
)
import os
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)
initialize_model_parallel()

system_config = BaseEndpointConfig().create_system_config(replica_config)
system_config.cache_config.num_gpu_blocks = 21000 # Some random high number from previous profiles
system_config.cache_config.num_cpu_blocks = 4096
set_attention_backend("flashinfer")
MetricsStore.get_or_create_instance(
    system_config.replica_config,
    system_config.model_config,
    system_config.metrics_config,
)
model_runner = ModelRunner(config=system_config, device=torch.device('cuda'), rank=0)
model_runner.sampler = None
cache_engine = CacheEngine(config=system_config)

vocab_size = model_runner.model.config.vocab_size

@dataclass
class BenchmarkDim:
    batch_size: int
    max_num_batched_tokens: int
    num_decode_iters: int
    num_blocks_swap_out: int
    num_blocks_swap_in: int


benchmark_dims = []

for max_num_batched_tokens in [262144]:
    for batch_size in [8]:
        for num_decode_iters in [4, 16, 64]:
            for num_blocks_swap_out in [2048, 1024, 512, 256, 128, 64]:
                benchmark_dims.append(
                    BenchmarkDim(
                        batch_size=batch_size,
                        max_num_batched_tokens=max_num_batched_tokens,
                        num_decode_iters=num_decode_iters,
                        num_blocks_swap_out=num_blocks_swap_out,
                        num_blocks_swap_in=0
                    )
                )
            for num_blocks_swap_in in [2048, 1024, 512, 256, 128, 64]:
                benchmark_dims.append(
                    BenchmarkDim(
                        batch_size=batch_size,
                        max_num_batched_tokens=max_num_batched_tokens,
                        num_decode_iters=num_decode_iters,
                        num_blocks_swap_out=0,
                        num_blocks_swap_in=num_blocks_swap_in
                    )
                )
            for num_blocks in [1024, 512, 256, 128, 64]:
                    benchmark_dims.append(
                        BenchmarkDim(
                            batch_size=batch_size,
                            max_num_batched_tokens=max_num_batched_tokens,
                            num_decode_iters=num_decode_iters,
                            num_blocks_swap_out=num_blocks,
                            num_blocks_swap_in=num_blocks
                        )
                    )
            benchmark_dims.append(
                BenchmarkDim(
                    batch_size=batch_size,
                    max_num_batched_tokens=max_num_batched_tokens,
                    num_decode_iters=num_decode_iters,
                    num_blocks_swap_out=0,
                    num_blocks_swap_in=0
                )
            )

            
dfs = []
for benchmark_dim in benchmark_dims:
    print("Profiling with benchmark dim:", benchmark_dim)
    stem = f"swap_{benchmark_dim.batch_size}_{benchmark_dim.max_num_batched_tokens}_{benchmark_dim.num_decode_iters}_{benchmark_dim.num_blocks_swap_out}_{benchmark_dim.num_blocks_swap_in}"
    csv_file = f"interference_results/pandas/{stem}.csv"
    trace_file = f"interference_results/trace/{stem}.json"
    # if os.path.exists(csv_file) and os.path.exists(trace_file):
    #     dfs.append(pd.read_csv(csv_file, index_col=False))
    #     print(f"Skipping {stem}")
    #     continue

    assert benchmark_dim.max_num_batched_tokens % benchmark_dim.batch_size == 0
    seq_len = benchmark_dim.max_num_batched_tokens // benchmark_dim.batch_size
    num_blocks_per_seq = (seq_len + cache_engine.block_size - 1) // cache_engine.block_size

    seq_metadata_list = []
    for seq_id in range(benchmark_dim.batch_size):
        seq = Sequence(
            seq_id=str(seq_id),
            prompt=None,
            prompt_token_ids=random.choices(range(10, 100), k=seq_len),
            block_size=cache_engine.block_size,
            eos_token_id=1,
            arrival_time=None,
            sampling_params=None,
        )

        block_table_start = seq_id * num_blocks_per_seq
        block_table = list(range(block_table_start, block_table_start + num_blocks_per_seq))

        seq_metadata = SequenceMetadata(
            seq=seq,
            block_table=block_table,
            prompt_chunk_len=0,
        )
        seq_metadata_list.append(seq_metadata)
    
    num_blocks_active = num_blocks_per_seq * benchmark_dim.batch_size
    assert num_blocks_active + benchmark_dim.num_blocks_swap_out + benchmark_dim.num_blocks_swap_in <= cache_engine.num_gpu_blocks
    assert benchmark_dim.num_blocks_swap_out + benchmark_dim.num_blocks_swap_in <= cache_engine.num_cpu_blocks

    swap_out_gpu_region = (
        num_blocks_active + benchmark_dim.num_blocks_swap_out,
        num_blocks_active + benchmark_dim.num_blocks_swap_out * 2
    )

    swap_in_gpu_region = (
        num_blocks_active + benchmark_dim.num_blocks_swap_out * 2 + benchmark_dim.num_blocks_swap_in,
        num_blocks_active + benchmark_dim.num_blocks_swap_out * 2 + benchmark_dim.num_blocks_swap_in * 2)

    swap_out_cpu_region = (0, benchmark_dim.num_blocks_swap_out)

    swap_in_cpu_region = (
        benchmark_dim.num_blocks_swap_out * 2,
        benchmark_dim.num_blocks_swap_out * 2 + benchmark_dim.num_blocks_swap_in
    )
    # swap_out_mapping = [
    #     (num_blocks_active + i, i) for i in range(benchmark_dim.num_blocks_swap_out)
    # ]
    # swap_in_mapping = [
    #     (benchmark_dim.num_blocks_swap_out, num_blocks_active + benchmark_dim.num_blocks_swap_out + i) for i in range(benchmark_dim.num_blocks_swap_in)
    # ]
    print("Number of active blocks:", num_blocks_active)
    print("Number of blocks to swap out:", benchmark_dim.num_blocks_swap_out)
    print("Number of blocks to swap in:", benchmark_dim.num_blocks_swap_in)
    print("Swap out GPU region:", swap_out_gpu_region)
    print("Swap in GPU region:", swap_in_gpu_region)
    print("Swap out CPU region:", swap_out_cpu_region)
    print("Swap in CPU region:", swap_in_cpu_region)
    print("Min GPU block in sequence:", [min(seq_metadata.block_table) for seq_metadata in seq_metadata_list])
    print("Max GPU block in sequence:", [max(seq_metadata.block_table) for seq_metadata in seq_metadata_list])
    print("Number of unique GPU blocks in sequence:", [len(set(seq_metadata.block_table)) for seq_metadata in seq_metadata_list])

    swap_out_stream = torch.cuda.Stream()
    swap_in_stream = torch.cuda.Stream()
    
    pass_latencies = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.cuda.cudart().cudaProfilerStart()

        swap_out_times = []
        swap_in_times = []
        for pass_num in range(NUM_WARMUP_PASSES + NUM_PASSES):
            start_time = time.perf_counter_ns()
            torch.cuda.nvtx.range_push(f"Pass {pass_num - NUM_WARMUP_PASSES}")

            with record_function(f"Pass {pass_num - NUM_WARMUP_PASSES}"):
                # NOTE: This isn't really realistic, but since we aren't adding any tokens it should be fine
                with record_function("Begin forward"):
                    get_attention_wrapper().begin_forward(seq_metadata_list)
                
                swap_out_start_event = torch.cuda.Event(enable_timing=True)
                swap_out_end_event = torch.cuda.Event(enable_timing=True)
                swap_in_start_event = torch.cuda.Event(enable_timing=True)
                swap_in_end_event = torch.cuda.Event(enable_timing=True)

                for i in range(benchmark_dim.num_decode_iters):
                    with record_function(f"Decode Iter {i}"):
                        with record_function("Prepare inputs"):
                            input_tokens, input_positions = model_runner._prepare_inputs(seq_metadata_list)

                        if i == 0:
                            if benchmark_dim.num_blocks_swap_out > 0:
                                with torch.cuda.stream(swap_out_stream):
                                    swap_out_start_event.record()
                                    for layer in range(cache_engine.num_layers):
                                        cache_engine.cpu_cache[layer][swap_out_cpu_region[0]:swap_out_cpu_region[1]].copy_(
                                            cache_engine.gpu_cache[layer][swap_out_gpu_region[0]:swap_out_gpu_region[1]],
                                            non_blocking=True
                                        )
                                    swap_out_end_event.record()
                            
                            if benchmark_dim.num_blocks_swap_in > 0:
                                with torch.cuda.stream(swap_in_stream):
                                    swap_in_start_event.record()
                                    for layer in range(cache_engine.num_layers):
                                        cache_engine.gpu_cache[layer][swap_in_gpu_region[0]:swap_in_gpu_region[1]].copy_(
                                            cache_engine.cpu_cache[layer][swap_in_cpu_region[0]:swap_in_cpu_region[1]],
                                            non_blocking=True
                                        )
                                    swap_in_end_event.record()

                        torch.cuda.nvtx.range_push(f"Decode Iter {i}")
                        model_runner.model.forward(
                            hidden_states=input_tokens,
                            positions=input_positions,
                            kv_caches=cache_engine.gpu_cache
                        )
                        torch.cuda.nvtx.range_pop()
                
                with record_function("End forward"):
                    get_attention_wrapper().end_forward()

                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                end_time = time.perf_counter_ns()
                pass_latency = (end_time - start_time) / 1e6
                if pass_num >= NUM_WARMUP_PASSES:
                    pass_latencies.append(pass_latency)
                print(f"Pass {pass_num - NUM_WARMUP_PASSES} latency: {pass_latency} ms")

                if benchmark_dim.num_blocks_swap_out > 0:
                    swap_out_times.append(swap_out_start_event.elapsed_time(swap_out_end_event))

                if benchmark_dim.num_blocks_swap_in > 0:
                    swap_in_times.append(swap_in_start_event.elapsed_time(swap_in_end_event))

        torch.cuda.cudart().cudaProfilerStop()
    
    df = pd.DataFrame({
        "batch_size": [benchmark_dim.batch_size],
        "max_num_batched_tokens": [benchmark_dim.max_num_batched_tokens],
        "num_decode_iters": [benchmark_dim.num_decode_iters],
        "num_blocks_swap_out": [benchmark_dim.num_blocks_swap_out],
        "num_blocks_swap_in": [benchmark_dim.num_blocks_swap_in],
        "total_latency": [sum(pass_latencies) / NUM_PASSES],
        "decode_latency": [sum(pass_latencies) / NUM_PASSES / benchmark_dim.num_decode_iters],
        "average_swap_out_latency": [sum(swap_out_times) / len(swap_out_times) if len(swap_out_times) > 0 else None],
        "average_swap_in_latency": [sum(swap_in_times) / len(swap_in_times) if len(swap_in_times) > 0 else None],
    })
    dfs.append(df)
    df.to_csv(csv_file, index=False)
    prof.export_chrome_trace(trace_file)

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv("interference_results/interference_results.csv", index=False)