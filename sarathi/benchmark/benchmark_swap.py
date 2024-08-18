import gc
import pandas as pd
import time
import torch
import random

from sarathi.benchmark.config import BenchmarkConfig
from sarathi.config.config import BaseEndpointConfig, ReplicaConfig
from sarathi.worker.cache_engine import CacheEngine
from sarathi.model_executor.attention import set_attention_backend, get_attention_wrapper


CACHE_BLOCK_SIZE = 2097152
NUM_PASSES = 4


benchmark_config = BenchmarkConfig.create_from_cli_args()
replica_config = ReplicaConfig(
    0,
    benchmark_config.output_dir,
    [('node:172.19.128.82', 0)]
)
system_config = BaseEndpointConfig().create_system_config(replica_config)


benchmark_results = []
for number_of_swap_blocks in [64, 32, 16, 8]:
    system_config.cache_config.num_gpu_blocks = int(number_of_swap_blocks * 1.1)
    system_config.cache_config.num_cpu_blocks = int(number_of_swap_blocks * 1.1)
    print(system_config)

    set_attention_backend("flashinfer")
    get_attention_wrapper().init(system_config.model_config, system_config.parallel_config, system_config.cache_config.block_size, device="cuda")

    cache_engine = CacheEngine(system_config)

    def swap_out():
        gpu_blocks_to_swap_out = torch.tensor(random.sample(range(0, cache_engine.num_gpu_blocks), number_of_swap_blocks), device="cpu").view(-1, 1)
        cpu_blocks_to_swap_in = torch.tensor(random.sample(range(0, cache_engine.num_cpu_blocks), number_of_swap_blocks), device="cpu").view(-1, 1)
        block_mapping = torch.hstack([gpu_blocks_to_swap_out, cpu_blocks_to_swap_in])

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        cache_engine.swap_out(block_mapping)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)

    def swap_in():
        gpu_blocks_to_swap_in = torch.tensor(random.sample(range(0, cache_engine.num_gpu_blocks), number_of_swap_blocks), device="cpu").view(-1, 1)
        cpu_blocks_to_swap_out = torch.tensor(random.sample(range(0, cache_engine.num_cpu_blocks), number_of_swap_blocks), device="cpu").view(-1, 1)
        block_mapping = torch.hstack([cpu_blocks_to_swap_out, gpu_blocks_to_swap_in])

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        cache_engine.swap_in(block_mapping)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)
    
    # Warmup
    for _ in range(4):
        swap_out()
        swap_in()
    
    swap_out_latencies = []
    swap_in_latenices = []
    total_latencies = []
    for _ in range(4):
        swap_out_time = swap_out()
        swap_out_latencies.append(swap_out_time)
        print(f"Swap out {number_of_swap_blocks} took: {swap_out_time} ms")
        swap_in_time = swap_in()
        swap_in_latenices.append(swap_in_time)
        print(f"Swap in {number_of_swap_blocks} took: {swap_in_time} ms")
        print(f"Total swap {number_of_swap_blocks} took: {swap_out_time + swap_in_time} ms")
        total_latencies.append(swap_out_time + swap_in_time)
        
    del cache_engine.gpu_cache
    del cache_engine.cpu_cache
    gc.collect()
    torch.cuda.empty_cache()

    benchmark_results.append({
        'number_of_swap_blocks': number_of_swap_blocks,
        'kv_cache_size': CACHE_BLOCK_SIZE * number_of_swap_blocks,
        'mean_swap_out_latency': sum(swap_out_latencies) / len(swap_out_latencies),
        'mean_swap_in_latency': sum(swap_in_latenices) / len(swap_in_latenices),
        'mean_round_trip_latency': sum(total_latencies) / len(total_latencies)
    })

#     benchmark_results.append({
#         'chunk_size': benchmark_dim.chunk_size,
#         'token_count': benchmark_dim.max_seq_len * benchmark_dim.batch_size,
#         'batch_size': benchmark_dim.batch_size,
#         'max_seq_len': benchmark_dim.max_seq_len,
#         'mean_latency': mean_latency,
#         'std_latency': std_latency,
#         'mean_latency_all_div': mean_latency_all_div,
#         'kv_cache_size': CACHE_SIZE_PER_TOKEN * benchmark_dim.max_seq_len * benchmark_dim.batch_size
#     })
    
df = pd.DataFrame(benchmark_results)
df.to_csv("swap_profiling.csv", index=False)
