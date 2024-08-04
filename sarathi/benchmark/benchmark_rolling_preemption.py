from sarathi.benchmark.config import BenchmarkConfig
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.engine.llm_engine import BaseLLMEngine
from sarathi.config.config import BaseEndpointConfig, ReplicaConfig, RollingPreemptionProfilingSchedulerConfig

import time


benchmark_config = BenchmarkConfig.create_from_cli_args()
replica_config = ReplicaConfig(
    0,
    benchmark_config.output_dir,
    [('node:172.19.128.82', 0)]
)

batch_size=8
max_seq_len=2048
chunk_size=512

scheduler_config = RollingPreemptionProfilingSchedulerConfig(max_num_seqs=batch_size, chunk_size=chunk_size)

# NOTE: max_model_len doesn't have an effect on prefilltime (DUH, since we're using PAGED attention haha)
system_config = BaseEndpointConfig(
    scheduler_config=scheduler_config,
).create_system_config(replica_config)
engine = BaseLLMEngine(system_config)

for _ in range(2 * batch_size):
    sampling_params = SamplingParams(temperature=0, max_tokens=max_seq_len)
    engine.add_request(None, sampling_params, prompt_token_ids=list(range(max_seq_len)))

engine.start_profiling()
for _ in range(100):

    start = time.perf_counter_ns()
    outputs = engine.step()
    end = time.perf_counter_ns()
    print(f"Step took: {(end - start) / 1e6} ms")

    print("State:")
    for seq in engine.seq_manager.seq_map.values():
        print(seq.seq_id, seq.prompt_tokens_processed)

engine.pull_worker_metrics()
engine.plot_metrics()
engine.stop_profiling()