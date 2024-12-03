"""A GPU worker class."""

import os
import time
from pathlib import Path
from threading import Event, Thread
from typing import Optional, Tuple

import torch
import torch.distributed
import zmq

from sarathi.config import (
    CacheConfig,
    DisaggEmulationSchedulerConfig,
    ParallelConfig,
    SystemConfig,
)
from sarathi.core.block_space_manager import BlockDevice
from sarathi.core.datatypes.comm_info import CommInfo
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs
from sarathi.core.sequence_manager.worker_sequence_manager import WorkerSequenceManager
from sarathi.logger import init_logger
from sarathi.metrics.alt_metrics_store import WorkerMetricsStore
from sarathi.model_executor.attention import (
    get_attention_wrapper,
    set_attention_backend,
)
from sarathi.model_executor.model_runner import ModelRunner
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    initialize_model_parallel,
)
from sarathi.model_executor.utils import set_random_seed
from sarathi.utils.threading_utils import exit_on_error, synchronized

logger = init_logger(__name__)


_READY_ACK_WAIT_TIME = 1


class BaseWorker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        config: SystemConfig,
        local_rank: int,
        rank: int,
        comm_info: CommInfo,
    ) -> None:
        # Not: the cache config is partially initialized at this point, ie. it doesn't have
        # information about the number of blocks, it will get updated after profiling
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.comm_info = comm_info

        # Sequence manager also needs number of blocks for initialization
        self.seq_manager = None

        set_attention_backend(config.worker_config.attention_backend)

        self._verify_parallel_config()
        self.metrics_store = WorkerMetricsStore(
            isinstance(config.scheduler_config, DisaggEmulationSchedulerConfig)
        )

        self._init_zmq_sockets()

        self.worker_ready_event = Event()
        self.execution_thread = Thread(target=self._execution_loop, daemon=True)

        self.curr_batch_id = 0

    def _init_zmq_sockets(self):
        self.zmq_context = zmq.Context()
        self.enqueue_socket = self.zmq_context.socket(zmq.SUB)
        self.enqueue_socket.connect(
            f"tcp://{self.comm_info.engine_ip_address}:{self.comm_info.enqueue_socket_port}"
        )
        self.enqueue_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.notify_socket = self.zmq_context.socket(zmq.PUSH)
        self.notify_socket.connect(
            f"tcp://{self.comm_info.engine_ip_address}:{self.comm_info.notify_socket_port}"
        )
        self.output_socket = self.zmq_context.socket(zmq.PUSH)
        self.output_socket.connect(
            f"tcp://{self.comm_info.engine_ip_address}:{self.comm_info.output_socket_port}"
        )

    def _verify_parallel_config(self) -> None:
        assert self.config.parallel_config.pipeline_parallel_size == 1

    @torch.inference_mode()
    @synchronized
    def init_model(self):
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ["KINETO_LOG_LEVEL"] = "5"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

        logger.info(f"Worker {self.rank} is using device {self.local_rank}")
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # Initialize the distributed environment.
        _init_distributed_environment(
            self.config.parallel_config,
            self.rank,
            self.comm_info.distributed_init_method,
        )

        self.tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        self.pipeline_model_parallel_rank = get_pipeline_model_parallel_rank()

        self.is_tensor_parallel_rank_zero = self.tensor_model_parallel_rank == 0
        self.is_first_pipeline_stage = self.pipeline_model_parallel_rank == 0
        self.is_last_pipeline_stage = (
            self.pipeline_model_parallel_rank
            == self.config.parallel_config.pipeline_parallel_size - 1
        )

        logger.info(
            f"Initializing worker {self.rank} on device {self.device}, "
            f"tensor parallel rank {self.tensor_model_parallel_rank} "
            f"and pipeline parallel rank {self.pipeline_model_parallel_rank}."
        )

        # Initialize the model.
        set_random_seed(self.config.model_config.seed)
        self.model_runner = ModelRunner(
            self.config,
            self.device,
            self.rank,
        )
        logger.info(f"Model initialized on worker {self.rank}.")

    @torch.inference_mode()
    @synchronized
    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        torch.cuda.set_device(self.device)

        self.config.cache_config = cache_config

        print("Called init cache engine")
        get_attention_wrapper().init_cache_engine(cache_config)

        self.seq_manager = WorkerSequenceManager(
            self.config,
        )

        self.execution_thread.start()

    def wait_till_ready(self) -> None:
        self.worker_ready_event.wait()
        time.sleep(_READY_ACK_WAIT_TIME)

    @synchronized
    def get_model_parallel_ranks(self) -> Tuple[int, int]:
        return self.tensor_model_parallel_rank, self.pipeline_model_parallel_rank

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
    ) -> Optional[SamplerOutputs]:
        # NOTE: not synchronizing at the top because we already did at the bottom
        self.metrics_store.on_batch_start(batch_id=self.curr_batch_id)

        # on_schedule will set up block tables, but not extract them
        self.seq_manager.on_schedule(scheduler_outputs)

        # This will actually extract the block tables
        seq_exec_metadata_list = self.seq_manager.get_seq_exec_metadata_list(
            scheduler_outputs
        )

        self.metrics_store.on_batch_scheduled(
            batch_id=self.curr_batch_id, seq_exec_metadata_list=seq_exec_metadata_list
        )

        # NOTE: Ordering of which ones are swapped out first
        # Perform sync swap out
        # This will wait for swap outs to finish
        if not self.config.cache_config.duplicate_kv_cache:
            # NOTE: We don't actually perform the cache swap out operation, since it's already all stored in host memory
            swap_out_mappings = self.seq_manager.get_swap_out_mappings(
                scheduler_outputs.swap_out_seq_ids
            )
            now = time.perf_counter()
            for i, (seq_id, mapping) in enumerate(swap_out_mappings.items()):
                assert (
                    not scheduler_outputs.swap_out_lens
                    or len(mapping) == scheduler_outputs.swap_out_lens[i]
                )
                self.metrics_store.on_swap_out_start(
                    seq_id, len(mapping), start_timestamp=now
                )
            get_attention_wrapper().cache_engine.swap_out(swap_out_mappings)
        else:
            # ???????????????????????????????
            now = time.perf_counter()
            for i, seq_id in enumerate(scheduler_outputs.swap_out_seq_ids):
                if scheduler_outputs.swap_out_lens:
                    num_blocks = scheduler_outputs.swap_out_lens[i]
                else:
                    num_blocks = (
                        self.seq_manager.block_manager.get_seq_num_blocks_allocated(
                            seq_id, BlockDevice.GPU
                        )
                    )
                self.metrics_store.on_swap_out_start(
                    seq_id, num_blocks, start_timestamp=now
                )

        # Perform async swap in after sync swap out
        swap_in_mappings = self.seq_manager.get_swap_in_mappings(
            scheduler_outputs.swap_in_seq_ids
        )
        now = time.perf_counter()
        for seq_id in swap_in_mappings.keys():
            self.metrics_store.on_swap_in_start(seq_id, start_timestamp=now)

        if self.config.cache_config.async_swap_in:
            get_attention_wrapper().cache_engine.begin_swap_in(swap_in_mappings)
        else:
            get_attention_wrapper().cache_engine.swap_in(swap_in_mappings)
            now = time.perf_counter()
            for seq_id in swap_in_mappings.keys():
                self.metrics_store.on_swap_in_end(seq_id, end_timestamp=now)

        if seq_exec_metadata_list:
            assert not scheduler_outputs.is_empty()  # Superset
            # print(f"Iteration: {self.curr_batch_id}, executing model!")
            # NOTE: Under KV cache duplication, model_runner is responsible for pipelining KV cache out to swap
            scheduled_seqs = [
                seq_exec_metadata.seq for seq_exec_metadata in seq_exec_metadata_list
            ]
            seq_num_tokens = [
                seq_exec_metadata.num_tokens
                for seq_exec_metadata in seq_exec_metadata_list
            ]
            sampler_outputs = self.model_runner.run(
                seq_exec_metadata_list,
                duplicate_mapping=(
                    self.seq_manager.block_manager.get_duplicate_mapping(
                        scheduled_seqs, seq_num_tokens
                    )
                    if self.config.cache_config.duplicate_kv_cache
                    else None
                ),
            )
            # print(f"Iteration: {self.curr_batch_id}, model executed!")
        else:
            # print(f"Iteration: {self.curr_batch_id}, no seq_exec_metadata_list!")
            sampler_outputs = []

        finished_seq_ids = self.seq_manager.on_step_completed(
            scheduler_outputs, sampler_outputs
        )

        # print(f"Iteration: {self.curr_batch_id}, before synchronize @ end of iteration")
        torch.cuda.current_stream().synchronize()
        self.metrics_store.on_batch_end(
            batch_id=self.curr_batch_id, finished_seq_ids=finished_seq_ids
        )
        if self.config.cache_config.async_swap_in:
            now = time.perf_counter()
            for seq_id in swap_in_mappings.keys():
                self.metrics_store.on_swap_in_end(seq_id, end_timestamp=now)

        self.curr_batch_id += 1

        return sampler_outputs

    @exit_on_error
    def _execution_loop(self) -> None:
        torch.cuda.set_device(self.device)

        self.worker_ready_event.set()

        while True:
            logger.debug(f"Iteration: {self.curr_batch_id}")

            # NOTE: This is for debugging purposes only to block the scheduler for easier reading. We'll remove this soon.
            self.notify_socket.send_pyobj(True)

            start_engine_scheduler = time.perf_counter()
            step_inputs = self.enqueue_socket.recv_pyobj()
            end_engine_scheduler = time.perf_counter()
            self.metrics_store.add_engine_scheduler_latency(
                end_engine_scheduler - start_engine_scheduler
            )

            if step_inputs is None:
                continue

            for new_seq in step_inputs.new_seqs:
                self.metrics_store.request_arrived(new_seq.seq_id, new_seq.arrival_time)
                self.seq_manager.add_seq(new_seq)

            output = self.execute_model(step_inputs.scheduler_outputs)

            if not self.is_tensor_parallel_rank_zero:
                continue

            self.output_socket.send_pyobj(output)

    @synchronized
    def get_metrics_store(self) -> WorkerMetricsStore:
        return self.metrics_store

    @synchronized
    def mark_initial_memory_profiling_done(self):
        self.metrics_store.mark_initial_memory_profiling_done()

    @synchronized
    def reset_metrics(self) -> None:
        self.metrics_store.reset()

    @synchronized
    def start_profiling(self) -> None:
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()

    @synchronized
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
    ) -> Tuple[int, int]:
        return self.model_runner.profile_num_available_blocks(
            block_size, gpu_memory_utilization
        )

    @synchronized
    def stop_profiling(self) -> None:
        self.profiler.__exit__(None, None, None)
        trace_path = Path(
            f"{self.config.replica_config.output_dir}/profiler_trace_rank_{self.rank}.json"
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.profiler.export_chrome_trace(str(trace_path))


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: str,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size})."
            )
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(
        parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size
    )
