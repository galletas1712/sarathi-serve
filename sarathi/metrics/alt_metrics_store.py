from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
from collections import deque

import numpy as np

from sarathi.core.datatypes.sequence import SequenceExecutionMetadata


def calculate_percentile_values(data: List[float], percentiles: List[float] = [50, 90, 95, 99, 99.9, 100]) -> Dict[float, float]:
    """Calculate multiple percentiles from list of values."""
    if len(data) == 0:
        return {}
    return {p: float(np.percentile(data, p)) if p < 100 else max(data) for p in percentiles}


@dataclass
class SwapInterval:
    """
    start_batch_id: the batch id that started the swap (before the model runs, after the batch is scheduled)
    end_batch_id: the batch id that ended the swap (at the beginning of the batch)
    start_timestamp: timestamp in worker loop when the swap started
    end_timestamp: timestamp in worker right before we notify the engine that the swap has completed
    """
    start_swap_out_batch_id: int
    start_swap_out_timestamp: float
    start_swap_in_batch_id: Optional[int] = None
    start_swap_in_timestamp: Optional[float] = None
    finish_swap_in_batch_id: Optional[int] = None
    finish_swap_in_timestamp: Optional[float] = None
    swap_out_lens: List[int] = field(default_factory=list)
    timestamp_offset: int = 0
    batch_offset: int = 0

    def get_total_duration(self) -> float:
        """Return total swap duration if swap is complete."""
        assert (self.finish_swap_in_timestamp is not None and 
            self.start_swap_out_timestamp is not None)

        # NOTE: Calculating with offset
        return self.finish_swap_in_timestamp - self.start_swap_out_timestamp - self.timestamp_offset
    
    def get_batch_duration(self) -> int:
        """Return number of batches this swap spans if complete."""
        assert (self.finish_swap_in_batch_id is not None and 
            self.start_swap_out_batch_id is not None)

        return self.finish_swap_in_batch_id - self.start_swap_out_batch_id - self.batch_offset


@dataclass
class BatchMetrics:
    """Metrics for a singular batch of requests."""
    batch_id: int
    start_timestamp: float
    scheduled_timestamp: Optional[float]
    end_timestamp: Optional[float]
    prefill_kv_cache_tokens: Optional[int]
    prefill_batched_tokens: Optional[int]
    decode_kv_cache_tokens: Optional[int]
    decode_batched_tokens: Optional[int]
    num_requests: Optional[int]

    def __init__(self, batch_id: int, start_timestamp: float):
        self.batch_id = batch_id
        self.start_timestamp = start_timestamp
    
    def schedule(self,
                 scheduled_timestamp: float,
                 prefill_kv_cache_tokens: int,
                 prefill_batched_tokens: int,
                 decode_kv_cache_tokens: int,
                 decode_batched_tokens: int,
                 num_requests: int):
        self.scheduled_timestamp = scheduled_timestamp
        self.prefill_kv_cache_tokens = prefill_kv_cache_tokens
        self.prefill_batched_tokens = prefill_batched_tokens
        self.decode_kv_cache_tokens = decode_kv_cache_tokens
        self.decode_batched_tokens = decode_batched_tokens
        self.num_requests = num_requests
    
    def end(self, end_timestamp: float):
        self.end_timestamp = end_timestamp
    

@dataclass
class SequenceMetrics:
    seq_id: str
    arrival_timestamp: float
    swap_out_time_intervals: List[SwapInterval]
    swap_in_time_intervals: List[SwapInterval]
    batch_ids_scheduled: List[int]
    TBTs: List[float]
    arrival_to_scheduled_delay: Optional[float]

    num_prompt_tokens: int
    num_output_tokens: int

    is_prefill: bool
    last_prefill_batch_id: Optional[int]
    prefill_done_to_first_decode_delay: Optional[float]
    arrival_to_first_decode_delay: Optional[float]

    next_tbt_offset: float
    total_offset: float

    def __init__(self, seq_id: str, arrival_timestamp: float):
        self.seq_id = seq_id
        self.arrival_timestamp = arrival_timestamp
        self.swap_intervals: List[SwapInterval] = []
        self.batch_ids_scheduled: List[int] = []
        self.TBTs: List[float] = []

        self.num_prompt_tokens = 0
        self.num_output_tokens = 0

        self.is_prefill = True
        self.last_prefill_batch_id = None
        self.prefill_done_to_first_decode_delay = None
        self.arrival_to_first_decode_delay = None

        self.next_tbt_offset = 0
        self.total_offset = 0

        # Keep track of what the current swap interval is, append later
        self._curr_swap_interval: Optional[SwapInterval] = None
    
    def start_swap_out(self, batch_id: int, num_tokens: int, start_timestamp: float):
        # NOTE: We support many swaps (if we incrementally swap)
        if self._curr_swap_interval is None:
            self._curr_swap_interval = SwapInterval(start_swap_out_batch_id=batch_id, start_swap_out_timestamp=start_timestamp)
        self._curr_swap_interval.swap_out_lens.append(num_tokens)
    
    def start_swap_in(self, batch_id: int, start_timestamp: float):
        assert (
            self._curr_swap_interval is not None and
            self._curr_swap_interval.start_swap_out_batch_id is not None and
            self._curr_swap_interval.start_swap_out_timestamp is not None and
            self._curr_swap_interval.start_swap_in_batch_id is None and
            self._curr_swap_interval.start_swap_in_timestamp is None and
            self._curr_swap_interval.finish_swap_in_batch_id is None and
            self._curr_swap_interval.finish_swap_in_batch_id is None
        )
        self._curr_swap_interval.start_swap_in_batch_id = batch_id
        self._curr_swap_interval.start_swap_in_timestamp = start_timestamp

    def finish_swap_in(self, batch_id: int, end_timestamp: float):
        assert (
            self._curr_swap_interval is not None and
            self._curr_swap_interval.start_swap_out_batch_id is not None and
            self._curr_swap_interval.start_swap_out_timestamp is not None and
            self._curr_swap_interval.start_swap_in_batch_id is not None and
            self._curr_swap_interval.start_swap_in_batch_id is not None and
            self._curr_swap_interval.start_swap_in_timestamp is not None and
            self._curr_swap_interval.finish_swap_in_batch_id is None and
            self._curr_swap_interval.finish_swap_in_timestamp is None
        )
        self._curr_swap_interval.finish_swap_in_batch_id = batch_id
        self._curr_swap_interval.finish_swap_in_timestamp = end_timestamp

        self.swap_intervals.append(self._curr_swap_interval)
        self._curr_swap_interval = None

    def schedule(self, batch_id: int, scheduled_timestamp: float, num_prompt_tokens: int, num_output_tokens: int):
        if not self.batch_ids_scheduled:
            assert self.total_offset == self.next_tbt_offset
            self.arrival_to_scheduled_delay = scheduled_timestamp - self.arrival_timestamp - self.total_offset
        self.batch_ids_scheduled.append(batch_id)
        self.num_prompt_tokens += num_prompt_tokens
        self.num_output_tokens += num_output_tokens

    def get_swap_durations(self) -> List[float]:
        """Get list of complete swap durations."""
        return [interval.get_total_duration() for interval in self.swap_intervals 
                if interval.finish_swap_in_timestamp is not None]

    def get_swap_batch_durations(self) -> List[int]:
        """Get list of swap batch spans."""
        return [interval.get_batch_duration() for interval in self.swap_intervals 
                if interval.finish_swap_in_batch_id is not None]
    
    
class WorkerMetricsStore:
    """
    Metric Store customized for our modifications to Sarathi - we don't benchmark at a granular level like Sarathi
    We assume batch start/end occurs in the worker at the start and end of each call to execute_model
    """

    def __init__(self, disagg_emulation: bool):
        print("Starting WorkerMetricsStore with disagg_emulation:", disagg_emulation)

        self.initial_memory_profiling_done = False
        self.batch_metrics: List[BatchMetrics] = []
        self.sequence_metrics: Dict[str, SequenceMetrics] = {}
        self.engine_scheduler_latencies: List[float] = []

        self.disagg_emulation = disagg_emulation
        self.arrived_sequences_queue = deque()
        self.active_sequences = set()
        self.curr_batch_is_prefill = None
    
    def request_arrived(self, seq_id: str, arrival_timestamp: float):
        # NOTE: arrival_timestamp is simulated so needs to be passed
        if not self.initial_memory_profiling_done:
            return
        
        self.sequence_metrics[seq_id] = SequenceMetrics(seq_id, arrival_timestamp=arrival_timestamp)
        self.arrived_sequences_queue.append(seq_id)
        
    def on_batch_start(self, batch_id: int):
        if not self.initial_memory_profiling_done:
            return

        assert batch_id == len(self.batch_metrics)
        self.batch_metrics.append(BatchMetrics(batch_id, start_timestamp=time.perf_counter()))
    
    def on_batch_scheduled(self, batch_id: int, seq_exec_metadata_list: List[SequenceExecutionMetadata]):
        if not self.initial_memory_profiling_done:
            return

        assert self.batch_metrics[-1].batch_id == batch_id

        # Batch-level metrics
        prefill_kv_cache_tokens = 0  # Number of prefill tokens in the KV cache while this batch is running (includes new tokens)
        decode_kv_cache_tokens = 0  # Number of decode tokens in the KV cache while this batch is running (includes new tokens)
        prefill_batched_tokens = 0
        decode_batched_tokens = 0
        num_requests = len(seq_exec_metadata_list)
        for seq_exec_metadata in seq_exec_metadata_list:

            if self.disagg_emulation:
                if self.curr_batch_is_prefill is None:
                    self.curr_batch_is_prefill = seq_exec_metadata.is_prompt
                else:
                    assert self.curr_batch_is_prefill == seq_exec_metadata.is_prompt

            if seq_exec_metadata.is_prompt:
                prefill_kv_cache_tokens += seq_exec_metadata.seq.get_prompt_len()  # We already allocated the full sequence in KV cache
                prefill_batched_tokens += seq_exec_metadata.num_prompt_tokens
            else:
                decode_kv_cache_tokens += len(seq_exec_metadata.seq.get_all_token_ids())
                decode_batched_tokens += seq_exec_metadata.num_output_tokens
        
        scheduled_timestamp = time.perf_counter()

        self.batch_metrics[-1].schedule(
            scheduled_timestamp=scheduled_timestamp,
            prefill_kv_cache_tokens=prefill_kv_cache_tokens,
            prefill_batched_tokens=prefill_batched_tokens,
            decode_kv_cache_tokens=decode_kv_cache_tokens,
            decode_batched_tokens=decode_batched_tokens,
            num_requests=num_requests
        )

        # print("Batch ID:", batch_id, "Scheduled at:", scheduled_timestamp)
        # print("Sequences", [seq_exec_metadata.seq.seq_id for seq_exec_metadata in seq_exec_metadata_list])
        # Sequence-level metrics
        for seq_exec_metadata in seq_exec_metadata_list:
            self.sequence_metrics[seq_exec_metadata.seq.seq_id].schedule(
                batch_id,
                scheduled_timestamp,
                seq_exec_metadata.num_prompt_tokens,
                seq_exec_metadata.num_output_tokens
            )
    
    def on_batch_end(self, batch_id: int, finished_seq_ids: List[str]):
        if not self.initial_memory_profiling_done:
            return

        assert self.batch_metrics[-1].batch_id == batch_id
        end_timestamp = time.perf_counter()
        self.batch_metrics[-1].end(end_timestamp=end_timestamp)

        # Update self.active_sequences right before we use it for computing total_offset
        while self.arrived_sequences_queue:
            seq_id = self.arrived_sequences_queue.popleft()
            arrival_timestamp = self.sequence_metrics[seq_id].arrival_timestamp
            if end_timestamp > arrival_timestamp:
                self.active_sequences.add(seq_id)
            else:
                self.arrived_sequences_queue.appendleft(seq_id)
                break

        for seq_id in self.active_sequences:
            seq_metrics_obj = self.sequence_metrics[seq_id]
        
            if not seq_metrics_obj.batch_ids_scheduled or seq_metrics_obj.batch_ids_scheduled[-1] != batch_id:
                if self.disagg_emulation and self.curr_batch_is_prefill:
                    delta = end_timestamp - max(seq_metrics_obj.arrival_timestamp, self.batch_metrics[-1].start_timestamp)
                    seq_metrics_obj.total_offset += delta
                    seq_metrics_obj.next_tbt_offset += delta
                    # assert seq_metrics_obj._curr_swap_interval is None
                    # if seq_metrics_obj._curr_swap_interval is not None:
                    #     seq_metrics_obj._curr_swap_interval.timestamp_offset += delta
                    #     seq_metrics_obj._curr_swap_interval.batch_offset += 1
                else:
                    # TODO: maybe keep track of idle time in decode (this excludes time between prefill and first decode token)?
                    pass
            else:
                # NOTE: DANGER: if prefills and decodes are in the same batch, this will be wrong!
                if self.curr_batch_is_prefill:
                    seq_metrics_obj.last_prefill_batch_id = batch_id
                else:
                    delay = (
                        end_timestamp - 
                        # NOTE: assumes batch IDs are just the indices of batch_metrics
                        self.batch_metrics[seq_metrics_obj.batch_ids_scheduled[-2]].end_timestamp -
                        seq_metrics_obj.next_tbt_offset
                    )
                    if seq_metrics_obj.is_prefill:
                        # Corresponds to first decode
                        seq_metrics_obj.prefill_done_to_first_decode_delay = delay
                        seq_metrics_obj.arrival_to_first_decode_delay = end_timestamp - seq_metrics_obj.arrival_timestamp - seq_metrics_obj.total_offset
                        seq_metrics_obj.is_prefill = False
                    else:
                        # Corresponds to subsequent decodes
                        seq_metrics_obj.TBTs.append(delay)

                # Reset offset since we were scheduled this batch
                seq_metrics_obj.next_tbt_offset = 0
        
        for finished_seq_id in finished_seq_ids:
            self.active_sequences.remove(finished_seq_id)
        
        self.curr_batch_is_prefill = None
    
    def on_swap_out_start(self, seq_id: str, num_tokens: int, start_timestamp: float):
        if not self.initial_memory_profiling_done:
            return

        assert seq_id in self.sequence_metrics
        self.sequence_metrics[seq_id].start_swap_out(self.batch_metrics[-1].batch_id, num_tokens, start_timestamp)
    
    def on_swap_in_start(self, seq_id: str, start_timestamp: float):
        if not self.initial_memory_profiling_done:
            return

        assert (
            seq_id in self.sequence_metrics and
            self.sequence_metrics[seq_id]._curr_swap_interval is not None
        )
        self.sequence_metrics[seq_id].start_swap_in(self.batch_metrics[-1].batch_id, start_timestamp)

    def on_swap_in_end(self, seq_id: str, end_timestamp: float):
        if not self.initial_memory_profiling_done:
            return

        assert (
            seq_id in self.sequence_metrics and 
            self.sequence_metrics[seq_id]._curr_swap_interval is not None
        )
        self.sequence_metrics[seq_id].finish_swap_in(self.batch_metrics[-1].batch_id, end_timestamp)
    
    def add_engine_scheduler_latency(self, latency: float):
        self.engine_scheduler_latencies.append(latency)
    
    def mark_initial_memory_profiling_done(self):
        self.initial_memory_profiling_done = True
    
    def reset(self):
        self.batch_metrics = []
        self.sequence_metrics = {}
        self.active_sequences = set()
        self.curr_batch_is_prefill = None
        self.engine_scheduler_latencies = []
    
    def process_metrics(self) -> Dict[str, Any]:
        metrics = {
            "benchmark_metrics": {},
            "sequence_metrics": {},
            "sequence_metrics_raw": {},
            "benchmark_metrics_raw": {}
        }

        # Collect all TBTs and scheduling delays
        all_tbts = []
        all_arrival_to_scheduled_delays = []
        for seq_metrics in self.sequence_metrics.values():
            all_tbts.extend(seq_metrics.TBTs)
            if seq_metrics.arrival_to_scheduled_delay is not None:
                all_arrival_to_scheduled_delays.append(seq_metrics.arrival_to_scheduled_delay)

        # Calculate benchmark-wide metrics
        benchmark_metrics = metrics["benchmark_metrics"]
        benchmark_metrics_raw = metrics["benchmark_metrics_raw"]

        benchmark_metrics["qps"] = len(self.sequence_metrics) / (
            self.batch_metrics[-1].end_timestamp - self.batch_metrics[0].start_timestamp
        )

        # TBT percentiles
        benchmark_metrics["tbt"] = calculate_percentile_values(all_tbts)

        benchmark_metrics["prefill_done_to_first_decode_delay"] = calculate_percentile_values(
            [seq_metrics.prefill_done_to_first_decode_delay for seq_metrics in self.sequence_metrics.values()])

        benchmark_metrics["arrival_to_first_decode_delay"] = calculate_percentile_values(
            [seq_metrics.arrival_to_first_decode_delay for seq_metrics in self.sequence_metrics.values()])

        # Scheduling delay percentiles
        benchmark_metrics["arrival_to_scheduled_delay"] = calculate_percentile_values(all_arrival_to_scheduled_delays)

        benchmark_metrics["engine_scheduler_latency"] = calculate_percentile_values(self.engine_scheduler_latencies)
        benchmark_metrics_raw["engine_scheduler_latency"] = self.engine_scheduler_latencies

        # Calculate per-sequence metrics
        sequence_metrics = metrics["sequence_metrics"]
        sequence_metrics_raw = metrics["sequence_metrics_raw"]
        for seq_id, seq_metrics in self.sequence_metrics.items():
            sequence_metrics[seq_id] = {}
            sequence_metrics_raw[seq_id] = {}
            seq_dict = sequence_metrics[seq_id]

            # Token counts
            seq_dict["num_prompt_tokens"] = seq_metrics.num_prompt_tokens
            seq_dict["num_output_tokens"] = seq_metrics.num_output_tokens

            # Batch counts
            seq_dict["num_batches"] = len(seq_metrics.batch_ids_scheduled)
            seq_dict["num_decode_batches"] = sum(1 for batch_id in seq_metrics.batch_ids_scheduled 
                if not self.batch_metrics[batch_id].prefill_batched_tokens)

            # Timing metrics
            if seq_metrics.batch_ids_scheduled:
                seq_dict["end_to_end_time"] = (
                    self.batch_metrics[seq_metrics.batch_ids_scheduled[-1]].end_timestamp -
                    seq_metrics.arrival_timestamp -
                    seq_metrics.total_offset
                )
                
                # Find first decode batch
                first_decode_batch = None
                for batch_id in seq_metrics.batch_ids_scheduled:
                    if not self.batch_metrics[batch_id].prefill_batched_tokens:
                        first_decode_batch = batch_id
                        break
                
                assert first_decode_batch is not None
                seq_dict["decode_time"] = (
                    self.batch_metrics[seq_metrics.batch_ids_scheduled[-1]].end_timestamp -
                    self.batch_metrics[first_decode_batch].start_timestamp
                )

            seq_dict["arrival_to_scheduled_delay"] = seq_metrics.arrival_to_scheduled_delay
            
            # TBT percentiles
            seq_dict["tbt"] = calculate_percentile_values(seq_metrics.TBTs)
            sequence_metrics_raw[seq_id]["tbt"] = seq_metrics.TBTs

            # Swap metrics
            seq_dict["num_swaps"] = len(seq_metrics.swap_intervals)
            # seq_dict["total_tokens_swapped"] = seq_metrics.get_total_tokens_swapped()
            
            # Swap duration percentiles
            swap_durations = seq_metrics.get_swap_durations()
            seq_dict["swap_duration"] = calculate_percentile_values(swap_durations)
            sequence_metrics_raw[seq_id]["swap_duration"] = swap_durations

            # Swap batch duration percentiles
            swap_duration_num_batches = seq_metrics.get_swap_batch_durations()
            seq_dict["swap_duration_num_batches"] = calculate_percentile_values(swap_duration_num_batches)
            sequence_metrics_raw[seq_id]["swap_duration_num_batches"] = swap_duration_num_batches
        
        benchmark_metrics["end_to_end_time"] = calculate_percentile_values([
            seq_metrics["end_to_end_time"] for seq_metrics in sequence_metrics.values()
        ])

        return metrics