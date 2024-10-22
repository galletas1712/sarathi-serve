from dataclasses import dataclass
import time
from typing import Dict, List, Optional
from collections import deque

from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SequenceMetadata

@dataclass
class SwapInterval:
    """
    start_batch_id: the batch id that started the swap (before the model runs, after the batch is scheduled)
    end_batch_id: the batch id that ended the swap (at the beginning of the batch)
    start_timestamp: timestamp in worker loop when the swap started
    end_timestamp: timestamp in worker right before we notify the engine that the swap has completed
    """
    start_batch_id: int
    start_timestamp: float
    end_batch_id: Optional[int] = None
    end_timestamp: Optional[float] = None


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

    next_tbt_offset: float
    total_offset: float

    def __init__(self, seq_id: str, arrival_timestamp: float):
        self.seq_id = seq_id
        self.arrival_timestamp = arrival_timestamp
        self.swap_out_time_intervals = []
        self.swap_in_time_intervals = []
        self.batch_ids_scheduled = []
        self.TBTs = []

        self.next_tbt_offset = 0
        self.total_offset = 0

        # Keep track of what the current swap interval is, append later
        self._curr_swap_interval = None
        self._curr_swap_is_swap_in = None

    def start_swap(self, batch_id: int, swap_in: bool, start_timestamp: float):
        assert (
            self._curr_swap_interval is None and
            self._curr_swap_is_swap_in is None
        )
        self._curr_swap_interval = SwapInterval(start_batch_id=batch_id, start_timestamp=start_timestamp)
        self._curr_swap_is_swap_in = swap_in
    
    def finish_swap(self, batch_id: int, swap_in: bool, end_timestamp: float):
        assert (
            self._curr_swap_interval is not None and 
            self._curr_swap_is_swap_in == swap_in
        )
        self._curr_swap_interval.end_batch_id = batch_id
        self._curr_swap_interval.end_timestamp = end_timestamp

        if swap_in:
            self.swap_in_time_intervals.append(self._curr_swap_interval)
        else:
            self.swap_out_time_intervals.append(self._curr_swap_interval)

        self._curr_swap_interval = None
        self._curr_swap_is_swap_in = None
    
    def schedule(self, batch_id: int, scheduled_timestamp: float):
        if not self.batch_ids_scheduled:
            self.arrival_to_scheduled_delay = scheduled_timestamp - self.arrival_timestamp - self.total_offset
        self.batch_ids_scheduled.append(batch_id)
    
    
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
    
    def on_batch_scheduled(self, batch_id: int, seq_metadata_list: List[SequenceMetadata]):
        if not self.initial_memory_profiling_done:
            return

        assert self.batch_metrics[-1].batch_id == batch_id

        # Batch-level metrics
        prefill_kv_cache_tokens = 0  # Number of prefill tokens in the KV cache while this batch is running (includes new tokens)
        decode_kv_cache_tokens = 0  # Number of decode tokens in the KV cache while this batch is running (includes new tokens)
        prefill_batched_tokens = 0
        decode_batched_tokens = 0
        num_requests = len(seq_metadata_list)
        for seq_metadata in seq_metadata_list:

            if self.disagg_emulation:
                if self.curr_batch_is_prefill is None:
                    self.curr_batch_is_prefill = seq_metadata.is_prompt
                else:
                    assert self.curr_batch_is_prefill == seq_metadata.is_prompt

            if seq_metadata.is_prompt:
                prefill_kv_cache_tokens += seq_metadata.seq.get_prompt_len()  # We already allocated the full sequence in KV cache
                prefill_batched_tokens += seq_metadata.num_prompt_tokens
            else:
                decode_kv_cache_tokens += len(seq_metadata.seq.get_token_ids())
                decode_batched_tokens += seq_metadata.num_output_tokens
        
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
        # print("Sequences", [seq_metadata.seq.seq_id for seq_metadata in seq_metadata_list])
        # Sequence-level metrics
        for seq_metadata in seq_metadata_list:
            self.sequence_metrics[seq_metadata.seq.seq_id].schedule(batch_id, scheduled_timestamp)
    
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
                else:
                    # TODO: maybe keep track of idle time in decode (this excludes time between prefill and first decode token)?
                    pass
            else:
                if not self.curr_batch_is_prefill and len(seq_metrics_obj.batch_ids_scheduled) > 1:
                    seq_metrics_obj.TBTs.append(
                        end_timestamp - 
                        # NOTE: assumes batch IDs are just the indices of batch_metrics
                        self.batch_metrics[seq_metrics_obj.batch_ids_scheduled[-2]].end_timestamp -
                        seq_metrics_obj.next_tbt_offset
                    )

                seq_metrics_obj.next_tbt_offset = 0

        for finished_seq_id in finished_seq_ids:
            self.active_sequences.remove(finished_seq_id)
        
        self.curr_batch_is_prefill = None
    
    def on_swap_start(self, seq_id: str, swap_in: bool, start_timestamp: float):
        if not self.initial_memory_profiling_done:
            return

        assert (
            seq_id in self.sequence_metrics and 
            self.sequence_metrics[seq_id]._curr_swap_interval is None
        )
        self.sequence_metrics[seq_id].start_swap(self.batch_metrics[-1].batch_id, swap_in, start_timestamp)

    def on_swap_end(self, seq_id: str, swap_in: bool, end_timestamp: float):
        if not self.initial_memory_profiling_done:
            return

        assert (
            seq_id in self.sequence_metrics and 
            self.sequence_metrics[seq_id]._curr_swap_interval is not None and
            self.sequence_metrics[seq_id]._curr_swap_is_swap_in == swap_in
        )
        self.sequence_metrics[seq_id].finish_swap(self.batch_metrics[-1].batch_id, swap_in, end_timestamp)
    
    def mark_initial_memory_profiling_done(self):
        self.initial_memory_profiling_done = True
    
    def reset(self):
        self.batch_metrics = []
        self.sequence_metrics = {}
        self.active_sequences = set()
        self.curr_batch_is_prefill = None
    
    def plot(self):
        print("Sequence metrics:")
        for seq_metrics in self.sequence_metrics.values():
            print("Decode length:", len(seq_metrics.batch_ids_scheduled))
            print("TBT length:", len(seq_metrics.TBTs))
            print("Min TBT:", min(seq_metrics.TBTs))
            print("Max TBT:", max(seq_metrics.TBTs))
            e2e_time = (
                self.batch_metrics[seq_metrics.batch_ids_scheduled[-1]].end_timestamp -
                seq_metrics.arrival_timestamp - 
                seq_metrics.total_offset
           )
            print("E2E time:", e2e_time)
            print("Scheduling delay:", seq_metrics.arrival_to_scheduled_delay)
            print("Arrived at timestamp:", seq_metrics.arrival_timestamp)
            print("Scheduled at iteration with timestamp:", seq_metrics.batch_ids_scheduled[0], self.batch_metrics[seq_metrics.batch_ids_scheduled[0]].scheduled_timestamp)
            print("Swap outs:", seq_metrics.swap_out_time_intervals)
            print("Swap ins:", seq_metrics.swap_in_time_intervals)
            print()
            # print("Total offset:", seq_metrics.total_offset)
            # print("Arrival timestamp:", seq_metrics.arrival_timestamp)
            # print("Timestamp of start batch:", self.batch_metrics[seq_metrics.batch_ids_scheduled[0]].start_timestamp)
            # print("End timestamp of last batch:", self.batch_metrics[seq_metrics.batch_ids_scheduled[-1]].end_timestamp)
            # print("Difference:", self.batch_metrics[seq_metrics.batch_ids_scheduled[-1]].end_timestamp - self.batch_metrics[seq_metrics.batch_ids_scheduled[0]].start_timestamp)


        # print("Batch Metrics:")
        # for batch_metrics in self.batch_metrics:
            # print("Batch ID:", batch_metrics.batch_id, "PREFILL" if batch_metrics.prefill_batched_tokens > 0 else "decode")
            # print("Total batch elapsed:", (batch_metrics.end_timestamp - batch_metrics.start_timestamp) * 1000)
            # print()