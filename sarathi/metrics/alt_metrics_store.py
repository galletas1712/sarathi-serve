import time
from collections import deque
from dataclasses import asdict
from typing import Any, Dict, List

from sarathi.core.datatypes.sequence import SequenceExecutionMetadata
from sarathi.metrics.types.batch_metrics import BatchMetrics
from sarathi.metrics.types.global_metrics import GlobalMetrics
from sarathi.metrics.types.seq_metrics import ProcessedSeqMetrics, SequenceMetrics


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

        self.sequence_metrics[seq_id] = SequenceMetrics(
            seq_id, arrival_timestamp=arrival_timestamp
        )
        self.arrived_sequences_queue.append(seq_id)

    def on_batch_start(self, batch_id: int):
        if not self.initial_memory_profiling_done:
            return

        assert batch_id == len(self.batch_metrics)
        self.batch_metrics.append(
            BatchMetrics(batch_id, start_timestamp=time.perf_counter())
        )

    def on_batch_scheduled(
        self, batch_id: int, seq_exec_metadata_list: List[SequenceExecutionMetadata]
    ):
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
                prefill_kv_cache_tokens += (
                    seq_exec_metadata.seq.get_prompt_len()
                )  # We already allocated the full sequence in KV cache
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
            num_requests=num_requests,
        )

        # print("Batch ID:", batch_id, "Scheduled at:", scheduled_timestamp)
        # print("Sequences", [seq_exec_metadata.seq.seq_id for seq_exec_metadata in seq_exec_metadata_list])
        # Sequence-level metrics
        for seq_exec_metadata in seq_exec_metadata_list:
            self.sequence_metrics[seq_exec_metadata.seq.seq_id].schedule(
                batch_id,
                scheduled_timestamp,
                seq_exec_metadata.num_prompt_tokens,
                seq_exec_metadata.num_output_tokens,
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

            if (
                not seq_metrics_obj.batch_ids_scheduled
                or seq_metrics_obj.batch_ids_scheduled[-1] != batch_id
            ):
                if self.disagg_emulation and self.curr_batch_is_prefill:
                    delta = end_timestamp - max(
                        seq_metrics_obj.arrival_timestamp,
                        self.batch_metrics[-1].start_timestamp,
                    )
                    seq_metrics_obj.total_offset += delta
                    seq_metrics_obj.next_tbt_offset += delta
                else:
                    # TODO: maybe keep track of idle time in decode (this excludes time between prefill and first decode token)?
                    pass
            else:
                # NOTE: DANGER: if prefills and decodes are in the same batch, this will be wrong!
                if self.curr_batch_is_prefill:
                    seq_metrics_obj.last_prefill_batch_id = batch_id
                else:
                    delay = (
                        end_timestamp
                        -
                        # NOTE: assumes batch IDs are just the indices of batch_metrics
                        self.batch_metrics[
                            seq_metrics_obj.batch_ids_scheduled[-2]
                        ].end_timestamp
                        - seq_metrics_obj.next_tbt_offset
                    )
                    if seq_metrics_obj.is_prefill:
                        # Corresponds to first decode
                        seq_metrics_obj.prefill_done_to_first_decode_delay = delay
                        seq_metrics_obj.arrival_to_first_decode_delay = (
                            end_timestamp
                            - seq_metrics_obj.arrival_timestamp
                            - seq_metrics_obj.total_offset
                        )
                        seq_metrics_obj.is_prefill = False
                    else:
                        # Corresponds to subsequent decodes
                        seq_metrics_obj.TBTs.append(delay)

                # Reset offset since we were scheduled this batch
                seq_metrics_obj.next_tbt_offset = 0

        for finished_seq_id in finished_seq_ids:
            self.active_sequences.remove(finished_seq_id)

        self.curr_batch_is_prefill = None

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
        all_processed_seq_metrics = {
            seq_id: asdict(
                ProcessedSeqMetrics.create(
                    self.sequence_metrics[seq_id], self.batch_metrics
                )
            )
            for seq_id in self.sequence_metrics
        }
        global_metrics = asdict(
            GlobalMetrics.create(
                all_processed_seq_metrics,
                self.batch_metrics,
                self.engine_scheduler_latencies,
            )
        )
        return {
            "global_metrics": global_metrics,
            "seq_metrics": all_processed_seq_metrics,
        }
