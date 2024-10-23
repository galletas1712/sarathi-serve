from typing import Dict, List
from collections import deque

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SarathiSchedulerConfig,
)
from sarathi.config.config import MLFQDisaggEmulationSchedulerConfig
from sarathi.core.block_space_manager.base_block_space_manager import BlockDevice
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.policy import PolicyFactory
from sarathi.core.scheduler.disagg_emulation_base_scheduler import DisaggEmulationBaseScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)


class MLFQDisaggEmulationScheduler(DisaggEmulationBaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: MLFQDisaggEmulationSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)
        self.policy = PolicyFactory.get_policy("mlfq")
        self.quantums = scheduler_config.get_quantums()

        self.num_consecutive_iterations: Dict[str, int] = {}
        self.queues = [deque() for _ in range(len(self.quantums))]

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        assert not seq.is_finished()
        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed(),
            self.scheduler_config.chunk_size - num_batched_tokens,
        )

        return next_num_tokens

    def _schedule_prefills(self, running_prefills: List[Sequence], running_decodes: List[Sequence], now: float):
        running = [*running_decodes] # NOTE: running decodes, doesn't strictly have to come first in order
        ignored_seq_ids = []
        scheduled_seq_id_metadata_list = []

        num_batched_tokens = 0

        # Schedule currently running request
        for seq in running_prefills:
            assert not seq.prompt_stage_processing_finished

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            # as long as the request could fit in the batch previously
            # it should be able to fit in the batch now
            # so in non-pipeline case this condition should always be false
            # however, in pipeline case, the grouping of requests can change
            # between different microbatches, so this is not guaranteed to be always true
            if next_num_prefill_tokens == 0:
                running.append(seq)
                continue

            num_batched_tokens += next_num_prefill_tokens
            
            scheduled_seq_id_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        # Schedule new prefills
        while self.waiting:
            seq = self.waiting[0]

            # This is required to handle benchmarking where we set request arrival time ahead of time
            if seq.arrival_time > now:
                break

            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            # If the sequence group cannot be allocated, stop.
            if not self.block_manager.can_allocate(seq):
                # this is different from vllm scheduler
                # even if we cannot allocate this sequence group
                # there might be other sequence groups that can be allocated
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:  # NOTE: running here will already incldue decodes, which is great
                break

            # check if we can fit the prefill in the batch
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            if next_num_prefill_tokens == 0:
                break

            seq = self.waiting.pop(0)
            self._allocate(seq)
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_id_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)
        
        return (
            running,
            ignored_seq_ids,
            [],
            [],
            [],
            scheduled_seq_id_metadata_list
        )

    def _get_quantum(self, num_running_iterations: int):
        quantum_idx = 0
        for i in range(len(self.quantums)):
            if num_running_iterations > self.quantums[i]:
                quantum_idx = i

        quantum_idx = min(quantum_idx, len(self.quantums) - 1)
        
        return quantum_idx
    
    def add_seq(self, seq: Sequence) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq)
        self.num_consecutive_iterations[seq.seq_id] = 0

    def _schedule_decodes(self, running_decodes: List[Sequence]):
        running = []
        begin_swap_in_seq_ids = []
        begin_swap_out_seq_ids = []
        scheduled_seq_id_metadata_list = []

        num_batched_tokens = 0

        while running_decodes:
            seq = running_decodes.pop(0)

            if not seq.is_paused():
                running.append(seq)
                continue

            while not self.block_manager.can_append_slot():
                if running_decodes:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = running_decodes.pop(-1)
                    self._begin_swap_out(victim_seq)
                    begin_swap_out_seq_ids.append(victim_seq.seq_id)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._begin_swap_out(seq)
                    begin_swap_out_seq_ids.append(seq.seq_id)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq)
                running.append(seq)
                num_batched_tokens += 1
                scheduled_seq_id_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
        
        # Swap in outstanding swapped out sequences if we have room
        # This assumes FCFS backpressure behavior in the base scheduler
        swapped_out = list(sorted(self.swapped_out.values(), key=lambda x: x.arrival_time))
        while swapped_out:
            seq = swapped_out.pop(0)

            if not self.block_manager.can_swap_in(seq.seq_id):
                break
            
            self._begin_swap_in(seq)
            begin_swap_in_seq_ids.append(seq.seq_id)
        
        return (
            running,
            [],
            [],
            begin_swap_in_seq_ids,
            begin_swap_out_seq_ids,
            scheduled_seq_id_metadata_list
        )