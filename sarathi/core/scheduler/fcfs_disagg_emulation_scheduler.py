import enum
import time
from typing import List

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SarathiSchedulerConfig,
)
from sarathi.config.config import FCFSDisaggEmulationSchedulerConfig
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


class FCFSDisaggEmulationScheduler(DisaggEmulationBaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: FCFSDisaggEmulationSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)
        self.policy = PolicyFactory.get_policy("fcfs")

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
        if self.swapped_out:
            # There are requests currently swapped ot, so we can't schedule any new requests
            return (
                running_decodes + running_prefills,
                [],
                [],
                [],
                [],
                [],
            )

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

    def _schedule_decodes(self, running_decodes: List[Sequence], now: float):
        running = []
        swap_out_seq_ids = []
        begin_swap_in_seq_ids = []
        scheduled_seq_id_metadata_list = []

        num_batched_tokens = 0

        # True FCFS order
        queue = self.policy.sort_by_priority(now, [*running_decodes, *self.swapped_out.values()])

        while queue:
            seq = queue.pop(0)

            assert seq.is_paused() or seq.is_swapped_out()

            def can_append_slot():
                return self.block_manager.can_append_slot(seq)
            
            def can_swap_in_and_append_slot():
                return self.block_manager.can_swap_in_and_append_slot(seq.seq_id, len(seq.logical_token_blocks))
            
            can_schedule = can_append_slot if seq.is_paused() else can_swap_in_and_append_slot

            while not can_schedule():
                if queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = queue.pop(-1)
                    if victim_seq.is_paused():
                        print(f"Iteration {self._iteration_id}: Swapping out {victim_seq.seq_id} to make space for {seq.seq_id}")
                        self._swap_out(victim_seq)
                        swap_out_seq_ids.append(victim_seq.seq_id)
                    else:
                        print(f"Iteration {self._iteration_id}: Leaving {victim_seq.seq_id} swapped out to make space for {seq.seq_id}")
                        assert victim_seq.is_swapped_out()
                else:
                    # No other sequence groups can be prempted.
                    # Preempt the current sequence group.
                    if seq.is_paused():
                        print(f"Iteration {self._iteration_id}: Swapping out {seq.seq_id} because can't run :(")
                        self._swap_out(seq)
                        swap_out_seq_ids.append(seq.seq_id)
                    else:
                        print(f"Iteration {self._iteration_id}: Can't swap in {seq.seq_id} because no space :(")
                        assert seq.is_swapped_out()
                    break
            else:
                if seq.is_paused():
                    print(f"Iteration {self._iteration_id}: Scheduling {seq.seq_id}")
                    # Append new slots to the sequence group.
                    self._append_slot(seq)
                    running.append(seq)
                    num_batched_tokens += 1
                    scheduled_seq_id_metadata_list.append(
                        SequenceScheduleMetadata.from_sequence(seq)
                    )
                elif seq.is_swapped_out():
                    print(f"Iteration {self._iteration_id}: Swapping in {seq.seq_id}")
                    assert self.block_manager.can_swap_in_and_append_slot(seq.seq_id, len(seq.logical_token_blocks))
                    self._begin_swap_in(seq)
                    begin_swap_in_seq_ids.append(seq.seq_id)
                else:
                    assert False, f"Sequence {seq.seq_id} is in an invalid state: {seq.get_status()}"
        
        return (
            running,
            [],
            [],
            swap_out_seq_ids,
            begin_swap_in_seq_ids,
            scheduled_seq_id_metadata_list
        )