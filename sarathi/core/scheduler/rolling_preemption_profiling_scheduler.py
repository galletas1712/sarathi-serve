import time
from typing import List

import numpy as np

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
)
from sarathi.config.config import RollingPreemptionProfilingSchedulerConfig
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)


class RollingPreemptionProfilingScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: RollingPreemptionProfilingSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        self.chunk_size = self.scheduler_config.chunk_size

    def get_block_space_manager_class(self):
        return SarathiBlockSpaceManager

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        # NOTE: Sarathi schedule treats chunk size as the maximum number of batched tokens
        # We're changing the meaning of chunk size here to mean the maximum number of consecutive tokens in each sequence in the batch

        assert not seq.is_finished()

        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed(),
            self.chunk_size,
        )

        return next_num_tokens

    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        running: List[Sequence] = []
        ignored_seq_ids: List[str] = []
        preempted_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        num_batched_tokens: int = 0

        ######################################################################
        # Phase 1: Add existing running sequence groups to the batch.
        # There are two cases:
        # 1. The sequence group has incomplete prefill. The routine
        # remains identical to the one in sarathi scheduler for such sequences.
        # 2. The sequence group has completed prefill. In this case, we need to
        # check for memory availability for the next chunk of decode tokens, and preempt
        # some sequence groups if necessary. Note that, the preempted sequence groups
        # might belong to either of the two categories.
        ######################################################################

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Which prefills are still running
        running_prefills: List[Sequence] = []
        to_preempt: List[Sequence] = []

        # First, separate into prefill currently running and those that are completed, which we will preempt
        while self.running:
            seq = self.running.pop(0)

            if not seq.prompt_stage_processing_finished:
                running_prefills.append(seq)
            else:
                to_preempt.append(seq)

        # now add the requests with prefill incomplete
        # the memory for all these prefills has already been allocated
        # so we should be able to run all of them
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
            # NOTE: assume non pipeline
            assert next_num_prefill_tokens > 0

            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            # print(f"Prefill request ({seq.seq_id}) scheduled to run, next_num_prefill_tokens = {next_num_prefill_tokens} processed {seq.prompt_tokens_processed} so far")
            running.append(seq)

        while self.waiting:
            seq = self.waiting[-1]

            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                break
        
            print(f"Allocating request ({seq.seq_id}), len(running) = {len(running)}, max_num_seqs = {self.scheduler_config.max_num_seqs}")

            # If the sequence group cannot be allocated, stop.
            assert self.block_manager.can_allocate(seq)

            # check if we can fit the prefill in the batch
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            if next_num_prefill_tokens == 0:
                break

            # Here, we will preempt something in our list if it exists
            if to_preempt:
                seq_to_preempt = to_preempt.pop(-1)
                preempted_seq_ids.append(seq_to_preempt.seq_id)
                self._preempt(seq_to_preempt)

            seq = self.waiting.pop(-1)
            self._allocate(seq)
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)
    
        while to_preempt:
            seq_to_preempt = to_preempt.pop(-1)
            preempted_seq_ids.append(seq_to_preempt.seq_id)
            self._preempt(seq_to_preempt)
        
        # print(f"Preempting requests: {preempted_seq_ids}")

        # make sure that prefills are at the start of the batch, so that we don't violate assumptions
        # made in the original vllm codebase
        self.running = running

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            swapped_seq_ids=[],
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
