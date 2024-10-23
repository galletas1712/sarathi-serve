import logging
import time
from typing import List

import numpy as np

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    OccasionalSwappingSchedulerConfig,
)
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)


class OccasionalSwappingScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: OccasionalSwappingSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        self.chunk_size = self.scheduler_config.chunk_size
        self._first_decode_iteration = None

    def get_block_space_manager_class(self):
        return SarathiBlockSpaceManager

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence
    ) -> int:
        assert not seq.is_finished()

        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed(),
            self.chunk_size,
        )

        return next_num_tokens

    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        seq_ids_to_swap_out: List[str] = []
        seq_ids_to_swap_in: List[str] = []
        scheduled_seq_id_metadata_list: List[SequenceScheduleMetadata] = []

        num_batched_tokens: int = 0

        # We will schedule only waiting requests and prefills first till completion, then decodes.
        # If decodes are running, we'll swap some of them out
        # NOTE: This scheduler only works if we submit all requests at the very beginning, and run until completion
        # This scheduler is designed for benchmarking decodes only!

        running_prefills: List[Sequence] = []
        running_decodes: List[Sequence] = []

        while self.swapped_in:
            seq = self.swapped_in.popitem()[1]
            self.running.insert(0, seq)
            logger.debug(f"(Iteration: {self._iteration_id}) Moving swapped in request {seq.seq_id} to beginning of running list")

        # This is different from the other schedulers. We *clear* the entire running list, and we don't need to assign a local version at the end
        while self.running:
            seq = self.running.pop(0)
            assert seq.is_paused(), f"Sequence {seq.seq_id} is not paused! The status is {seq.get_status()}"
            if not seq.prompt_stage_processing_finished:
                running_prefills.append(seq)
            else:
                running_decodes.append(seq)
        
        logger.debug(f"(Iteration: {self._iteration_id}) Waiting queue: {[seq.seq_id for seq in self.waiting]}")
        logger.debug(f"(Iteration: {self._iteration_id}) Running requests that need prefill: {[seq.seq_id for seq in running_prefills]}")
        logger.debug(f"(Iteration: {self._iteration_id}) Running requests that need decode: {[seq.seq_id for seq in running_decodes]}")

        # Schedule waiting and running prefills first
        if self.waiting or running_prefills:
            logger.debug(f"(Iteration: {self._iteration_id}) Scheduling new requests and running prefills")

            # Schedule waiting requests
            while self.waiting:
                seq = self.waiting[0]

                if seq.arrival_time > now:
                    break

                # Let's not even deal with requests that are too long
                assert self._check_request_prompt_length(seq)
                # Let's not even deal with requests that we can't allocate
                assert self.block_manager.can_allocate(seq)

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                if len(self.running) >= self.scheduler_config.max_num_seqs:
                    logger.debug(f"Reached max number of sequences in waiting queue. Will try again in the next batch")
                    break

                next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(seq)
                # Since we don't have a max token count in batch setting, this shouldn't be 0 ever
                assert next_num_prefill_tokens > 0

                seq = self.waiting.pop(0)
                self._allocate(seq)  # Need to allocate since running sequence for the first time
                num_batched_tokens += next_num_prefill_tokens
                scheduled_seq_id_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(
                        seq, prompt_chunk_len=next_num_prefill_tokens
                    )
                )
                self.running.append(seq)
                logger.debug(f"(Iteration: {self._iteration_id}) Added waiting request {seq.seq_id} to running list!")
            
            # Schedule running prefills
            for seq in running_prefills:
                next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(seq)
                assert next_num_prefill_tokens > 0

                if len(self.running) >= self.scheduler_config.max_num_seqs:
                    logger.debug(f"Reached max number of sequences in prefill queue. Will try again in the next batch")
                    break

                # No need to allocate since we've already allocated memory for this sequence, and prefills are never preempted
                num_batched_tokens += next_num_prefill_tokens
                scheduled_seq_id_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(
                        seq, prompt_chunk_len=next_num_prefill_tokens
                    )
                )
                self.running.append(seq)
                logger.debug(f"(Iteration: {self._iteration_id}) Added previously running prefill {seq.seq_id} to running list!")
        else:
            if self._first_decode_iteration is None:
                self._first_decode_iteration = self._iteration_id
            logger.debug(f"(Iteration: {self._iteration_id}) Scheduling decodes only")

            if (self._iteration_id - self._first_decode_iteration) % 8 == 3:
                # Swap in all swapped out requests
                # Important: don't modify list while iterating
                swapped_out_seq_ids = list(self.swapped_out.keys())
                for seq_id in swapped_out_seq_ids:
                    logger.debug(f"(Iteration: {self._iteration_id}) Swapping in {seq_id}!")
                    seq_ids_to_swap_in.append(seq_id)
                    self._begin_swap_in(self.swapped_out[seq_id])
                
            # Schedule decodes
            for seq in running_decodes:
                # Swap out every last iteration (starting from the first decode iteration)
                if (self._iteration_id - self._first_decode_iteration) % 8 == 7 and (len(self.swapping_out) + len(self.swapped_out)) < len(running_decodes) // 2:
                    self._begin_swap_out(seq)
                    seq_ids_to_swap_out.append(seq.seq_id)
                    logger.debug(f"(Iteration: {self._iteration_id}) Swapped out {seq.seq_id}! {len(self.swapping_out)} sequences swapping, {len(self.swapped_out)} sequences swapped out")
                    continue

                if len(self.running) >= self.scheduler_config.max_num_seqs:
                    logger.debug(f"(Iteration: {self._iteration_id}) Reached max number of sequences in decode queue. Will try again in the next batch")
                    break

                self._append_slot(seq)
                self.running.append(seq)
                num_batched_tokens += 1
                scheduled_seq_id_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
                logger.debug(f"(Iteration: {self._iteration_id}) Added {seq.seq_id} (decode) to running list!")

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=[],
            preempted_seq_ids=[],
            begin_swap_in_seq_ids=seq_ids_to_swap_in,
            begin_swap_out_seq_ids=seq_ids_to_swap_out,
            scheduled_seq_id_metadata_list=scheduled_seq_id_metadata_list,
        )
