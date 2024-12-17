from collections import deque
from typing import List

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    RoundRobinDisaggEmulationSchedulerConfig,
)
from sarathi.core.block_space_manager import BlockDevice
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.disagg_emulation_base_scheduler import (
    DisaggEmulationBaseScheduler,
)
from sarathi.logger import init_logger

logger = init_logger(__name__)


class RoundRobinDisaggEmulationScheduler(DisaggEmulationBaseScheduler):
    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: RoundRobinDisaggEmulationSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)
        self.max_blocks_to_replace = scheduler_config.max_blocks_to_replace
        self.running_queue = []
        self.swapped_out_queue = []

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        assert not seq.is_finished()
        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_processed(),
            self.scheduler_config.chunk_size - num_batched_tokens,
        )

        return next_num_tokens

    def _update_running_queue_with_freshly_finished_prefills(
        self, running_decodes: List[Sequence]
    ):
        # Update the running queue with requests that finished prefill but haven't started decoding yet
        # Since these are contained in running_decodes
        for seq in running_decodes:
            if seq not in self.running_queue:  # TODO: DANGER check if this works
                self.running_queue.append(seq)

    def _schedule_prefills(
        self,
        running_prefills: List[Sequence],
        running_decodes: List[Sequence],
        now: float,
    ):
        # NOTE: Both lists sorted in FCFS order
        running = []  # NOTE: running decodes, doesn't strictly have to come first in order
        ignored_seq_ids = []
        swap_out_seq_ids = []
        scheduled_seq_id_metadata_list = []

        num_batched_tokens = 0

        self._update_running_queue_with_freshly_finished_prefills(running_decodes)

        # Schedule currently running request
        for seq in running_prefills:
            assert not seq.is_prompt_processing_finished()

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

            # NOTE: Ignoring this for now
            # # The total number of sequences in the RUNNING state should not
            # # exceed the maximum number of sequences.
            # if len(running) >= self.scheduler_config.max_num_seqs:  # NOTE: running here will already incldue decodes, which is great
            #     break

            # check if we can fit the prefill in the batch
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            if next_num_prefill_tokens == 0:
                break

            # Swap lowest priority requests in running list
            num_required_blocks = seq.get_num_logical_blocks()
            total_cpu_blocks_required = 0
            running_decodes_to_swap_out = []
            while num_required_blocks > self.block_manager.get_num_free_blocks(
                BlockDevice.GPU
            ):
                if not self.running_queue:
                    break
                decode_seq = self.running_queue[0]
                num_blocks_allocated = self.block_manager.get_seq_num_blocks_allocated(
                    decode_seq.seq_id, BlockDevice.GPU
                )
                total_cpu_blocks_required += num_blocks_allocated
                if total_cpu_blocks_required > self.max_blocks_to_replace:
                    break
                num_required_blocks -= num_blocks_allocated
                running_decodes_to_swap_out.append(decode_seq)
                self.running_queue.pop(0)

            if num_required_blocks > self.block_manager.get_num_free_blocks(
                BlockDevice.GPU
            ):
                # Restore state
                self.running_queue = running_decodes_to_swap_out + self.running_queue
                break

            for victim_seq in running_decodes_to_swap_out:
                self._swap_out(victim_seq)
                self.swapped_out_queue.append(victim_seq)
                swap_out_seq_ids.append(victim_seq.seq_id)

            seq = self.waiting.pop(0)
            self._allocate(seq)
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_id_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        running.extend(self.running_queue)

        return (
            running,
            ignored_seq_ids,
            [],
            swap_out_seq_ids,
            [],
            [],
            scheduled_seq_id_metadata_list,
        )

    def _free_seq(self, seq: Sequence) -> None:
        super()._free_seq(seq)

        # Remove seq out of running queue
        self.running_queue = list(
            filter(lambda s: s.seq_id != seq.seq_id, self.running_queue)
        )

        for i, s in enumerate(self.swapped_out_queue):
            assert s.seq_id != seq.seq_id

    def _schedule_decodes(self, running_decodes: List[Sequence], now: float):
        running = []
        swap_out_seq_ids = []
        swap_out_lens = []
        swap_in_seq_ids = []
        scheduled_seq_id_metadata_list = []
        num_batched_tokens = 0
        seqs_to_finish_swapping_in = []

        self._update_running_queue_with_freshly_finished_prefills(running_decodes)

        new_swap_outs = []
        new_running_queue_seqs = []
        while self.swapped_out_queue:
            seq = self.swapped_out_queue[0]
            logger.info(f"Trying to schedule swapped out sequence {seq.seq_id}")
            assert seq.is_swapped_out()
            # Swap lowest priority requests in running list
            num_required_blocks = (
                seq.get_num_logical_blocks()
                - self.block_manager.get_seq_num_blocks_allocated(
                    seq.seq_id, BlockDevice.GPU
                )
            )
            total_cpu_blocks_required = 0
            running_decodes_to_swap_out = []
            j = 0
            while num_required_blocks > self.block_manager.get_num_free_blocks(
                BlockDevice.GPU
            ):
                if j == len(self.running_queue):
                    break
                decode_seq = self.running_queue[j]
                num_blocks_allocated = self.block_manager.get_seq_num_blocks_allocated(
                    decode_seq.seq_id, BlockDevice.GPU
                )
                if not num_blocks_allocated:
                    j += 1
                    logger.info(
                        f"Target victim seq not allocated on GPU: {decode_seq.seq_id}"
                    )
                    continue

                blocks_to_swap = (
                    min(num_blocks_allocated, num_required_blocks)
                    if self.cache_config.partial_swap_out
                    else num_blocks_allocated
                )
                total_cpu_blocks_required += blocks_to_swap
                if not self.cache_config.duplicate_kv_cache:
                    if (
                        total_cpu_blocks_required
                        > self.block_manager.get_num_free_blocks(BlockDevice.CPU)
                    ):
                        logger.info(
                            f"Target victim seq {decode_seq.seq_id} swap out requirements exceeds CPU blocks"
                        )
                        break

                if total_cpu_blocks_required > self.max_blocks_to_replace:
                    logger.info(
                        f"Target victim seq {decode_seq.seq_id} swap out requirements exceeds max blocks to replace"
                    )
                    break

                num_required_blocks -= blocks_to_swap
                running_decodes_to_swap_out.append((decode_seq, blocks_to_swap))
                j += 1

            if num_required_blocks <= self.block_manager.get_num_free_blocks(
                BlockDevice.GPU
            ):
                self.running_queue = self.running_queue[j:]
                self.swapped_out_queue.pop(0)
            else:
                logger.info(
                    f"Couldn't schedule swapped out sequence {seq.seq_id}, giving up on scheduling swapped out sequences"
                )
                break  # NOTE: unlike MLFQ we don't skip to the next swapped out request to attempt swap in if we can't swap in this one

            # Swap the sequences we promised to swap out
            for seq_to_swap, num_blocks_to_swap in running_decodes_to_swap_out:
                print(
                    f"Iteration {self._iteration_id}: Swapping out {num_blocks_to_swap} blocks in sequence {seq_to_swap.seq_id}"
                )
                if self.cache_config.partial_swap_out:
                    self._swap_out(seq_to_swap, num_blocks_to_swap)
                    swap_out_lens.append(num_blocks_to_swap)
                else:
                    self._swap_out(seq_to_swap)
                swap_out_seq_ids.append(seq_to_swap.seq_id)
                new_swap_outs.append(seq_to_swap)

            assert (
                seq.is_paused() or seq.is_swapped_out()
            ), f"Sequence {seq.seq_id} is in an invalid state: {seq.get_status()}"

            if seq.is_swapped_out():
                print(f"Iteration {self._iteration_id}: Swapping in {seq.seq_id}")
                assert self.block_manager.can_swap_in_and_append_slot(
                    seq.seq_id, seq.get_num_logical_blocks()
                )
                if self.cache_config.async_swap_in:
                    self._begin_swap_in(seq)
                    seqs_to_finish_swapping_in.append(seq)
                else:
                    self._swap_in(seq)
                swap_in_seq_ids.append(seq.seq_id)

            print(f"Iteration {self._iteration_id}: Scheduling {seq.seq_id}")
            # Append new slots to the sequence group.
            self._append_slot(seq)
            running.append(seq)
            new_running_queue_seqs.append(seq)
            num_batched_tokens += 1
            scheduled_seq_id_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq)
            )

        # NOW WE SCHEDULE FROM THE RUNNING LIST
        # At this point running_queue should only have deocdes that we know for sure we don't need to swap in
        existing_running_seqs = []
        while self.running_queue:
            seq = self.running_queue.pop(-1)
            logger.info(f"Trying to schedule running sequence {seq.seq_id}")
            assert seq.is_paused()

            while not self.block_manager.can_append_slot(seq):
                if self.running_queue:
                    victim_seq = self.running_queue.pop(0)
                    should_break = False
                    logger.info(f"Evicting other running sequence {victim_seq.seq_id}")
                else:
                    victim_seq = seq
                    should_break = True
                    logger.info("Evicting self and giving up on running sequences")
                    # TODO: support partial swap outs here
                self._swap_out(victim_seq)
                swap_out_seq_ids.append(victim_seq.seq_id)
                new_swap_outs.append(victim_seq)
                if should_break:
                    break
            else:
                self._append_slot(seq)
                running.append(seq)
                num_batched_tokens += 1
                scheduled_seq_id_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
                existing_running_seqs.append(seq)

        self.running_queue = (
            list(reversed(existing_running_seqs)) + new_running_queue_seqs
        )  #### !!!!!!
        self.swapped_out_queue.extend(new_swap_outs)

        assert not seqs_to_finish_swapping_in or self.cache_config.async_swap_in
        for seq in seqs_to_finish_swapping_in:
            self._finish_swap_in(seq)

        logger.info(f"Running queue: {[seq.seq_id for seq in self.running_queue]}")
        logger.info(
            f"Swapped out queue: {[seq.seq_id for seq in self.swapped_out_queue]}"
        )
        logger.info(
            f"Scheduler outputs:\nrunning {[seq.seq_id for seq in running]}\nswap_out_seqs_ids {swap_out_seq_ids}\nswap_out_lens {swap_out_lens}\nswap_in_seq_ids {swap_in_seq_ids}\nscheduled_seq_ids {[seq_m.seq_id for seq_m in scheduled_seq_id_metadata_list]}"
        )

        return (
            running,
            [],
            [],
            swap_out_seq_ids,
            swap_out_lens,
            swap_in_seq_ids,
            scheduled_seq_id_metadata_list,
        )
