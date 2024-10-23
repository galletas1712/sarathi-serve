import enum
import time
from typing import List

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SarathiSchedulerConfig,
)
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)


class DisaggEmulationBaseScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SarathiSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

    def get_block_space_manager_class(self):
        return SarathiBlockSpaceManager

    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        # Swapped in requests should be moved to the running list
        while self.swapped_in:
            seq = self.swapped_in.popitem()[1]
            self.running.append(seq)
            logger.debug(f"(Iteration: {self._iteration_id}) Moving swapped in request {seq.seq_id} into running list")
        
        # Sort both waiting and running queues
        self.running = self.policy.sort_by_priority(now, self.running)
        self.waiting = self.policy.sort_by_priority(now, self.waiting)

        # Get running prefills and running decodes
        running_prefills: List[Sequence] = []
        running_decodes: List[Sequence] = []
        for seq in self.running:
            assert seq.is_paused()

            if not seq.prompt_stage_processing_finished:
                running_prefills.append(seq)
            else:
                running_decodes.append(seq)

        prefill_scheduled_success = False

        # NOTE: We will never schedule a prefill if there's decode sequences swapping in/out - we want to profile this
        # We should also never schedule a prefill if there are outstanding swapped out requests - this emulates FCFS backpressure behavior
        # TODO: implement keeping KV cache resident in CPU memory always to mitigate potential issues with this
        # Schedule prefill!
        if (
            not self.swapping_out and
            not self.swapping_in and
            not self.swapped_out
        ):
            # print(f"Iteration {self._iteration_id}: scheduling prefill")
            # NOTE: we keep decodes in memory, but don't add it to scheduled_seq_id_metadata list so it doesn't get run
            # NOTE: _schedule_prefills should also schedule running prefills
            (
                running,
                ignored_seq_ids,
                preempted_seq_ids,
                begin_swap_in_seq_ids,
                begin_swap_out_seq_ids,
                scheduled_seq_id_metadata_list
            ) = self._schedule_prefills(running_prefills, running_decodes, now)

            if scheduled_seq_id_metadata_list:
                prefill_scheduled_success = True
        
        if not prefill_scheduled_success:
            # print(f"Iteration {self._iteration_id}: scheduling decode")
            (
                running,
                ignored_seq_ids,
                preempted_seq_ids,
                begin_swap_in_seq_ids,
                begin_swap_out_seq_ids,
                scheduled_seq_id_metadata_list
            ) = self._schedule_decodes(running_decodes, now)

        self.running = running
        self.running = self.policy.sort_by_priority(now, self.running)
        self.waiting = self.policy.sort_by_priority(now, self.waiting)

        # print(f"Iteration {self._iteration_id} running: {running}")
        # print(f"Iteration {self._iteration_id} ignored: {ignored_seq_ids}")
        # print(f"Iteration {self._iteration_id} preempted: {preempted_seq_ids}")
        # print(f"Iteration {self._iteration_id} begin swap in: {begin_swap_in_seq_ids}")
        # print(f"Iteration {self._iteration_id} begin swap out: {begin_swap_out_seq_ids}")
        # print(f"Iteration {self._iteration_id} scheduled: {scheduled_seq_id_metadata_list}")

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            begin_swap_in_seq_ids=begin_swap_in_seq_ids,
            begin_swap_out_seq_ids=begin_swap_out_seq_ids,
            scheduled_seq_id_metadata_list=scheduled_seq_id_metadata_list,
        )
