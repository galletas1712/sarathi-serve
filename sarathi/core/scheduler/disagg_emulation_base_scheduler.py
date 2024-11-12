import time
from typing import List

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SarathiSchedulerConfig,
)
from sarathi.core.block_space_manager import BlockDevice
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence
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

    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        # Swapped in requests should be moved to the running list
        while self.swapped_in:
            seq = self.swapped_in.popitem()[1]
            self.running.append(seq)
            logger.debug(f"(Iteration: {self._iteration_id}) Moving swapped in request {seq.seq_id} into running list")
        
        # Sort both waiting and running queues
        self.running = sorted(self.running, key=lambda seq: now - seq.arrival_time, reverse=True)
        self.waiting = sorted(self.waiting, key=lambda seq: now - seq.arrival_time, reverse=True)

        # Get running prefills and running decodes
        running_prefills: List[Sequence] = []
        running_decodes: List[Sequence] = []
        for seq in self.running:
            assert seq.is_paused(), f"Sequence {seq.seq_id} is not paused, {seq.get_status()}"

            if not seq.is_prompt_processing_finished():
                running_prefills.append(seq)
            else:
                running_decodes.append(seq)

        print(f"------ START SCHEDULER {self._iteration_id} -------")
        print(f"Swapped out: {list(self.swapped_out.keys())}")
        print(f"Running prefills: {[seq.seq_id for seq in running_prefills]}")
        print(f"Running decodes: {[seq.seq_id for seq in running_decodes]}")
        print(self.block_manager.get_block_table_metadata_str())

        prefill_scheduled_success = False

        # NOTE: We will never schedule a prefill if there's decode sequences swapping in - we want to profile this
        # TODO: implement keeping KV cache resident in CPU memory always to mitigate potential issues with this

        # Schedule prefill
        if not self.swapping_in:
            # We will swap out decodes to make room for prefills. They'll remain swapped out until the next iteration
            # In FCFS, they'll be brought back in immediately (begin swap in)
            # In MLFQ, the new requests will take priority
            # NOTE: If we don't add a sequence to scheduled_seq_id_metadata list so it doesn't get run
            # NOTE: _schedule_prefills should also schedule running prefills
            (
                running,
                ignored_seq_ids,
                preempted_seq_ids,
                swap_out_seq_ids,
                begin_swap_in_seq_ids,
                scheduled_seq_id_metadata_list
            ) = self._schedule_prefills(running_prefills, running_decodes, now)

            if scheduled_seq_id_metadata_list:
                print(f"Iteration {self._iteration_id}: scheduled PREFILL")
                prefill_scheduled_success = True
        
        if not prefill_scheduled_success:
            (
                running,
                ignored_seq_ids,
                preempted_seq_ids,
                swap_out_seq_ids,
                begin_swap_in_seq_ids,
                scheduled_seq_id_metadata_list
            ) = self._schedule_decodes(running_decodes, now)

        self.running = running
        self.running = sorted(self.running, key=lambda seq: now - seq.arrival_time, reverse=True)
        self.waiting = sorted(self.waiting, key=lambda seq: now - seq.arrival_time, reverse=True)

        # Calculate num waiting just for print
        num_waiting = 0
        for seq in self.waiting:
            if seq.arrival_time <= now:
                num_waiting += 1
        print("Number of waiting requests: ", num_waiting)
        print(f"Swapped out: {len(self.swapped_out)}, Swapped in: {len(self.swapped_in)}, Swapping in: {len(self.swapping_in)}")
        print(f"------ END SCHEDULER {self._iteration_id} -------")

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            swap_out_seq_ids=swap_out_seq_ids,
            begin_swap_in_seq_ids=begin_swap_in_seq_ids,
            scheduled_seq_id_metadata_list=scheduled_seq_id_metadata_list,
        )
