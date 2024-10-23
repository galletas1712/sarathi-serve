from typing import Dict, List

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
        self.quantums = scheduler_config.get_quantums()

        self.num_consecutive_iterations: Dict[str, int] = {}
        self.decode_queues: List[List[Sequence]] = [[] for _ in range(len(self.quantums))]
        self.request_quantum_map = {}

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
    
    def _update_quantums(self):
        for quantum_idx in reversed(range(len(self.quantums))):
            indices_to_remove = []
            for i in range(len(self.decode_queues[quantum_idx])):
                seq = self.decode_queues[quantum_idx][i]
                next_quantum = self._get_quantum(self.num_consecutive_iterations[seq.seq_id])
                if next_quantum != quantum_idx:
                    # print(f"Moving {seq.seq_id} from quantum {quantum_idx} to {next_quantum} since run count is {self.num_consecutive_iterations[seq.seq_id]}")
                    self.decode_queues[next_quantum].append(seq)
                    self.request_quantum_map[seq.seq_id] = next_quantum
                    indices_to_remove.append(i)
            for i in reversed(indices_to_remove):
                self.decode_queues[quantum_idx].pop(i)
    
    def _update_num_consecutive_iterations(self, running: List[Sequence]):
        running_seq_ids = [seq.seq_id for seq in running]

        for seq_id in running_seq_ids:
            if seq_id not in self.num_consecutive_iterations:
                self.num_consecutive_iterations[seq_id] = 0
            self.num_consecutive_iterations[seq_id] += 1
        
        for seq_id in list(self.num_consecutive_iterations.keys()):
            if seq_id not in running_seq_ids:
                self.num_consecutive_iterations[seq_id] = 0
    
    def _free_seq(self, seq: Sequence) -> None:
        super()._free_seq(seq)

        if seq.seq_id in self.num_consecutive_iterations:
            del self.num_consecutive_iterations[seq.seq_id]
        if seq.seq_id in self.request_quantum_map:
            self.decode_queues[self.request_quantum_map[seq.seq_id]].remove(seq)
            del self.request_quantum_map[seq.seq_id]
        
    def _schedule_decodes(self, running_decodes: List[Sequence], now: float):
        running = []
        begin_swap_in_seq_ids = []
        begin_swap_out_seq_ids = []
        scheduled_seq_id_metadata_list = []
        num_batched_tokens = 0

        self._update_quantums()

        # At this point, running_decodes could include some previously finished prefills
        # It can also include recently swapped in requests
        # It's also in FCFS order
        for seq in running_decodes:
            if seq.seq_id not in self.request_quantum_map:
                quantum_idx = self._get_quantum(0)  # Always 0 quantum
                self.decode_queues[quantum_idx].append(seq)
                self.request_quantum_map[seq.seq_id] = quantum_idx
        
        # Here, we're sorting all of our requests in order of quantum
        queue: List[Sequence] = []
        for seqs in self.decode_queues:
            queue.extend(seqs)
    
        while queue:
            seq = queue.pop(0)

            if not seq.is_paused() and not seq.is_swapped_out():
                assert seq.is_swapping_out() or seq.is_swapping_in()
                continue

            def can_schedule():
                if seq.is_paused():
                    return self.block_manager.can_append_slot()
                elif seq.is_swapped_out():
                    return self.block_manager.can_swap_in(seq.seq_id)
                else:
                    raise ValueError(f"Invalid sequence status: {seq.get_status()}")

            while not can_schedule():
                # Need to search for the lowest priority sequence actually running
                victim_idx = len(queue) - 1
                while victim_idx >= 0:
                    if queue[victim_idx].is_paused():
                        break
                    victim_idx -= 1

                if victim_idx >= 0:
                    victim_seq = queue.pop(victim_idx)
                    self._begin_swap_out(victim_seq)
                    begin_swap_out_seq_ids.append(victim_seq.seq_id)
                else:
                    if seq.is_paused():
                        self._begin_swap_out(seq)
                        begin_swap_out_seq_ids.append(seq.seq_id)
                    break
            else:
                if seq.is_paused():
                    # Append new slots to the sequence group.
                    self._append_slot(seq)
                    running.append(seq)
                    num_batched_tokens += 1
                    scheduled_seq_id_metadata_list.append(
                        SequenceScheduleMetadata.from_sequence(seq)
                    )
                elif seq.is_swapped_out():
                    self._begin_swap_in(seq)
                    begin_swap_in_seq_ids.append(seq.seq_id)
        
        self._update_num_consecutive_iterations(running)

        return (
            running,
            [],
            [],
            begin_swap_in_seq_ids,
            begin_swap_out_seq_ids,
            scheduled_seq_id_metadata_list
        )