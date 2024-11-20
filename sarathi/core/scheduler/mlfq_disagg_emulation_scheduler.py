from typing import List

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    MLFQDisaggEmulationSchedulerConfig
)
from sarathi.core.block_space_manager import BlockDevice
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
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

        self.decode_queues: List[List[Sequence]] = [[] for _ in range(len(self.quantums))]
        self.request_quantum_map = {}
        self.priorities = {}
        self.last_iteration_ran = {}

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        assert not seq.is_finished()
        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_processed(),
            self.scheduler_config.chunk_size - num_batched_tokens,
        )

        return next_num_tokens

    def _schedule_prefills(self, running_prefills: List[Sequence], running_decodes: List[Sequence], now: float):
        # NOTE: Both lists sorted in FCFS order 
        running = [] # NOTE: running decodes, doesn't strictly have to come first in order
        ignored_seq_ids = []
        swap_out_seq_ids = []
        scheduled_seq_id_metadata_list = []

        num_batched_tokens = 0

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
        
        queue = self._update_and_get_queue(running_decodes)
        queue = list(filter(lambda seq: seq.is_paused(), queue))

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
            while num_required_blocks > self.block_manager.get_num_free_blocks(BlockDevice.GPU):
                if not queue:
                    break
                decode_seq = queue[-1]
                num_blocks_allocated = self.block_manager.get_seq_num_blocks_allocated(decode_seq.seq_id, BlockDevice.GPU)
                total_cpu_blocks_required += num_blocks_allocated
                if total_cpu_blocks_required > self.block_manager.get_num_free_blocks(BlockDevice.CPU):
                    break
                num_required_blocks -= num_blocks_allocated
                running_decodes_to_swap_out.append(decode_seq)
                queue.pop()
            
            if num_required_blocks > self.block_manager.get_num_free_blocks(BlockDevice.GPU):
                # Restore state
                queue.extend(reversed(running_decodes_to_swap_out))
                break

            for seq in running_decodes_to_swap_out:
                self._swap_out(seq)
            swap_out_seq_ids.extend(seq.seq_id for seq in running_decodes_to_swap_out)
        
            seq = self.waiting.pop(0)
            self._allocate(seq)
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_id_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)
        
        running.extend(queue)
        
        return (
            running,
            ignored_seq_ids,
            [],
            swap_out_seq_ids,
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
    
    def _update_priorities(self, running: List[Sequence]):
        # NOTE: We use increments because we want to reset priorities
        for seq in running:
            self.priorities[seq.seq_id] += 1

        for quantum_idx in range(len(self.quantums)):
            for seq in self.decode_queues[quantum_idx]:
                if (
                    seq.seq_id in self.last_iteration_ran and 
                    self.scheduler_config.starvation_limit is not None and
                    self._iteration_id - self.last_iteration_ran[seq.seq_id] > self.scheduler_config.starvation_limit
                ):
                    self.priorities[seq.seq_id] = 0

    def _update_quantums(self):
        for quantum_idx in reversed(range(len(self.quantums))):
            indices_to_remove = []
            for i in range(len(self.decode_queues[quantum_idx])):
                seq = self.decode_queues[quantum_idx][i]
                next_quantum = self._get_quantum(self.priorities[seq.seq_id])
                if next_quantum != quantum_idx:
                    # print(f"Moving {seq.seq_id} from quantum {quantum_idx} to {next_quantum} since run count is {seq.get_output_len()}")
                    self.decode_queues[next_quantum].append(seq)
                    self.request_quantum_map[seq.seq_id] = next_quantum
                    indices_to_remove.append(i)
            for i in reversed(indices_to_remove):
                self.decode_queues[quantum_idx].pop(i)
    
    def _free_seq(self, seq: Sequence) -> None:
        super()._free_seq(seq)

        if seq.seq_id in self.request_quantum_map:
            self.decode_queues[self.request_quantum_map[seq.seq_id]].remove(seq)
            del self.request_quantum_map[seq.seq_id]
            del self.priorities[seq.seq_id]
            del self.last_iteration_ran[seq.seq_id]
    
    def _update_and_get_queue(self, running_decodes: List[Sequence]):
        self._update_quantums()

        # At this point, running_decodes could include some previously finished prefills
        # It can also include recently swapped in requests
        # It's also in FCFS order
        for seq in running_decodes:
            if seq.seq_id not in self.request_quantum_map:
                quantum_idx = self._get_quantum(0)  # Always 0 quantum
                self.decode_queues[quantum_idx].append(seq)
                self.request_quantum_map[seq.seq_id] = quantum_idx
                self.priorities[seq.seq_id] = 0
        
        # Here, we're sorting all of our requests in order of quantum
        queue: List[Sequence] = []
        for seqs in self.decode_queues:
            queue.extend(seqs)
        
        queue = list(filter(lambda seq: not seq.is_swapping_in(), queue))
        for seq in queue:
            assert seq.is_prompt_processing_finished()
        
        return queue
    
    def _schedule_decodes(self, running_decodes: List[Sequence], now: float):
        running = []
        swap_out_seq_ids = []
        swap_out_lens = []
        swap_in_seq_ids = []
        scheduled_seq_id_metadata_list = []
        num_batched_tokens = 0

        queue = self._update_and_get_queue(running_decodes)

        i = 0
        while i < len(queue):
            seq = queue[i]
            assert seq.is_paused() or seq.is_swapped_out(), f"Sequence {seq.seq_id} is in an invalid state: {seq.get_status()}"

            # Swap lowest priority requests in running list
            num_required_blocks = seq.get_num_logical_blocks() - self.block_manager.get_seq_num_blocks_allocated(seq.seq_id, BlockDevice.GPU)
            total_cpu_blocks_required = 0
            running_decodes_to_swap_out = []
            j = len(queue) - 1
            while num_required_blocks > self.block_manager.get_num_free_blocks(BlockDevice.GPU):
                assert j >= i
                if j == i:
                    break
                decode_seq = queue[j]
                num_blocks_allocated = self.block_manager.get_seq_num_blocks_allocated(decode_seq.seq_id, BlockDevice.GPU)
                if not num_blocks_allocated:
                    j -= 1
                    continue

                blocks_to_swap = min(num_blocks_allocated, num_required_blocks) if self.cache_config.partial_swap_out else num_blocks_allocated
                if not self.cache_config.duplicate_kv_cache:
                    total_cpu_blocks_required += blocks_to_swap
                    if total_cpu_blocks_required > self.block_manager.get_num_free_blocks(BlockDevice.CPU):
                        break
                num_required_blocks -= blocks_to_swap
                running_decodes_to_swap_out.append((decode_seq, blocks_to_swap))
                j -= 1
            
            if num_required_blocks <= self.block_manager.get_num_free_blocks(BlockDevice.GPU):
                queue = queue[:j+1]
            else:
                # NOTE: We allow skipping to the next sequence if we can't fit the current one
                i += 1
                continue
            
            # Swap the sequences we promised to swap out
            for seq_to_swap, num_blocks_to_swap in running_decodes_to_swap_out:
                print(f"Iteration {self._iteration_id}: Swapping out {num_blocks_to_swap} blocks in sequence {seq_to_swap.seq_id}")
                if self.cache_config.partial_swap_out:
                    self._swap_out(seq_to_swap, num_blocks_to_swap)
                    swap_out_lens.append(num_blocks_to_swap)
                else:
                    self._swap_out(seq_to_swap)
                swap_out_seq_ids.append(seq_to_swap.seq_id)
            
            assert seq.is_paused() or seq.is_swapped_out(), f"Sequence {seq.seq_id} is in an invalid state: {seq.get_status()}"

            if seq.is_swapped_out():
                print(f"Iteration {self._iteration_id}: Swapping in {seq.seq_id}")
                assert self.block_manager.can_swap_in_and_append_slot(
                    seq.seq_id,
                    seq.get_num_logical_blocks()
                )
                self._swap_in(seq)
                swap_in_seq_ids.append(seq.seq_id)
            
            if seq.is_paused() or (seq.is_swapped_out() and not self.cache_config.async_swap_in):
                print(f"Iteration {self._iteration_id}: Scheduling {seq.seq_id}")
                # Append new slots to the sequence group.
                self._append_slot(seq)
                running.append(seq)
                num_batched_tokens += 1
                scheduled_seq_id_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
                self.last_iteration_ran[seq.seq_id] = self._iteration_id
            
            i += 1

        self._update_priorities(running)
        
        return (
            running,
            [],
            [],
            swap_out_seq_ids,
            swap_out_lens,
            swap_in_seq_ids,
            scheduled_seq_id_metadata_list
        )