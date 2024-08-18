from typing import List, Tuple

from sarathi.config import SystemConfig
from sarathi.core.block_space_manager.block_space_manager_registry import (
    BlockSpaceManagerRegistry,
)
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.sequence_manager.base_sequence_manager import BaseSequenceManager


class WorkerSequenceManager(BaseSequenceManager):

    def __init__(
        self,
        config: SystemConfig,
    ):
        super().__init__(config)
        # we will have a clone of block manager here, it is supposed
        # to work in sync block manager in scheduler the idea is to avoid
        # sending block table every time to the worker
        self.block_manager = BlockSpaceManagerRegistry.get(
            config.scheduler_config.get_type(),
            config.cache_config.block_size,
            config.cache_config.num_gpu_blocks,
            config.cache_config.num_cpu_blocks,
            config.model_config.max_model_len,
        )

    def _free_seq(self, seq_id: str) -> None:
        # ignored sequences might not have been allocated
        assert seq_id in self.seq_map
        seq = self.seq_map[seq_id]
        if self.block_manager.is_allocated_in_gpu(seq) or self.block_manager.is_allocated_in_cpu(seq):
            self.block_manager.free(seq)
        super()._free_seq(seq_id)

    def _preempt_seq(self, seq_id: str) -> None:
        super()._preempt_seq(seq_id)
        seq = self.seq_map[seq_id]
        self.block_manager.free(seq)
    
    def _swap_seq(self, seq_id: str) -> None:
        super()._swap_seq(seq_id)
        seq = self.seq_map[seq_id]
        self.block_manager.swap_out(seq)
    
    def _on_seq_scheduled(self, seq_sched_metadata: SequenceScheduleMetadata) -> None:
        assert seq_sched_metadata.seq_id in self.seq_map
        seq = self.seq_map[seq_sched_metadata.seq_id]
        needs_swapping_in = seq.is_swapped()

        super()._on_seq_scheduled(seq_sched_metadata)  # This just sets the status to resumed

        # Handle new/preempted request first - simple case. This is lazy allocation.
        if not self.block_manager.is_allocated_in_gpu(seq) and not self.block_manager.is_allocated_in_cpu(seq):
            assert not needs_swapping_in
            assert len(seq.prompt_token_ids) > 0 and len(seq.output_token_ids) == 0
            assert self.block_manager.can_allocate(seq) # GPU by default
            self.block_manager.allocate(seq)
            return
        
        # If the request was swapped out, let's update the block numbers in GPU first
        if self.block_manager.is_allocated_in_cpu(seq):
            assert needs_swapping_in
            self.block_manager.swap_in(seq)
        
        # All prefills are already allocated, so we only need to append slots to decode
        # Note that if a decode request was swapped in, we update the block numbers in GPU first, then append slots
        if not seq_sched_metadata.is_prompt:
            self.block_manager.can_append_slot()
            self.block_manager.append_slot(seq)


    def _on_append_token(self, seq: Sequence) -> None:
        # the engine performs detokenization at this point
        # but we don't need to do anything here on worker side
        pass

    def _get_gpu_block_table(self, seq: Sequence) -> List[int]:
        return self.block_manager.get_gpu_block_table(seq)

    def get_and_clear_swap_mappings(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        return self.block_manager.get_and_clear_swap_mappings()