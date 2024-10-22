from typing import Dict, List, Tuple

from sarathi.config import SystemConfig
from sarathi.core.block_space_manager.base_block_space_manager import BlockDevice
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
        if self.block_manager.is_allocated_in_gpu(seq_id) or self.block_manager.is_allocated_in_cpu(seq_id):
            self.block_manager.free(seq_id)
        super()._free_seq(seq_id)

    def _preempt_seq(self, seq_id: str) -> None:
        super()._preempt_seq(seq_id)
        self.block_manager.free(seq_id)
    
    def _begin_swap_in_seq(self, seq_id: str) -> None:
        super()._begin_swap_in_seq(seq_id)
        self.block_manager.begin_swap_in(seq_id)
    
    def _begin_swap_out_seq(self, seq_id: str) -> None:
        super()._begin_swap_out_seq(seq_id)
        self.block_manager.begin_swap_out(seq_id)
    
    def _finish_swap_in_seq(self, seq_id: str) -> None:
        super()._finish_swap_in_seq(seq_id)
        self.block_manager.finish_swap_in(seq_id)
    
    def _finish_swap_out_seq(self, seq_id: str) -> None:
        super()._finish_swap_out_seq(seq_id)
        self.block_manager.finish_swap_out(seq_id)
    
    def _on_seq_scheduled(self, seq_sched_metadata: SequenceScheduleMetadata) -> None:
        assert seq_sched_metadata.seq_id in self.seq_map
        seq = self.seq_map[seq_sched_metadata.seq_id]

        if seq.is_waiting():
            assert len(seq.prompt_token_ids) > 0 and len(seq.output_token_ids) == 0
            # print(f"Trying to schedule {seq.seq_id}, free blocks: {self.block_manager.allocators[BlockDevice.GPU].get_num_free_blocks()}, required blocks: {len(seq.logical_token_blocks)}")
            assert self.block_manager.can_allocate(seq, BlockDevice.GPU)
            self.block_manager.allocate(seq, BlockDevice.GPU)
        elif not seq_sched_metadata.is_prompt:
            self.block_manager.can_append_slot(BlockDevice.GPU)
            self.block_manager.append_slot(seq, BlockDevice.GPU)
        
        # NOTE: Here, we assume that in chunked prefill mode, the full sequence is allocated,
        # which means in later chunks, we don't need to allocate. But when decoding, we do need to append slots.

        super()._on_seq_scheduled(seq_sched_metadata)  # This just sets the status to resumed

    def _on_append_token(self, seq: Sequence) -> None:
        # the engine performs detokenization at this point
        # but we don't need to do anything here on worker side
        pass

    def get_gpu_block_table(self, seq_id: str) -> List[int]:
        return self.block_manager.get_gpu_block_table(seq_id)

    def get_cpu_block_table(self, seq_id: str) -> List[int]:
        return self.block_manager.get_cpu_block_table(seq_id)
    
    def get_swap_out_mappings(self, swap_out_seq_ids: List[str]) -> Dict[str, List[int]]:
        return {
            seq_id: self.block_manager.get_swap_out_mapping(seq_id)
            for seq_id in swap_out_seq_ids
        }

    def get_swap_in_mappings(self, swap_in_seq_ids: List[str]) -> Dict[str, List[int]]:
        return {
            seq_id: self.block_manager.get_swap_in_mapping(seq_id)
            for seq_id in swap_in_seq_ids
        }
    