"""A block manager that manages token blocks, but without materializing any block tables - just a dry run for the scheduler to simulate what happens."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Union

from sarathi.core.block_space_manager import BlockDevice
from sarathi.core.datatypes.sequence import Sequence


class BaseBlockAllocator:
    """Manages the number of free physical token blocks for a device."""

    def __init__(
        self,
        num_blocks: int,
        watermark: Optional[float] = None,
    ) -> None:
        self.__num_blocks = num_blocks - (0 if watermark is None else int(watermark * num_blocks))
        self.__num_blocks_allocated = 0

    @property
    def num_total_blocks(self) -> int:
        return self.__num_blocks
    
    @property
    def num_blocks_allocated(self) -> int:
        return self.__num_blocks_allocated
    
    @property
    def num_free_blocks(self) -> int:
        return self.__num_blocks - self.__num_blocks_allocated

    def can_allocate(self, num_blocks: int) -> bool:
        return self.num_free_blocks >= num_blocks

    def allocate(self, num_blocks: int) -> int:
        if not self.can_allocate(num_blocks):
            raise ValueError("Out of memory! Not enough free blocks are available.")
        self.__num_blocks_allocated += num_blocks
        return num_blocks

    def free(self, num_blocks: int) -> None:
        if num_blocks > self.__num_blocks_allocated:
            raise ValueError("Invalid number of blocks to free.")
        self.__num_blocks_allocated -= num_blocks


class DryRunBlockAllocator(BaseBlockAllocator):
    pass


class BlockAllocator(BaseBlockAllocator):
    def __init__(self, num_blocks: int, watermark: Optional[float] = None) -> None:
        super().__init__(num_blocks, watermark=watermark)
        self.__free_blocks = list(range(self.num_total_blocks))
    
    def allocate(self, num_blocks: int) -> List[int]:
        super().allocate(num_blocks)
        result = deepcopy(self.__free_blocks[-num_blocks:])
        del self.__free_blocks[-num_blocks:]
        self.__ensure_consistent_with_parent()
        return result
    
    def free(self, blocks: List[int]) -> None:
        self.__free_blocks.extend(blocks)
        self.__ensure_consistent_with_parent()
    
    def __ensure_consistent_with_parent(self):
        assert self.num_free_blocks == len(self.__free_blocks)


class BaseBlockSpaceManager(ABC):
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        max_model_len: int,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.max_model_len = max_model_len

        assert watermark >= 0.0

        self._allocators = {
            BlockDevice.GPU: self._init_allocator(num_gpu_blocks, watermark=watermark),
            BlockDevice.CPU: self._init_allocator(num_cpu_blocks, watermark=None),
        }

        # Mapping: seq_id -> device -> available blocks
        self._block_tables: Dict[str, Dict[BlockDevice, Union[List[int], int]]] = {}

    @abstractmethod
    def _init_allocator(self, num_blocks: int, watermark: Optional[float] = None) -> Union[DryRunBlockAllocator, BlockAllocator]:
        raise NotImplementedError

    ########## Read-only methods ##########

    def get_num_free_blocks(self, device: BlockDevice) -> int:
        return self._allocators[device].num_free_blocks

    def get_seq_num_blocks_allocated(self, seq_id: str, device: BlockDevice) -> int:
        assert isinstance(seq_id, str)
        assert seq_id in self._block_tables
        if device not in self._block_tables[seq_id]:
            return 0
        assert isinstance(self._block_tables[seq_id][device], list) or isinstance(self._block_tables[seq_id][device], int)
        return len(self._block_tables[seq_id][device]) if isinstance(self._block_tables[seq_id][device], list) else self._block_tables[seq_id][device]

    def is_allocated_in_gpu(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        assert seq_id in self._block_tables
        return BlockDevice.GPU in self._block_tables[seq_id]

    def is_allocated_in_cpu(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        assert seq_id in self._block_tables
        return BlockDevice.CPU in self._block_tables[seq_id]
    
    def can_allocate(self, seq: Sequence, device: BlockDevice = BlockDevice.GPU) -> bool:
        """Check if we can allocate memory for the entire sequence."""
        assert isinstance(seq, Sequence)
        return self._allocators[device].can_allocate(seq.get_num_logical_blocks())

    def can_append_slot(self, seq: Sequence) -> bool:
        assert isinstance(seq, Sequence)
        num_blocks_to_allocate = seq.get_num_logical_blocks() - self.get_seq_num_blocks_allocated(seq.seq_id, BlockDevice.GPU)
        assert num_blocks_to_allocate <= 1
        return self._allocators[BlockDevice.GPU].can_allocate(num_blocks_to_allocate)

    def can_swap_in(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        assert seq_id in self._block_tables and BlockDevice.GPU not in self._block_tables[seq_id] and BlockDevice.CPU in self._block_tables[seq_id]
        return self.get_num_free_blocks(BlockDevice.GPU) >= self.get_seq_num_blocks_allocated(seq_id, BlockDevice.CPU)
    
    def can_swap_in_and_append_slot(self, seq_id: str, num_logical_blocks: int) -> bool:
        assert isinstance(seq_id, str)
        return self.get_num_free_blocks(BlockDevice.GPU) >= num_logical_blocks

    def can_swap_out(self, seq_id: str) -> bool:
        assert isinstance(seq_id, str)
        assert seq_id in self._block_tables and BlockDevice.CPU not in self._block_tables[seq_id] and BlockDevice.GPU in self._block_tables[seq_id]
        return self.get_num_free_blocks(BlockDevice.CPU) >= self.get_seq_num_blocks_allocated(seq_id, BlockDevice.GPU)
    
    ########## Allocations/Swaps ##########

    def allocate(self, seq: Sequence, initial_device: BlockDevice = BlockDevice.GPU) -> None:
        """Allocates new physical token blocks that will store the prompt tokens. Almost always on GPU."""
        assert isinstance(seq, Sequence)
        assert seq.seq_id not in self._block_tables
        self._block_tables[seq.seq_id] = {
            initial_device: self._allocators[initial_device].allocate(seq.get_num_logical_blocks())
        }
    
    def append_slot(self, seq: Sequence) -> None:
        """Allocate a physical slot for a new token."""
        assert isinstance(seq, Sequence)
        num_blocks_to_allocate = seq.get_num_logical_blocks() - self.get_seq_num_blocks_allocated(seq.seq_id, BlockDevice.GPU)
        assert num_blocks_to_allocate <= 1
        if num_blocks_to_allocate > 0:
            self._block_tables[seq.seq_id][BlockDevice.GPU] += self._allocators[BlockDevice.GPU].allocate(num_blocks_to_allocate)
    
    def begin_swap_in(self, seq_id: str):
        assert isinstance(seq_id, str)
        assert self.can_swap_in(seq_id)

        self._block_tables[seq_id][BlockDevice.GPU] = self._allocators[BlockDevice.GPU].allocate(
            self.get_seq_num_blocks_allocated(seq_id, BlockDevice.CPU)
        )
        assert self.get_seq_num_blocks_allocated(seq_id, BlockDevice.CPU) == self.get_seq_num_blocks_allocated(seq_id, BlockDevice.GPU)
        self._update_swap_in_mapping(seq_id)

    def finish_swap_in(self, seq_id: str):
        assert isinstance(seq_id, str)
        self._free_device_blocks(seq_id, BlockDevice.CPU)

    def swap_out(self, seq_id: str):
        assert isinstance(seq_id, str)
        assert self.can_swap_out(seq_id)

        self._block_tables[seq_id][BlockDevice.CPU] = self._allocators[BlockDevice.CPU].allocate(
            self.get_seq_num_blocks_allocated(seq_id, BlockDevice.GPU)
        )
        assert self.get_seq_num_blocks_allocated(seq_id, BlockDevice.GPU) == self.get_seq_num_blocks_allocated(seq_id, BlockDevice.CPU)
        self._update_swap_out_mapping(seq_id)
        self._free_device_blocks(seq_id, BlockDevice.GPU)
    
    @abstractmethod
    def _update_swap_in_mapping(self, seq_id: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _update_swap_out_mapping(self, seq_id: str) -> None:
        raise NotImplementedError

    ########## Frees ##########

    def _free_device_blocks(self, seq_id: str, device: BlockDevice) -> None:
        assert isinstance(seq_id, str)
        self._allocators[device].free(self._block_tables[seq_id][device])
        del self._block_tables[seq_id][device]

    def _free_block_table(self, seq_id: str) -> None:
        assert isinstance(seq_id, str)
        if seq_id not in self._block_tables:
            # Already freed or haven't been scheduled yet.
            return
        devices = list(self._block_tables[seq_id].keys())
        for device in devices:
            self._free_device_blocks(seq_id, device)
        del self._block_tables[seq_id]

    def free(self, seq_id: str) -> None:
        assert isinstance(seq_id, str)
        self._free_block_table(seq_id)

    def reset(self) -> None:
        for seq_id in self._block_tables.keys():
            self._free_block_table(seq_id)
    
    ########## Metadata ##########

    def get_block_table_metadata_str(self) -> str:
        result_gpu = []
        result_cpu = []
        for seq_id in self._block_tables.keys():
            if BlockDevice.GPU in self._block_tables[seq_id]:
                result_gpu.append((seq_id, self.get_seq_num_blocks_allocated(seq_id, BlockDevice.GPU)))
            if BlockDevice.CPU in self._block_tables[seq_id]:
                result_cpu.append((seq_id, self.get_seq_num_blocks_allocated(seq_id, BlockDevice.CPU)))

        return '\n'.join([
            f"Num free GPU blocks: {self._allocators[BlockDevice.GPU].get_num_free_blocks()}",
            f"Num free CPU blocks: {self._allocators[BlockDevice.CPU].get_num_free_blocks()}",
            f"All GPU block table lens: {result_gpu}",
            f"All CPU block table lens: {result_cpu}"])
        

class DryRunBlockSpaceManager(BaseBlockSpaceManager):
    def _init_allocator(self, num_blocks: int, watermark: Optional[float] = None) -> DryRunBlockAllocator:
        return DryRunBlockAllocator(num_blocks, watermark=watermark)
    
    def begin_swap_in(self, seq_id: str) -> None:
        assert isinstance(seq_id, str)
        assert self.can_swap_in(seq_id)

        self._block_tables[seq_id][BlockDevice.GPU] = self._allocators[BlockDevice.GPU].allocate(
            self.get_seq_num_blocks_allocated(seq_id, BlockDevice.CPU)
        )
    
    def finish_swap_in(self, seq_id: str) -> None:
        assert isinstance(seq_id, str)
        self._free_device_blocks(seq_id, BlockDevice.CPU)
    
    def swap_out(self, seq_id: str) -> None:
        assert isinstance(seq_id, str)
        assert self.can_swap_out(seq_id)
        self._block_tables[seq_id][BlockDevice.CPU] = self._allocators[BlockDevice.CPU].allocate(
            self.get_seq_num_blocks_allocated(seq_id, BlockDevice.GPU)
        )
        self._free_device_blocks(seq_id, BlockDevice.GPU)
    
    def _update_swap_in_mapping(self, seq_id: str) -> None:
        pass

    def _update_swap_out_mapping(self, seq_id: str) -> None:
        pass
    

class BlockSpaceManager(BaseBlockSpaceManager):

    def __init__(self, block_size: int, num_gpu_blocks: int, num_cpu_blocks: int, max_model_len: int, watermark: float = 0.01) -> None:
        super().__init__(block_size, num_gpu_blocks, num_cpu_blocks, max_model_len, watermark=watermark)
        self.__swap_in_mapping: Dict[str, List[int]] = {}
        self.__swap_out_mapping: Dict[str, List[int]] = {}

    def _init_allocator(self, num_blocks: int, watermark: Optional[float] = None) -> BlockAllocator:
        return BlockAllocator(num_blocks, watermark=watermark)

    def _update_swap_in_mapping(self, seq_id: str) -> None:
        self.__swap_in_mapping[seq_id] = list(zip(self._block_tables[seq_id][BlockDevice.CPU], self._block_tables[seq_id][BlockDevice.GPU]))
    
    def _update_swap_out_mapping(self, seq_id: str) -> None:
        self.__swap_out_mapping[seq_id] = list(zip(self._block_tables[seq_id][BlockDevice.GPU], self._block_tables[seq_id][BlockDevice.CPU]))

    def get_swap_in_mapping(self, seq_id: str) -> List[int]:
        assert isinstance(seq_id, str)
        return self.__swap_in_mapping[seq_id]
    
    def get_swap_out_mapping(self, seq_id: str) -> List[int]:
        assert isinstance(seq_id, str)
        return self.__swap_out_mapping[seq_id]
    
    ########## Metadata ##########
    
    def get_gpu_block_table(self, seq_id: str) -> List[int]:
        assert isinstance(seq_id, str) and seq_id in self._block_tables
        if BlockDevice.GPU not in self._block_tables[seq_id]:
            return []
        return self._block_tables[seq_id][BlockDevice.GPU]
